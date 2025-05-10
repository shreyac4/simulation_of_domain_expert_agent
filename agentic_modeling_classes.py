import os
import re
import boto3
import time
import logging
import random
import torch
from datetime import datetime
from dotenv import load_dotenv
from mesa import Agent, Model
from mesa.time import RandomActivation
from collections import Counter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document, AIMessage
from langchain_anthropic import ChatAnthropic
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from safetensors import safe_open
#from openai import OpenAI  
import anthropic
from typing import List, Optional, Dict, Any
import json
from safetensors import safe_open
import platform
import tempfile
import pypdf
from decimal import Decimal
from collections import Counter
from huggingface_hub import InferenceClient
from together import Together
import streamlit as st
import asyncio
import difflib
import requests

#!pip install boto3 python-dotenv mesa langchain langchain_huggingface langchain_anthropic anthropic langchain-community faiss-cpu pypdf
#!pip install -U langchain-community
# Loading environment variables
load_dotenv()
#os.environ['OPENAI_API_KEY'] = os.getenv('TOGETHER_API_KEY')

# Creating a function to safely access credentials
# Getting credentials either from Streamlit secrets or environment variables
def get_credentials():
    try:
        # First try environment variables (for local dev)
        env_vars = {
            "aws_access_key_id": os.getenv('AWS_ACCESS_KEY_ID'),
            "aws_secret_access_key": os.getenv('AWS_SECRET_ACCESS_KEY'),
            "region_name": os.getenv('AWS_REGION', 'us-east-1'),
            "anthropic_api_key": os.getenv('ANTHROPIC_API_KEY'),
            "huggingface_api_key": os.getenv('HUGGINGFACE_API_KEY'),
            "together_api_key": os.getenv('TOGETHER_API_KEY')
        }
        
        # Return env vars for local development
        return env_vars
    except Exception as e:
        logger.warning(f"Failed to get credentials: {e}")
        return None

# Functions to get AWS clients with credentials
def get_s3_client(credentials=None):
    """Get S3 client with credentials"""
    if credentials and credentials.get("aws_access_key_id") and credentials.get("aws_secret_access_key"):
        return boto3.client(
            's3',
            region_name=credentials.get("region_name", "us-east-1"),
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"]
        )
    else:
        # Use default credentials (from .aws/credentials or EC2 role)
        return boto3.client('s3', region_name="us-east-1")

def get_dynamodb_resource(credentials=None):
    """Get DynamoDB resource with credentials"""
    if credentials and credentials.get("aws_access_key_id") and credentials.get("aws_secret_access_key"):
        return boto3.resource(
            'dynamodb',
            region_name=credentials.get("region_name", "us-east-1"),
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"]
        )
    else:
        # Use default credentials (from .aws/credentials or EC2 role)
        return boto3.resource('dynamodb', region_name="us-east-1")

def get_cloudwatch_client(credentials=None):
    """Get CloudWatch client with credentials"""
    if credentials and credentials.get("aws_access_key_id") and credentials.get("aws_secret_access_key"):
        return boto3.client(
            'cloudwatch',
            region_name=credentials.get("region_name", "us-east-1"),
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"]
        )
    else:
        # Use default credentials (from .aws/credentials or EC2 role)
        return boto3.client('cloudwatch', region_name="us-east-1")

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defining a RAG metrics calculation and logging class to DynamoDB and Cloudwatch
# Handles conversion of Python data types to DynamoDB-compatible formats 
class MetricsLogger:
    def __init__(self, dynamodb_resource=None, cloudwatch_client=None):
        self.dynamodb = dynamodb_resource or boto3.resource('dynamodb', region_name='us-east-1')
        self.cloudwatch = cloudwatch_client or boto3.client('cloudwatch', region_name='us-east-1')
        self.metrics_table = self.get_or_create_metrics_table()
    
    # Function for recursively converting float values to Decimal for DynamoDB
    def float_to_decimal(self, obj):
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: self.float_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.float_to_decimal(item) for item in obj]
        else:
            return obj
        
    
    # Gets existing DynamoDB table or creates a new one if it doesn't exist
    # Table schema includes session_id as partition key and timestamp as sort key
    def get_or_create_metrics_table(self):
        table_name = 'rag_evaluation_metrics'
        try:
            # Checking if table exists
            existing_tables = self.dynamodb.meta.client.list_tables()["TableNames"]
            if table_name not in existing_tables:
                table = self.dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=[
                        {'AttributeName': 'session_id', 'KeyType': 'HASH'},
                        {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'session_id', 'AttributeType': 'S'},
                        {'AttributeName': 'timestamp', 'AttributeType': 'S'}
                    ],
                    ProvisionedThroughput={
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                )
                table.wait_until_exists()
                return table
            return self.dynamodb.Table(table_name)
        except Exception as e:
            logger.error(f"Error creating/accessing metrics table: {e}")
            raise

    # Storing detailed evaluation metrics in DynamoDB
    def log_to_dynamodb(self, session_id: str, metrics_data: Dict):
        try:
            # Building the item dictionary for DynamoDB
            item = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'interaction_metrics': {
                    'M': {
                        'average_retrieval_time': {'N': str(Decimal(str(metrics_data['average_retrieval_time'])))},
                        'average_response_time': {'N': str(Decimal(str(metrics_data['average_response_time'])))},
                        'total_questions': {'N': str(Decimal(str(metrics_data['total_questions_asked'])))},
                        'context_coverage': {'N': str(Decimal(str(metrics_data['context_coverage'])))},
                        'context_utilization': {'N': str(Decimal(str(metrics_data['average_context_utilization'])))},
                    }
                },
                'qa_pairs': {
                    'L': [
                        {
                            'M': {
                                'question': {'S': str(qa['question'])},
                                'user_answer': {'S': str(qa['user_answer'])},
                                'ground_truth': {'S': str(qa['ground_truth'])},
                                'similarity_score': {'N': str(Decimal(str(qa['similarity_score'])))},
                            }
                        }
                        for qa in metrics_data['qa_pairs']
                    ]
                },
                'metadata': {
                    'M': {
                        'session_date': {'S': datetime.now().strftime('%Y-%m-%d')},
                        'session_time': {'S': datetime.now().strftime('%H:%M:%S')},
                        'session_duration': {'N': str(Decimal(str(
                            float(metrics_data['average_response_time']) * 
                            float(metrics_data['total_questions_asked'])
                        )))},
                    }
                }
            }

            # Logging to DynamoDB
            self.metrics_table.put_item(Item=item)
            logger.info(f"Metrics logged to DynamoDB for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to log metrics to DynamoDB: {e}")


    # Logging performance metrics to CloudWatch for real-time monitoring
    def log_to_cloudwatch(self, metrics_data: Dict):
        try:
            self.cloudwatch.put_metric_data(
                Namespace='RAGSystem',
                MetricData=[
                    {
                        'MetricName': 'RetrievalTime',
                        'Value': float(metrics_data['average_retrieval_time']),
                        'Unit': 'Seconds'
                    },
                    {
                        'MetricName': 'ResponseTime',
                        'Value': float(metrics_data['average_response_time']),
                        'Unit': 'Seconds'
                    },
                    {
                        'MetricName': 'ContextCoverage',
                        'Value': float(metrics_data['context_coverage'] * 100),
                        'Unit': 'Percent'
                    },
                    {
                        'MetricName': 'ContextUtilization',
                        'Value': float(metrics_data['average_context_utilization'] * 100),
                        'Unit': 'Percent'
                    },
                    {
                        'MetricName': 'QuestionsAsked',
                        'Value': float(metrics_data['total_questions_asked']),
                        'Unit': 'Count'
                    }
                ]
            )
            logger.info("Metrics logged to CloudWatch")
        except Exception as e:
            logger.error(f"Failed to log metrics to CloudWatch: {e}")

# Handles document processing and storage in AWS
class DocumentProcessor:

    def __init__(self, raw_bucket: str, processed_bucket: str, s3_client=None):
        self.raw_bucket = raw_bucket
        self.processed_bucket = processed_bucket
        self.s3_client = s3_client or get_s3_client()

        # Initializing text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    # This function attempts to load an existing FAISS index from S3.
    # If no index exists, it creates a new one from processed document chunks.
    # Function returns FAISS vector store for similarity search
    def get_vector_store(self, embeddings=None, new_documents: List[Document] = None, force_rebuild=False):
        if embeddings is None:
            embeddings = self.embeddings
        
        # Handle force rebuild first if requested
        if force_rebuild:
            logger.info("Forcing rebuild of vector store index")
            chunks = self.load_chunks_from_processed_bucket()
            if new_documents and len(new_documents) > 0:
                if not chunks:
                    chunks = []
                chunks.extend(new_documents)
            
            if not chunks or len(chunks) == 0:
                raise ValueError("No chunks available to create vector store")
                
            vector_store = FAISS.from_documents(chunks, embeddings)
            # Save to S3
            data = vector_store.serialize_to_bytes()
            self.s3_client.put_object(Bucket=self.processed_bucket, Key="vector_store/faiss_index.pickle", Body=data)
            logger.info("New FAISS index created and saved to S3.")
            return vector_store

        try:
            # Check if an index file already exists in S3
            try:
                self.s3_client.head_object(Bucket=self.processed_bucket, Key="vector_store/faiss_index.pickle")
                index_exists = True
            except Exception:
                index_exists = False

            if index_exists:
                logger.info("Serialized FAISS index found in S3. Loading index...")
                # Loading existing index from S3
                obj = self.s3_client.get_object(Bucket=self.processed_bucket, Key="vector_store/faiss_index.pickle")
                data = obj['Body'].read()
                try:
                    vector_store = FAISS.deserialize_from_bytes(data, embeddings)
                    logger.info("FAISS index loaded from S3.")
                    # If there are new documents, add them to the index
                    if new_documents and len(new_documents) > 0:
                        logger.info(f"Merging {len(new_documents)} new document chunks into the existing index.")
                        new_store = FAISS.from_documents(new_documents, embeddings)
                        vector_store.merge_from(new_store)  # combine the new vectors
                        # Saving the updated index back to S3
                        updated_data = vector_store.serialize_to_bytes()
                        self.s3_client.put_object(Bucket=self.processed_bucket, Key="vector_store/faiss_index.pickle", Body=updated_data)
                        logger.info("FAISS index updated with new documents and saved to S3.")
                    return vector_store
                except Exception as deserialize_error:
                    logger.error(f"Error deserializing FAISS index: {deserialize_error}")
                    # Fall through to create a new index
                    index_exists = False
                    
            # Either no index exists or we couldn't deserialize it
            logger.info("Creating a new FAISS index from available documents...")
            chunks = self.load_chunks_from_processed_bucket()
            
            # Adding new documents if they exist
            if new_documents and len(new_documents) > 0:
                if not chunks:
                    chunks = []
                chunks.extend(new_documents)
            
            if not chunks or len(chunks) == 0:
                raise ValueError("No chunks available to create vector store")
                
            vector_store = FAISS.from_documents(chunks, embeddings)
            # Saving the new index to S3 for future reuse
            data = vector_store.serialize_to_bytes()
            self.s3_client.put_object(Bucket=self.processed_bucket, Key="vector_store/faiss_index.pickle", Body=data)
            logger.info("New FAISS index created and saved to S3.")
            return vector_store
        except Exception as e:
            logger.error(f"Error in get_vector_store: {e}")
            # Create a minimal index if all else fails
            minimal_docs = new_documents if new_documents else self.load_chunks_from_processed_bucket()
            if not minimal_docs or len(minimal_docs) == 0:
                # Create an empty index as last resort
                minimal_docs = [Document(page_content="Empty placeholder", metadata={})]
            return FAISS.from_documents(minimal_docs, embeddings)  
        
    # Function to scan the raw bucket for all documents and processes them
    # Returns a list of all processed document chunks
    def process_all_raw_documents(self, progress_callback=None):
        try:
            # Listing all documents in the raw bucket
            response = self.s3_client.list_objects_v2(Bucket=self.raw_bucket)
            if 'Contents' not in response:
                logger.warning(f"No documents found in {self.raw_bucket}")
                return []
                
            all_documents = response['Contents']
            total_docs = len(all_documents)
            logger.info(f"Found {total_docs} documents in raw bucket")
            
            if progress_callback:
                progress_callback(0.1, f"Found {total_docs} documents to process")
            
            all_chunks = []
            processed_count = 0
            skipped_count = 0
            for idx, doc in enumerate(all_documents):
                file_key = doc['Key']
                # Skipping non-PDF files and system files
                if not file_key.lower().endswith('.pdf') or file_key.startswith('.'):
                    continue
                    
                # Checking if this document is already processed
                already_processed = False
                try:
                    self.s3_client.head_object(Bucket=self.processed_bucket, Key=f"processed/{file_key}.json")
                    already_processed = True
                except:
                    already_processed = False
                    
                if already_processed:
                    logger.info(f"Document {file_key} already processed, skipping")
                    skipped_count += 1
                    if progress_callback:
                        progress = 0.1 + 0.8 * ((idx + 1) / total_docs)
                        progress_callback(progress, f"Skipping already processed document {idx+1}/{len(all_documents)}")
                        # Don't show every skipped document message
                        # progress_callback(progress, f"Skipping already processed document {idx+1}/{total_docs}")
                    continue
                    
                # Downloading the document to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    temp_path = tmp_file.name
                    self.s3_client.download_file(self.raw_bucket, file_key, temp_path)
                    
                    if progress_callback and processed_count == 0:  # Only show for first document
                        progress = 0.1 + 0.8 * ((idx + 0.5) / total_docs)
                        progress_callback(progress, f"Processing document: {file_key}")
                    
                    # Processing the document based on its size
                    file_size = os.path.getsize(temp_path) / (1024 * 1024)  # size in MB
                    if file_size > 10:
                        chunks = self.process_large_document(temp_path, file_key)
                    else:
                        chunks = self.process_document(temp_path, file_key)
                    
                    all_chunks.extend(chunks)
                    os.unlink(temp_path)  # Clean up of temporary file
                    processed_count += 1
                
                # Only updating for significant milestones (25%, 50%, 75%)
                if progress_callback and (idx + 1) % max(1, total_docs // 4) == 0:
                    progress = 0.1 + 0.8 * ((idx + 1) / total_docs)
                    progress_callback(progress, f"Processed {processed_count} documents, skipped {skipped_count}")
            
            if progress_callback:
                progress_callback(0.95, "Building vector store from all documents")
                
            # Updating the vector store with all chunks
            vector_store = self.get_vector_store(new_documents=all_chunks)
            
            if progress_callback:
                progress_callback(1.0, f"All documents processed and indexed: {processed_count} processed, {skipped_count} skipped")
                
            return all_chunks
        
        except Exception as e:
            logger.error(f"Error processing all documents: {e}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            raise
            
    # Retrieving and return all chunked documents from the processed bucket as Document objects
    def load_chunks_from_processed_bucket(self) -> List[Document]:
        try:
            documents = []
            response = self.s3_client.list_objects_v2(Bucket=self.processed_bucket, Prefix='processed/')
            for obj in response.get('Contents', []):
                obj_data = self.s3_client.get_object(Bucket=self.processed_bucket, Key=obj['Key'])
                chunks_data = json.loads(obj_data['Body'].read().decode('utf-8'))
                # Recreating Document objects from stored chunks
                for content, metadata in zip(chunks_data["chunks"], chunks_data["metadata"]):
                    documents.append(Document(page_content=content, metadata=metadata))
            logger.info(f"Loaded {len(documents)} total chunks from processed bucket.")
            return documents
        except Exception as e:
            logger.error(f"Error loading chunks from processed bucket: {e}")
            raise
        
         
    # Processing a single document and store chunks in processed S3 bucket
    def process_document(self, file_path: str, file_key: str):
        # Processing a single document without explicit progress callback
        return self.process_with_progress(file_path, file_key, progress_callback=None)

    # Adding a progress function to process a document with progress updates
    def process_with_progress(self, file_path: str, file_key: str, progress_callback=None):
        try:
            # Start progress
            if progress_callback: 
                progress_callback(0.1, "Loading document")
            
            # Loading document
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            if progress_callback: 
                progress_callback(0.3, "Splitting into chunks")
            chunks = self.text_splitter.split_documents(pages)
            
            if progress_callback: 
                progress_callback(0.5, "Processing chunks")
            processed_data = {
                "chunks": [chunk.page_content for chunk in chunks],
                "metadata": [chunk.metadata for chunk in chunks]
            }
            
            if progress_callback: 
                progress_callback(0.7, "Uploading to S3")
            
            # Uploading the processed chunks to the processed S3 bucket as JSON
            self.s3_client.put_object(
                Bucket=self.processed_bucket,
                Key=f"processed/{file_key}.json",
                Body=json.dumps(processed_data)
            )
            
            if progress_callback: 
                progress_callback(0.9, "Finalizing")
            logger.info(f"Document '{file_key}' processed and uploaded to S3 with {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    # Processing a large document by splitting it into manageable chunks
    def process_large_document(self, file_path, file_key, max_pages_per_chunk=50, progress_callback=None):
        try:
            if progress_callback:
                progress_callback(0.1, "Loading document")
            loader = PyPDFLoader(file_path)
            all_pages = loader.load()
            total_pages = len(all_pages)
            logger.info(f"Loaded large document '{file_key}' with {total_pages} pages.")
            if progress_callback:
                progress_callback(0.2, "Splitting and uploading in batches")
            all_chunks = []
            # Process in batches of max_pages_per_chunk
            for start in range(0, total_pages, max_pages_per_chunk):
                batch_pages = all_pages[start : start + max_pages_per_chunk]
                chunks = self.text_splitter.split_documents(batch_pages)
                all_chunks.extend(chunks)
                # Store this batch to S3
                batch_index = start // max_pages_per_chunk
                batch_key = f"{file_key}_batch_{batch_index}"
                processed_data = {
                    "chunks": [c.page_content for c in chunks],
                    "metadata": [c.metadata for c in chunks]
                }
                self.s3_client.put_object(
                    Bucket=self.processed_bucket,
                    Key=f"processed/{batch_key}.json",
                    Body=json.dumps(processed_data)
                )
                # Update progress
                if progress_callback:
                    pages_done = min(start + max_pages_per_chunk, total_pages)
                    progress = 0.1 + 0.6 * (pages_done / total_pages)
                    status = f"Processed {pages_done}/{total_pages} pages"
                    progress_callback(progress, status)
            logger.info(f"Processed large document '{file_key}' into {len(all_chunks)} total chunks.")
            if progress_callback:
                progress_callback(0.7, "Finished uploading all chunks")
            return all_chunks
        except Exception as e:
            logger.error(f"Error processing large document '{file_key}': {e}")
            raise

# Class for Gemma model        
class GemmaLLM:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = Together(api_key=self.api_key)
        self.model = "google/gemma-2-9b-it"
        logger.info(f"[GemmaLLM] Using Together model: {self.model}")

    def invoke(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.7
            )
            content = response.choices[0].message.content.strip()
            logger.info(f"[GemmaLLM] Response: {content[:100]}...")
            return AIMessage(content=content)
        except Exception as e:
            logger.error(f"[GemmaLLM] Error: {str(e)}")
            return AIMessage(content=f"[Gemma Error] {str(e)}")
        
class GemmaDoRAFineTunedLLM:
    def __init__(self, model_path="Shreya-cn/gemma-dora-finetuned", hf_token=None):
        logger.info(f"Loading fine-tuned Gemma-2 model from Hugging Face: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load token from env if not passed
        self.hf_token = hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.hf_token:
            raise ValueError("Hugging Face API token not found. Set 'HUGGINGFACEHUB_API_TOKEN' in env or pass as argument.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_auth_token=self.hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_auth_token=self.hf_token
        ).to(self.device)

        logger.info(f"Model loaded on device: {self.device}")

    def invoke(self, prompt, generation_config=None):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

            default_config = {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }

            if generation_config:
                default_config.update(generation_config)

            with torch.no_grad():
                outputs = self.model.generate(input_ids=inputs["input_ids"], **default_config)

            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            logger.info(f"[DoRA LLM] Output: {response_text[:100]}...")
            return AIMessage(content=response_text)

        except Exception as e:
            logger.error(f"[DoRA LLM] Inference error: {str(e)}")
            return AIMessage(content=f"[DoRA LLM Error] {str(e)}")

class QwenLLM:
    def __init__(self, model_name: str, together_api_key: str):
        """
        Initialize Qwen model client with Together AI.
        
        Args:
            model_name: Model ID on Together AI (e.g., "Qwen/Qwen2.5-7B-Instruct-Turbo")
            together_api_key: Your Together AI API key
        """
        self.model_name = model_name
        # Initialize the OpenAI client with Together AI base URL
        self.client = OpenAI(
            api_key=together_api_key,
            base_url='https://api.together.xyz/v1',
        )
    
    def invoke(self, prompt: str):
        """
        Invokes the Qwen model using the Together AI API.
        """
        try:
            # For chat models, use chat.completions format
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            
            # Get content from the response
            content = response.choices[0].message.content
            
            logger.info(f"Qwen response: {content[:100]}...")
            return AIMessage(content=content)
        except Exception as e:
            logger.error(f"Error calling Qwen model: {str(e)}")
            return AIMessage(content=f"Error generating response from Qwen model: {str(e)}")
        
# class QwenFinetunedLLM:
#     _instance = None
#     _model = None
#     _tokenizer = None
    
#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             cls._instance = super(QwenFinetunedLLM, cls).__new__(cls)
#             cls._instance._initialized = False
#         return cls._instance
    
#     def __init__(self, model_path=None, device=None):
#         if hasattr(self, '_initialized') and self._initialized:
#             return
        
#         # Set model path to your fine-tuned model folder
#         self.model_path = model_path or "/Users/aafrinshaheen/Desktop/Domain_expert/qwen-0.5b-dora-adapter-5epochs"
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.is_mac = platform.system() == 'Darwin'
        
#         # Not using offload folder for LoRA models
#         self._initialized = True
        
#     def _load_model(self):
#         if self._model is None and self._tokenizer is None:
#             logger.info(f"Loading fine-tuned Qwen model from {self.model_path} on {self.device}")    
#             try:
#                 # Load tokenizer from the fine-tuned model folder
#                 self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
#                 self._tokenizer.pad_token = self._tokenizer.eos_token
                
#                 # Check if model_weights.pt exists
#                 weights_path = os.path.join(self.model_path, "model_weights.pt")
                
#                 if os.path.exists(weights_path):
#                     logger.info(f"Loading from model_weights.pt at {weights_path}")
#                     # Mac-specific loading with limited optimizations
#                     if self.is_mac:
#                         self._model = AutoModelForCausalLM.from_pretrained(
#                             "Qwen/Qwen2.5-0.5B-Instruct",
#                             device_map={"": self.device},
#                             torch_dtype=torch.float32,
#                             trust_remote_code=True
#                         )
#                         # Load the fine-tuned weights directly
#                         state_dict = torch.load(weights_path, map_location=self.device)
#                         self._model.load_state_dict(state_dict)
#                     else:
#                         # Non-Mac loading with optimizations
#                         self._model = AutoModelForCausalLM.from_pretrained(
#                             "Qwen/Qwen2.5-0.5B-Instruct",
#                             device_map="auto",
#                             load_in_8bit=True if torch.cuda.is_available() else False,
#                             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#                             trust_remote_code=True
#                         )
#                         # Load the weights
#                         self._model.load_state_dict(torch.load(weights_path, map_location=self.device))
#                 else:
#                     #logger.info(f"Loading complete model directly from {self.model_path}")
#                     logger.info(f"Loading DORA adapter from {self.model_path}")
#                     # Load the DORA adapter configuration
#                     with open(os.path.join(self.model_path, "adapter_config.json"), "r") as f:
#                         adapter_config = json.load(f)
                        
#                     base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")

#                     # Load the entire model directly from the folder
#                     if self.is_mac:
#                         # self._model = AutoModelForCausalLM.from_pretrained(
#                         #     self.model_path,
#                         #     device_map={"": self.device},
#                         #     torch_dtype=torch.float32,
#                         #     trust_remote_code=True
#                         # )
#                         self._model = AutoModelForCausalLM.from_pretrained(
#                             base_model_name,
#                             device_map={"": self.device},
#                             torch_dtype=torch.float32,
#                             trust_remote_code=True
#                         )
#                     else:
#                         # self._model = AutoModelForCausalLM.from_pretrained(
#                         #     self.model_path,
#                         #     device_map="auto",
#                         #     load_in_8bit=True if torch.cuda.is_available() else False,
#                         #     torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#                         #     trust_remote_code=True
#                         # )
#                         self._model = AutoModelForCausalLM.from_pretrained(
#                             base_model_name,
#                             device_map="auto",
#                             load_in_8bit=True if torch.cuda.is_available() else False,
#                             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#                             trust_remote_code=True
#                         )
                    
#                 adapter_weights_path = os.path.join(self.model_path, "adapter_model.safetensors")
                
#                 # Load the model with DORA adapter (avoiding PEFT's PeftModel which may have QLora assumptions)
#                 if os.path.exists(adapter_weights_path):
#                     # Load DORA weights from adapter_model.safetensors
#                     with safe_open(adapter_weights_path, framework="pt", device=self.device) as f:
#                         for key in f.keys():
#                             # Map adapter keys to model parameters
#                             if key in self._model.state_dict():
#                                 self._model.state_dict()[key].copy_(f.get_tensor(key))
                
#                 logger.info(f"DORA adapter loaded successfully")
                    
#                 # logger.info(f"QLora adapter loaded successfully")
#                 # logger.info(f"Fine-tuned Qwen model loaded successfully on {'Mac' if self.is_mac else 'non-Mac'} system using {self.device} device")
            
#             except Exception as e:
#                 logger.error(f"Error loading fine-tuned Qwen model: {e}")
#                 raise
                
#         return self._model, self._tokenizer
                    
#     def invoke(self, prompt: str):
#         """
#         Invokes the fine-tuned Qwen model.
#         """
#         try:
#             model, tokenizer = self._load_model()
            
#             # Format prompt for instruction tuning format
#             formatted_prompt = f"""### Instruction:
# Generate a high-quality question to extract information missing from a Knowledge Transfer document.

# ### Input:
# {prompt}

# ### Response:"""
            
#             # Tokenize the prompt
#             inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
#             # Generate response
#             with torch.no_grad():
#                 outputs = model.generate(
#                     inputs["input_ids"],
#                     max_new_tokens=256,
#                     temperature=0.6,
#                     top_p=0.9,
#                     do_sample=True
#                 )
            
#             # Decode the outputs
#             generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             # Extract only the response part (after the prompt)
#             response = generated_text[len(formatted_prompt):].strip()
            
#             logger.info(f"Fine-tuned Qwen generated response: {response[:100]}...")
#             return AIMessage(content=response)
#         except Exception as e:
#             logger.error(f"Error generating with fine-tuned Qwen: {e}")
#             return AIMessage(content="Error generating response from fine-tuned Qwen model.")
        




class QwenFinetunedLLM:
    def __init__(self, model_path="Shreya-cn/qwen-dora-finetuned", hf_token=None, device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        if not self.hf_token:
            raise ValueError("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN in .env or pass directly.")

        logger.info(f"Loading Qwen model from Hugging Face: {self.model_path} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_auth_token=self.hf_token,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map={"": self.device},
            use_auth_token=self.hf_token,
            trust_remote_code=True
        )

        logger.info("Qwen model and tokenizer loaded successfully.")
                    
    def invoke(self, prompt: str):
        try:
            # Format prompt for instruction tuning
            formatted_prompt = f"""### Instruction:
Generate a high-quality question to extract information missing from a Knowledge Transfer document.

### Input:
{prompt}

### Response:"""

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=256,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=True
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_output = response[len(formatted_prompt):].strip()
            logger.info(f"Qwen response: {final_output[:100]}...")
            return AIMessage(content=final_output)

        except Exception as e:
            logger.error(f"Qwen inference error: {e}")
            return AIMessage(content=f"[Qwen Error] {str(e)}")
        
# Class for Llama 3.2 base model  
class Llama3LLM:
    def _init_(self, model_path="saisahithi/llama-base", model_name="llama-3.2-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

    def invoke(self, prompt: str, context: Optional[List[str]] = None):
        try:
            context_text = "\n".join(context) if context else ""
            final_prompt = f"""### Instruction:
You are a domain-specific expert reviewing a knowledge transfer document. Ask one insightful and non-repetitive question to fill missing knowledge gaps.

### Context:
{context_text}

### Query so far:
{prompt}

### Response:"""

            inputs = self.tokenizer(final_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.6, top_p=0.9)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return AIMessage(content=response.strip())
        except Exception as e:
            logger.error(f"LLaMA error: {e}")
            return AIMessage(content="Error generating response from LLaMA3.")



# Class for Llama 3.2 lora model   
class Llama3LoraFineTunedLLM:
    def _init_(self, model_path="saisahithi/llama-3.2-lora", model_name="llama-lora-finetuned"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

    def invoke(self, prompt: str):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.6, top_p=0.9)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return AIMessage(content=response.strip())
        except Exception as e:
            logger.error(f"LoRA Fine-Tuned LLaMA error: {e}")
            return AIMessage(content="Error generating response from LoRA Fine-Tuned LLaMA3.")
# Class for Deepseek R1 model        
class DeepSeekLLM:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

    def invoke(self, input_text, metadata=None, context=None):
        metadata = metadata or {}  # To Avoid NoneType error
        final_prompt = f"""
You are an intelligent domain-specific interviewer assisting in the knowledge transfer process during project handovers. 
Your role is to simulate a knowledgeable peer by asking one **relevant, concise, and original question** based on the KT document shared.

Always begin by identifying the following metadata from the document:
- **Author**
- **Role**
- **Domain**
- **Industry**
- **Date**

Use this metadata to **tailor your question** to the expertise of the user and the project's business context. Your goal is to surface **one insightful and non-repetitive question** that reflects a deep understanding of both the content and the metadata context.

---

**Document Metadata:**
Author: {metadata.get('author', 'Unknown')}  
Role: {metadata.get('role', 'Unknown')}  
Domain: {metadata.get('domain', 'Unknown')}  
Industry: {metadata.get('industry', 'Unknown')}  
Date: {metadata.get('date', 'Unknown')}

---

**Document Context:**
{context or "No context provided"}

**Current Query:** {input_text}

---

**Instructions:**
1. Ask **only one** unique, high-quality question that ends with a question mark.
2. Avoid using generic phrasing like “what is the main topic...”
3. Do not include explanations or summaries—just the question.
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": final_prompt}]
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return AIMessage(content=response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            logger.error(f"❌ DeepSeek API error: {e}")
            return AIMessage(content="DeepSeek failed to respond.")
        
class DeepSeekCoderQLORAFinetuned8bit:
    def __init__(self, model_id="aiswaryards/deepseek1.3B-coder-qlora-finetuned-8bit", token=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
        import torch

        self.model_id = model_id
        self.token = token or os.getenv("huggingface_deepseek_finetuned")
        self.device = 0 if torch.cuda.is_available() else -1
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            # 8-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.token,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.token,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=bnb_config
            )

            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                #device=self.device
            )

            logger.info("[DeepSeekCoderQLORAFinetuned8bit] ✅ Model loaded successfully.")
        except Exception as e:
            logger.error(f"[DeepSeekCoderQLORAFinetuned8bit] ❌ Initialization failed: {e}")
            raise

    def format_prompt(self, context: str, previous_qa: str = "", extracted_keywords: str = "") -> str:
        return f"""
You are an intelligent handover assistant.

Your task is to generate ONE insightful, domain-specific question that targets gaps or undocumented insights in the KT document.

---

**KT Document Context**:
{context}

**Previous Questions and Answers**:
{previous_qa}

**Current Instruction**:
Ask a meaningful, non-redundant question that uncovers hidden technical or business knowledge.

Question:
"""

    def generate_response(self, prompt, previous_qa="", extracted_keywords="", max_tokens=512, temperature=0.7):
        try:
            formatted_prompt = self.format_prompt(prompt, previous_qa, extracted_keywords)

            outputs = self.generator(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated = outputs[0]['generated_text']
            response = generated[len(formatted_prompt):].strip()

            if "?" in response:
                sentences = re.split(r'(?<=[.!?])\s+', response)
                for sentence in sentences:
                    if "?" in sentence:
                        return sentence.strip()

            return response
        except Exception as e:
            logger.error(f"[DeepSeekCoderQLORAFinetuned8bit] Error: {str(e)}")
            return f"Error generating response: {str(e)}"

    def invoke(self, prompt):
        try:
            if isinstance(prompt, dict):
                result = self.generate_response(
                    prompt.get("context", ""),  
                    prompt.get("previous_qa", ""),
                    prompt.get("extracted_keywords", "")
                )
            elif isinstance(prompt, str):
                result = self.generate_response(prompt)
            else:
                result = self.generate_response(str(prompt))

            logger.info(f"Generated result: {result[:100]}...")
            return AIMessage(content=result)
        except Exception as e:
            logger.error(f"[DeepSeekCoderQLORAFinetuned8bit] Error in invoke: {str(e)}")
            return AIMessage(content=f"Error generating response: {str(e)}")

# Expert agent class with evaluation metrics 
class ExpertAgent(Agent):
    def __init__(self, unique_id: int, model: Model, vector_store, 
                 anthropic_api_key=None, huggingface_api_key=None, together_api_key=None):
        super().__init__(unique_id, model)
        self.vector_store = vector_store
        self.anthropic_api_key = anthropic_api_key
        self.huggingface_api_key = huggingface_api_key
        self.together_api_key = together_api_key
        
        # Defaulting to Claude Haiku
        self.llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            anthropic_api_key=anthropic_api_key,
            temperature=0.3
        )

        self.model_name = "claude-3-haiku-20240307"
        self.evaluator = RAGEvaluator()
        # Default prompt for Claude
        self.prompt_template = """You are an experienced data science expert reviewer. 
        You need to analyze the provided information and ask ONE relevant question to ensure 
        comprehensive knowledge transfer. Use the context below to inform your question.
        
        Context: {context}
        
        Current query: {input}
        
        Please provide EXACTLY ONE concise question. Do not include any other text, explanation,
        or multiple questions. Just one clear, focused question based on the document context."""

    def extract_keywords(self, known_info):
        # Extract important keywords from previous questions
        if not known_info:
            return "No previous questions"
        
        # Get just the questions (even indices)
        questions = [known_info[i] for i in range(0, len(known_info), 2) if i < len(known_info)]
        
        # Extract important words (non-stopwords)

        stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 
                    'how', 'what', 'why', 'where', 'when', 'who', 'does', 'do', 'in', 'to', 'of', 'that', 'are'}
        
        all_words = []
        for q in questions:
            # Extract words, remove punctuation
            words = re.findall(r'\b[a-zA-Z]{3,}\b', q.lower())
            # Filter out stopwords
            keywords = [w for w in words if w not in stopwords]
            all_words.extend(keywords)
        
        # Get most common keywords
        common_keywords = Counter(all_words).most_common(10)
        return ", ".join([word for word, count in common_keywords])

    # Updating the LLM model being used and customizing prompt accordingly
    def set_model(self, model_name, anthropic_api_key=None, huggingface_api_key=None, together_api_key=None):
        self.model_name = model_name
        
        if "claude" in model_name:
            self.llm = ChatAnthropic(
                model=model_name,
                anthropic_api_key=anthropic_api_key,
                temperature=0.3
            )
            self.prompt_template = """You are an experienced data science expert reviewer. 
You need to analyze the provided information and ask ONE relevant question to ensure 
comprehensive knowledge transfer. Use the context below to inform your question.

Context: {context}

Current query: {input}

Please provide EXACTLY ONE concise question. Do not include any other text, explanation,
or multiple questions. Just one clear, focused question based on the document context."""

        elif "gemma" in model_name.lower():
            self.llm = GemmaLLM(api_key=together_api_key or os.getenv("together_api_key"))
            self.prompt_template = """**Technical Question Generation Guide**

    Context:
    {context}

    Previous Discussion:
    {input}

    Generate ONE technical question meeting these criteria:
    1. Focuses specifically on: 
    - Algorithms | Data Pipelines | Model Evaluation | Feature Engineering
    - Statistical Methods | Implementation Details
    2. References explicit content from the context
    3. Cannot be answered without the provided documentation
    4. Ends with a question mark

    Examples of GOOD questions:
    - How does the feature importance calculation handle categorical variables in the described pipeline?
    - What regularization technique is applied in the neural network architecture on page 3?

    Your question:"""
    
        elif "gemma-dora" in model_name.lower():
            self.llm = GemmaDoRAFineTunedLLM(
                model_path="Shreya-cn/gemma-dora-finetuned", 
                hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") 
            )
            self.prompt_template = """### Instruction:
        Generate a high-quality technical question based on the document context.

        Context:
        {context}

        Prior Discussion:
        {input}

        Your task:
        Generate ONE specific technical question that reveals an unaddressed aspect of the document. Ensure the question ends with a question mark and is rooted in the content provided.

        Examples:
        - What method was used to validate data pipeline efficiency as described in section 3?
        - How are feature selection techniques applied during model retraining?

        Your question:"""

        elif "qwen-finetuned-dora" in model_name.lower():
            # Use the Hugging Face-hosted Qwen DoRA fine-tuned model
            model_path = "Shreya-cn/qwen-dora-finetuned"
            self.llm = QwenFinetunedLLM(
                model_path=model_path,
                hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )

            # Custom prompt template for fine-tuned Qwen DoRA
            self.prompt_template = """You are an expert knowledge extraction agent.

        Context from document:
        {context}

        Current query: {input}

        Generate ONE specific, technical question about information in the document that has NOT been covered yet.
        Your question must end with a question mark and directly relate to the technical content."""

        elif "qwen" in model_name.lower():
            self.llm = QwenLLM(
                model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",  
                together_api_key=self.together_api_key  
            )
            self.prompt_template = """You are an expert knowledge extraction agent.

Context from document:
{context}

Previous queries and answers:
{previous_qa}

Current query: {input}

TASK: Generate ONE specific technical question that has NEVER been asked before.

RULES:
1. Never repeat previous questions
2. Focus on technical details, algorithms, data structures
3. Reference specific content from the document
4. Each question must explore NEW information
5. End with a question mark

Avoid generic terms: "challenges," "benefits," "approach"
Target specific technical components not yet discussed.

YOUR UNIQUE TECHNICAL QUESTION:"""

        elif "deepseek-chat" in model_name:
            self.llm = DeepSeekLLM(api_key=os.getenv("DEEPSEEK_API_KEY"))
            
        elif model_name == "deepseekcoder-qlora8bit-finetuned":
            from agentic_modeling_classes import DeepSeekCoderQLORAFinetuned8bit
            huggingface_token = os.getenv("huggingface_deepseek_finetuned")
            self.model = DeepSeekCoderQLORAFinetuned8bit(token=huggingface_token)
            
        elif model_name == "llama-3.2-base":
            self.llm = Llama3LLM(model_path="saisahithi/llama-base")
            self.model_name = model_name
            self.prompt_template = """### Instruction:
        You are a domain expert reviewing a technical document. 
        Your task is to generate ONE highly specific and relevant question that identifies missing information.

        Do NOT answer the question.
        Do NOT include explanations.
        The question must end with a question mark.

        Avoid repeating previous questions.

        ### Context:
        {context}

        ### Previous Q&A:
        {previous_qa}

        ### Current topic:
        {input}

        ### Response:"""


        elif model_name == "llama-lora-finetuned":
            self.llm = Llama3LoraFineTunedLLM(model_path="saisahithi/llama-3.2-lora")
            self.model_name = model_name
            self.prompt_template = """You are a senior data science specialist reviewing technical documentation. 
            Ask ONE SPECIFIC TECHNICAL QUESTION about data science concepts, methods, or implementations found in the document context.
            
            Context from document:
            {context}
            
            Previous query: {input}
            
            Instructions:
            1. Focus ONLY on data science topics like: machine learning algorithms, data processing techniques, metrics calculation, model evaluation, feature engineering, statistical methods, programming frameworks, or technical implementations.
            2. Ask a SPECIFIC and TECHNICAL question that directly relates to the data science content in the context.
            3. Your question must be detailed enough that it couldn't apply to any general document.
            4. Make sure your question ends with a question mark.
            5. Provide ONLY the question - no introduction, explanation, or additional text.

            BAD EXAMPLES (too generic):
            - What is the primary focus of the handover process?
            - How does the system work?
            - What technologies are used?

            GOOD EXAMPLES (specific to data science):
            - How does the context utilization metric calculate the overlap between retrieved documents and generated responses?
            - What specific implementation of FAISS vector store is used for similarity search in the document retrieval pipeline?
            - How are the information density and response specificity metrics computed in the metrics tracking system?

            Your technical data science question:"""
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        

    """
    Generate relevant questions based on document context and previous information.
    
    This function:
    1. Searches for relevant context in the vector store
    2. Generates a focused question using the LLM
    3. Ensures the question is unique and well-formatted
    
    Args:
        query: The base query or previous question
        known_info: List of previously captured knowledge
        
    Returns:
        A single, focused question or "No further questions" signal
    """
    
    def evaluate_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Evaluates the quality of a knowledge transfer question based on the provided context.
        Returns only the score without feedback.
        """
        try:            
            # Create a Claude client
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            # Prepare the evaluation prompt
            system_prompt = "You are a Knowledge Transfer agent assisting to extract information about tasks and responsibilities."
            
            user_prompt = f"""Evaluate the quality of this question for extracting necessary information from a knowledge transfer document.

    Knowledge Transfer Document Content:
    {context}

    Question to evaluate:
    {question}

    Rate the question from 5-10, where:
    - 7-10: Clear question that asks about information missing from the document
    - 5-6: Decent question but might be partially addressed in the document

    Return ONLY a single number (5-10) as your response."""

            # Call Claude directly
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=10,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract content from response
            content = response.content[0].text.strip()
            
            # Try to extract and convert the score
            try:
                # Parse the score from the response                
                digits = re.search(r'\d+', content)
                if digits:
                    score = int(digits.group())
                    score = min(max(score, 5), 10)
                else:
                    score = 0
                
                print(f"Question evaluated with score: {score}")
            except (ValueError, TypeError):
                print(f"Could not extract score from: {content}")
                score = 0
                
            # Return with standardized key name
            return {"score": score}
        
        except Exception as e:
            print(f"Exception in evaluate_question: {e}")
            return {"score": 0}
                
    def ask_questions(self, query: str, known_info: List[str]) -> str:
        try:
            # Checking if we've reached the maximum number of questions (10)
            if len(known_info) >= 10:
                return "No further questions. Maximum questions reached."
            
            # Create formatted previous Q&A pairs
            previous_qa = ""
            for i in range(0, len(known_info), 2):  # Assuming even/odd pairs of Q/A
                if i+1 < len(known_info):
                    previous_qa += f"Q: {known_info[i]}\nA: {known_info[i+1]}\n\n"

            # Measuring retrieval time
            retrieval_start = time.time()
            # Instead of always getting top k=3 documents
            k_value = min(3 + len(known_info)//2, 8)  # Increases as more questions are asked
            relevant_docs = self.vector_store.similarity_search(query, k=k_value)
            retrieval_time = time.time() - retrieval_start

            # If we have enough docs, select a random subset for diversity
            if len(relevant_docs) > 3:
                selected_docs = random.sample(relevant_docs, 3)
            else:
                selected_docs = relevant_docs    
                
            # Store retrieved context
            self.last_retrieved_context = [doc.page_content for doc in selected_docs]
            context = "\n".join(self.last_retrieved_context)    
        
            # Helper function to check if all documents are covered
            def all_documents_covered(last_retrieved_context, known_info):
                for doc_content in last_retrieved_context:
                    doc_known = False
                    for info in known_info:
                        words = set(doc_content.lower().split())
                        info_words = set(info.lower().split())
                        overlap = len(words.intersection(info_words)) / len(words) if words else 0
                        if overlap > 0.5:
                            doc_known = True
                            break
                    if not doc_known:
                        return False
                return True
        
            # Check if all documents are already covered
            if all_documents_covered(self.last_retrieved_context, known_info):
                return "No further questions. All relevant information has been covered."
                            
            # Extract keywords from previous questions
            extracted_keywords = self.extract_keywords(known_info)

            # Pass this to your prompt
            formatted_prompt = self.prompt_template.format(
                context=context, 
                input=query,
                previous_qa=previous_qa,
                extracted_keywords=extracted_keywords
                )
            
            # Invoke the LLM that was configured (could be Claude, Falcon, or Gemma)
            response = self.llm.invoke(formatted_prompt)
        
            # Extract a clean question from the response
            single_question = self._extract_single_question(response.content)
            
            # Evaluate the question and store the result
            try:
                if single_question and single_question != "No further questions.":
                    evaluation_result = self.evaluate_question(single_question, context)
                    self.last_question_evaluation = evaluation_result  # Store as instance attribute
                    print(f"Question evaluated with score: {evaluation_result.get('score', 0)}")
                else:
                    self.last_question_evaluation = {"score": 0}
            except Exception as e:
                print(f"Error evaluating question: {e}")
                self.last_question_evaluation = {"score": 0}
            
            # #  # THIS IS THE MISSING PART - Evaluate the generated question
            # if hasattr(self, 'evaluate_question'):
            #     try:
            #         self.last_question_evaluation = self.evaluate_question(single_question, context)
            #         print(f"Question evaluated with score: {self.last_question_evaluation.get('score', 0)}")
            #     except Exception as e:
            #         print(f"Error evaluating question: {e}")
            #         self.last_question_evaluation = {"score": 0}
            
            # For the first question (when there's no known info yet), ensure proper evaluation
             # This is the added code to fix the first question issue
            if not known_info:  # If this is the first question
                initial_context = "Initial knowledge transfer session. " + context  
                # Evaluating the initial question as well
                if single_question and single_question != "No further questions.":
                    try:
                        self.last_question_evaluation = self.evaluate_question(single_question, initial_context)
                    except Exception as e:
                        logger.error(f"Error evaluating initial question: {e}")
                        self.last_question_evaluation = {"score": 0}  
            logger.info(f"Generated question: {single_question}")
        
            if single_question == "No further questions.":
                logger.info("No further questions determined, ending session.")            
            return single_question
            
        except Exception as e:
            logger.error(f"Error in ExpertAgent: {str(e)}")
            return "Error generating questions."
    
    # Extracting a single question from the text, removing numbering or explanations
    def _extract_single_question(self, text: str) -> str:

        # Logging the text being processed
        logger.info(f"Extracting question from text: {text[:100]}...")
    
        # Splitting by newlines and look for the first line that ends with a question mark
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.endswith('?'):
                # Removing any numbering (like "1. " or "Q1: ")
                clean_line = re.sub(r'^[\d\.\s:]*\s*', '', line)
                logger.info(f"Found question: {clean_line}")
                return clean_line
        
        # If that fails, try to find any sentence that ends with a question mark
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence.endswith('?'):
                clean_sentence = re.sub(r'^[\d\.\s:]*\s*', '', sentence)
                logger.info(f"Found question in sentence: {clean_sentence}")
                return clean_sentence
        
        # If still no clear question, return the first non-empty line
        # This is a fallback for unusual formatting
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Ensure it's not just a short header
                logger.info(f"Using first substantial line as question: {line}")
                return line
            
        # If all else fails, return the original text
        logger.warning("Could not extract a clear question, returning original text")
        return text
            
# Knowledge management and validation agent
class KnowledgeManagerAgent(Agent):
    def __init__(self, unique_id: int, model: Model, vector_store, dynamodb=None):
        super().__init__(unique_id, model)
        self.vector_store = vector_store
        self.dynamodb = dynamodb or boto3.resource('dynamodb', region_name='us-east-1')
        self.table_name = 'knowledge_base'
        self.kb_table = self.get_dynamodb_table()
        

    # Function to check for existance of DynamoDB table 
    def get_dynamodb_table(self):
        try:
            existing_tables = self.dynamodb.meta.client.list_tables()["TableNames"]
            if self.table_name not in existing_tables:
                raise ValueError(f"Table {self.table_name} does not exist in DynamoDB.")
            return self.dynamodb.Table(self.table_name)
        except Exception as e:
            print(f"Error accessing DynamoDB table: {e}")
            raise

    # Checking if similar content already exists in the knowledge
    def entry_exists(self, content: str) -> bool:
        try:
            response = self.kb_table.scan(
                FilterExpression="contains(content, :content)",
                ExpressionAttributeValues={":content": content}
            )
            # If any matching items are found, return True
            return bool(response.get("Items"))
        except Exception as e:
            logger.error(f"Error checking existing knowledge base entries: {e}")
            return False
        
    # Updating DynamoDB knowledge base if new knowledge is found
    def update_knowledge_base(self, content: str, analysis: str):
        # Validating if content already exists
        if self.entry_exists(content):
            return f"Duplicate entry found: Skipping update for '{content}'"
        
        # Proceeding with update if content is unique
        try:
            item_id = f"kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.kb_table.put_item(
                Item={
                    'id': item_id,
                    'content': content,
                    'analysis': analysis,
                    'timestamp': str(datetime.now())
                }
            )
            # Returning a success message with details of the update
            return f"Knowledge base updated: Added analysis for '{content}'"
        except Exception as e:
            print(f"Error updating knowledge base: {e}")
            raise

    # Updating multiple knowledge base items in a batch
    def batch_update_knowledge_base(self, items):
        try:
            with self.kb_table.batch_writer() as batch:
                for content, analysis in items:
                    if not self.entry_exists(content):
                        item_id = f"kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        batch.put_item(
                            Item={
                                'id': item_id,
                                'content': content,
                                'analysis': analysis,
                                'timestamp': str(datetime.now())
                            }
                        )
            return "Knowledge base batch updated successfully"
        except Exception as e:
            logger.error(f"Error in batch update: {str(e)}")
            raise

# Main KnowledgeSystem model to manage inquiry interactions
# Max questions restricted to 10 by default to prevent infinite question loops
class KnowledgeSystem(Model):
    def __init__(self, vector_store, 
                 anthropic_api_key=None, huggingface_api_key=None, together_api_key=None, 
                 max_questions=10, dynamodb_resource=None):
        # Initializing the parent Model class
        super().__init__()
        
        # Setting up the scheduler and vector store
        self.schedule = RandomActivation(self)
        self.vector_store = vector_store


        # Initializing agents with API keys
        self.expert = ExpertAgent(1, self, vector_store, anthropic_api_key, huggingface_api_key)
        # Adding the Together API key
        self.expert.together_api_key = together_api_key  
        self.knowledge_manager = KnowledgeManagerAgent(2, self, vector_store, dynamodb_resource)

        # Adding agents to the schedule
        self.schedule.add(self.expert)
        self.schedule.add(self.knowledge_manager)
        # Maximum questions per session
        self.max_questions = max_questions
        logger.info(f"KnowledgeSystem initialized with max_questions: {self.max_questions}")
        
        # For tracking evaluation metrics
        self.last_question_evaluation = None

    def inquire_user(self, query: str, user_response_callback):
        known_info = []
        question_count = 0

        while question_count < self.max_questions:
            try:
                question = self.expert.ask_questions(query, known_info)
            
                # Checking for a signal to stop the loop if no further questions are needed
                if question == "No further questions.":
                    print("Knowledge capture complete.")
                    break
            
                # Getting user response and update known information
                user_response = user_response_callback(question)
                known_info.append(user_response)
                try:
                    self.knowledge_manager.update_knowledge_base(query, user_response)
                except Exception as e:
                    logger.exception(f"Error updating knowledge base: {str(e)}")
                    
                question_count += 1
            
            except Exception as e:
                logger.exception(f"Error in inquire_user loop: {str(e)}")
                break

        # If max_questions is reached without receiving the stop signal
        if question_count >= self.max_questions:
            logger.info("Inquiry stopped: maximum number of questions reached.")

# Evaluation class
class RAGEvaluator:
    def __init__(self):
        self.metrics_log = []

    def evaluate_answer(self,
                       generated_answer: str,
                       retrieved_context: List[str],
                       generation_time: float, retrieval_time: float) -> Dict[str, float]:
        
        
        # Context Utilization calc
        context_words = set(' '.join(retrieved_context).lower().split()) if retrieved_context else set()
        answer_words = set(generated_answer.lower().split()) if generated_answer else set()
        context_utilization = (
            len(context_words.intersection(answer_words)) / len(context_words)
            if len(context_words) > 0 else 0.0
        )

        
        return {
            'context_utilization': context_utilization,
            'generation_time': generation_time, 
            'retrieval_time': retrieval_time
        }

    # Logging evaluation metrics
    def log_evaluation(self, query_metrics: Dict[str, Any]):
        self.metrics_log.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': query_metrics
        })

    # Calculating aggregate metrics from log
    def get_aggregate_metrics(self) -> Dict[str, float]:
        if not self.metrics_log:
            return {}
            
        aggregated = {
            'retrieval': {
                'retrieval_time': [], 
                'generation_time': []
            },
            'answer': {
                'context_utilization': [],
                'generation_time': []
            }
        }
        
        for entry in self.metrics_log:
            metrics = entry['metrics']
            for key in aggregated['retrieval'].keys():
                aggregated['retrieval'][key].append(metrics['retrieval'][key])
            for key in aggregated['answer'].keys():
                aggregated['answer'][key].append(metrics['answer'][key])
        
        return {
            'average_retrieval_time': sum(aggregated['retrieval']['retrieval_time']) / max(len(aggregated['retrieval']['retrieval_time']), 1),
            'average_context_utilization': sum(aggregated['answer']['context_utilization']) / max(len(aggregated['answer']['context_utilization']), 1),
            'average_generation_time': sum(aggregated['answer']['generation_time']) / max(len(aggregated['answer']['generation_time']), 1)
        }
