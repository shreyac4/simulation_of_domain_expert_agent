import streamlit as st
import asyncio
# Setting the page config as the very first Streamlit command
st.set_page_config(
    page_title="Knowledge Transfer System",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import time
import logging
import tempfile
from datetime import datetime

# Setting up logging after Streamlit page config commands
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importing all required libraries
from dotenv import load_dotenv
import plotly.graph_objects as go 
import plotly.express as px
import json
from typing import Dict, List, Any, Optional, Union
from agentic_modeling_classes import DocumentProcessor, KnowledgeSystem
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from metrics_tracker import StreamlitMetricsTracker, log_rag_question_metrics
from session_manager import SessionManager
import pandas as pd
from langchain_anthropic import ChatAnthropic
from huggingface_hub import InferenceClient
from together import Together
import boto3

# Main function for the Streamlit application.

# This function initializes the application, sets up AWS resources, creates the UI components, and handles user interactions.
# The application flow:
    # 1. User uploads a document
    # 2. Document is processed and stored in S3
    # 3. An expert agent generates questions based on the document
    # 4. User answers questions to transfer knowledge
    # 5. Responses are stored in DynamoDB

# Updating the model evaluation leaderboard with new metrics summary
def update_model_eval_leaderboard(summary):
    if "all_eval_summaries" not in st.session_state:
        st.session_state.all_eval_summaries = []

    model_name = summary["model"]
    total = summary["total_questions"]
    duplicates = summary["duplicates"]

    avg_relevance = round(sum(summary["relevance_scores"]) / total, 3) if total and summary["relevance_scores"] else 0 
    avg_diversity = round(sum(summary["diversity_scores"]) / total, 3) if total and summary["diversity_scores"] else 0
    # Calculating user rating average if available
    avg_user_rating = 0
    if "user_ratings" in summary and summary["user_ratings"] and len(summary["user_ratings"]) > 0: 
        avg_user_rating = round(sum(summary["user_ratings"]) / len(summary["user_ratings"]), 2)
    
    # Calculate final score with user feedback
    # Base score (80% weight)
    base_score = round(0.4 * avg_relevance + 0.6 * avg_diversity, 3)

    # If we have user ratings, they get 20% weight
    if "user_ratings" in summary and summary["user_ratings"]:
        # Convert user rating from [-1,1] to [0,1] scale for consistency
        user_score_normalized = (avg_user_rating + 1) / 2
        avg_score = round(0.8 * base_score + 0.2 * user_score_normalized, 3)
    else:
        avg_score = base_score

    # Checking if already exists
    existing = next((entry for entry in st.session_state.all_eval_summaries if entry["Model"] == model_name), None)

    if existing:
        existing["Total Questions"] = total
        existing["Repeated Questions"] = duplicates
        existing["Avg Context Relevance"] = avg_relevance
        existing["Avg Diversity"] = avg_diversity
        existing["Avg User Rating"] = avg_user_rating  # Added user rating
        existing["Avg Score"] = avg_score
    else:
        # Adding new entry
        st.session_state.all_eval_summaries.append({
            "Model": model_name,
            "Total Questions": total,
            "Repeated Questions": duplicates,
            "Avg Context Relevance": avg_relevance,
            "Avg Diversity": avg_diversity,
            "Avg User Rating": avg_user_rating,  # Added user rating
            "Avg Score": avg_score
        })
    logger.info(f"Updated leaderboard for {model_name}: score={avg_score:.3f}, user_rating={avg_user_rating:.2f}")

# Ensures all question counters are synchronized to qa_pairs
def synchronize_question_counts():
    if 'qa_pairs' in st.session_state and st.session_state.qa_pairs:
        qa_count = len(st.session_state.qa_pairs)
        
        # Update metrics count - this is the source of the "Questions Asked: x" display
        if 'metrics' in st.session_state:
            st.session_state.metrics['questions_asked'] = qa_count
            
        # Ensure known_info length is consistent (should be 2 * qa_pairs length)
        # This is needed because known_info stores both Q and A, while qa_pairs stores pairs
        if 'known_info' in st.session_state:
            expected_known_info_length = qa_count * 2
            if len(st.session_state.known_info) != expected_known_info_length:
                logger.warning(f"Question count inconsistency detected: qa_pairs={qa_count}, known_info={len(st.session_state.known_info)//2}")
                # Only fix if known_info is longer than expected (avoid data loss)
                if len(st.session_state.known_info) > expected_known_info_length:
                    st.session_state.known_info = st.session_state.known_info[:expected_known_info_length]
    else:
        # Reset counts to 0 if no qa_pairs
        if 'metrics' in st.session_state:
            st.session_state.metrics['questions_asked'] = 0

def main():

    # Loading environment variables
    load_dotenv()

    # Fixing asyncio event loop issue
    # try:
    #    asyncio.get_running_loop()
    # except RuntimeError:
    #    asyncio.set_event_loop(asyncio.new_event_loop())

    # Initializing asyncio properly for Streamlit
    if "asyncio_setup" not in st.session_state:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        st.session_state.asyncio_setup = True

    # Creating AWS clients with credentials first
    creds = get_streamlit_secrets()
    if creds is None:
        placeholder = st.empty()
        placeholder.warning("Error accessing secrets")
        placeholder.warning("Using environment variables instead of secrets.")
        # auto-clear after 2s
        time.sleep(2)
        placeholder.empty()
        # st.warning("Error accessing secrets")
        # st.warning("Using environment variables instead of secrets.")

    # ===== AWS UTILITY FUNCTIONS ===== #
    # Function for getting S3 client with credentials if provided
    def get_s3_client(credentials=None):
        if credentials and credentials.get("aws_access_key_id") and credentials.get("aws_secret_access_key"):
            return boto3.client(
                's3',
                region_name=credentials.get("region_name", "us-east-1"),
                aws_access_key_id=credentials["aws_access_key_id"],
                aws_secret_access_key=credentials["aws_secret_access_key"]
            )
        else:
            # Using default credentials (from environment variables)
            return boto3.client('s3', region_name="us-east-1")

    # Function for getting DynamoDB resource with credentials if provided
    def get_dynamodb_resource(credentials=None):
        if credentials and credentials.get("aws_access_key_id") and credentials.get("aws_secret_access_key"):
            return boto3.resource(
                'dynamodb',
                region_name=credentials.get("region_name", "us-east-1"),
                aws_access_key_id=credentials["aws_access_key_id"],
                aws_secret_access_key=credentials["aws_secret_access_key"]
            )
        else:
            # Trying to use Streamlit secrets if environment variables are missing
            try:
                secrets = st.secrets["aws"]
                return boto3.resource(
                    'dynamodb',
                    region_name=secrets["region_name"],
                    aws_access_key_id=secrets["aws_access_key_id"],
                    aws_secret_access_key=secrets["aws_secret_access_key"]
                )
            except:
                logger.error("❌ AWS credentials missing for DynamoDB!")
                return None  # Prevents crashing

    # Function for getting CloudWatch client with credentials if provided
    def get_cloudwatch_client(credentials=None):
        if credentials and credentials.get("aws_access_key_id") and credentials.get("aws_secret_access_key"):
            return boto3.client(
                'cloudwatch',
                region_name=credentials.get("region_name", "us-east-1"),
                aws_access_key_id=credentials["aws_access_key_id"],
                aws_secret_access_key=credentials["aws_secret_access_key"]
            )
        else:
            # Using default credentials (from environment variables)
            return boto3.client('cloudwatch', region_name="us-east-1")
    
    # Creating AWS clients with credentials
    s3_client = get_s3_client(creds)
    dynamodb_resource = get_dynamodb_resource(creds)
    cloudwatch_client = get_cloudwatch_client(creds)
    
    # Initializing session state
    initialize_session_state(dynamodb_resource, cloudwatch_client)

    # Initializing session manager
    session_manager = SessionManager()
    
    # Adding custom CSS modern palette & typography
    st.markdown("""
    <style>
        /* Font + base spacing */
        html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #F4F6F8;
        color: #202124;
        }
        /* Hide Streamlit default menu & footer */
        #MainMenu, footer { visibility: hidden; }

        /* CARD (tabs & content) */
        .stTabs [role="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        margin-right: 4px;
        background: #fff;
        color: #5F6368;
        transition: all .25s;
        font-weight: 500;
        }
        .stTabs [role="tab"][aria-selected="true"] {
        background: #556CD6;
        color: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .stTabs [data-testid="stTabContent"] {
        padding: 1.5rem;
        background: #fff;
        border: 1px solid #E0E0E0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        }

        /* BUTTONS */
        .stButton > button {
        border-radius: 6px;
        padding: 0.6rem 1.4rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all .2s;
        }
        .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        /* INPUTS & AREAS */
        .stTextArea > div > textarea {
        border-radius: 6px;
        border: 1px solid #DADCE0;
        padding: 0.8rem;
        font-size: 1rem;
        }
        .stFileUploader {
        border: 2px dashed #556CD6;
        border-radius: 8px;
        padding: 1rem;
        background: #fff;
        }

        /* HEADINGS */
        h1, .main-header { font-size: 2.4rem; font-weight: 600; margin-bottom: 1rem; }
        h2, .sub-header { font-size: 1.6rem; font-weight: 600; margin-bottom: 0.75rem; }
        h3 { font-size: 1.25rem; font-weight: 500; }

        /* MISC */
        .stProgress > div > div > div { background-color: #556CD6 !important; }
        .stDivider { border-color: #E0E0E0 !important; margin: 1.5rem 0; }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title with icon
    # st.title("📚 Interactive Knowledge Transfer System")
    st.markdown(
    """
    <div style="text-align:center; margin-bottom:1.5rem;">
      <h1 style="color:#556CD6; font-family:'Inter',sans-serif;">
        🧠 Interactive Knowledge Transfer System
      </h1>
      <p style="color:#5F6368; font-size:1.1rem; font-style:italic;">
        <span style="font-weight:bold; color:#556CD6;">AI</span> at your service!
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

    
    
    # ===== SIDEBAR CONFIGURATION ===== #
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Domain selection
        domain_options = ["Data Science", "Software Engineering", "Electrical Engineering"]
        # Default domain set to Data Science
        if 'selected_domain' not in st.session_state:
            st.session_state.selected_domain = "Data Science"
            
        selected_domain = st.selectbox(
            "Select Knowledge Domain",
            options=domain_options,
            index=domain_options.index(st.session_state.selected_domain),
            help="Choose your domain"
        )
        
        # Updating session state if domain changed
        if selected_domain != st.session_state.selected_domain:
            st.session_state.selected_domain = selected_domain
            # Reset of relevant session state for the new domain
            st.session_state.uploaded_file = None
            st.session_state.processing_complete = False
            st.session_state.knowledge_system = None
            st.session_state.inquiry_started = False
            st.rerun()
        
        # Model selection for expert agent
        model_options = [
            "claude-3-haiku-20240307", 
            "claude-3-sonnet-20240229", 
            "google/gemma-2-9b-it",
            "gemma-dora",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "qwen-finetuned-dora", 
            "deepseek-chat",
            "deepseekcoder-qlora8bit-finetuned",
            "llama-3.2-base",
            "llama-lora-finetuned"
            ]
    
        
        upload = st.session_state.get("uploaded_file") 
        if not upload: 
            st.selectbox( 
                "Expert Model", 
                options=["Upload a file to pick a model"], 
                index=0, 
                disabled=True, 
                help="Model will be assigned based on your document" 
                ) 
        else: 
            # picking exactly one routed_model and showing it read-only
            uploaded_filename = upload.name.lower()
            if "data analyst" in uploaded_filename:
                routed_model = "google/gemma-2-9b-it"
            elif "data engineer" in uploaded_filename:
                routed_model = "gemma-dora"
                # routed_model = "deepseekcoder-qlora8bit-finetuned"
            elif "research engineer" in uploaded_filename:
                routed_model = "claude-3-sonnet-20240229"
            else:
                routed_model = "claude-3-haiku-20240307"

            # Storing in session state for later use during initialization
            st.session_state.expert_model = routed_model

            # Setting the routed model
            st.selectbox(
                "Expert Agent (auto-selected)",
                options=[routed_model],
                index=0,
                disabled=True,
                key="expert_model_display",
                help="Expert agent model chosen based on profile type"
            )

        # model_options = [
        #     "claude-3-haiku-20240307", 
        #     "claude-3-sonnet-20240229", 
        #     "google/gemma-2-9b-it",
        #     "gemma-finetuned-dora",
        #     "Qwen/Qwen2.5-7B-Instruct-Turbo",
        #     "qwen-finetuned-dora", 
        #     "deepseek-chat",
        #     "deepseekcoder-qlora8bit-finetuned",
        #     "llama-3.2-base",
        #     "llama-lora-finetuned"
        #     ]
        # if 'expert_model' not in st.session_state:
        #     st.session_state.expert_model = model_options[0]
            
        # expert_model = st.selectbox(
        #     "Select Expert Model",
        #     options=model_options,
        #     # index=model_options.index(st.session_state.expert_model),
        #     index=0,
        #     help="Choose the expert agent model"
        # )
        
        # # Storing the selected model in session state
        # if st.session_state.expert_model != expert_model:
        #     st.session_state.expert_model = expert_model
        #     # Reset of knowledge system if it exists to use the new model
        #     if 'knowledge_system' in st.session_state and st.session_state.knowledge_system:
        #         st.session_state.knowledge_system = None
        #         st.session_state.processing_complete = False
        
        
        st.divider()
        
        # Session controls
        st.header("Session Controls")
        
        # Session controls
        if st.button("Reset Session", use_container_width=True):
            # Logging final metrics before resetting
            if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                st.session_state.metrics_tracker.log_metrics()
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.success("Session reset successfully!")
            st.rerun()
        
        # Knowledge base viewer button
        if st.button("View Knowledge Base", use_container_width=True):
            view_knowledge_base()
        
        # Displaying metrics if available
        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker is not None:
            # Using the new method instead of the full dashboard
            st.session_state.metrics_tracker.display_sidebar_metrics()
    
    # Main content area with tabs for better navigation
    # tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "💬 Knowledge Transfer", "📊 Results"])
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Upload", "💬 Knowledge Transfer", "📊 Qualitative Metrics", "📊 Quantitative Metrics"])
    
    # ===== DOCUMENT UPLOAD TAB ===== #
    with tab1:
        # Document upload section
        st.header("Document Upload")
        st.markdown("Upload a knowledge transfer document (PDF) to begin the process.")
        
        # Two-column layout for upload and processing
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Knowledge Transfer Document (PDF)",
                type=["pdf"],
                help="Upload the knowledge transition document",
                key="file_uploader_widget"
            )
            if uploaded_file:
                st.session_state.uploaded_file = uploaded_file
        
        with col2:
            if uploaded_file:
                st.success("✅ Document uploaded!")
                process_button = st.button("🚀 Process Document", use_container_width=True)

                # Button to process all S3 documents
                st.write("")  # Adding some spacing
                process_all_button = st.button("🔄 Process All S3 Documents", use_container_width=True)

                if process_button:
                    with st.spinner("Processing document..."):
                        # Saving temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_path = tmp_file.name

                        # Defining domain-specific bucket names
                        domain_buckets = {
                            "Data Science": ("final-team1-raw", "final-team1-processed"),
                            "Software Engineering": ("se-knowledge-raw-team1", "se-knowledge-processed-team1"),
                            "Electrical Engineering": ("ee-knowledge-raw-team1", "ee-knowledge-processed-team1")
                        }

                        # Getting appropriate S3 buckets for the selected domain
                        raw_bucket, processed_bucket = domain_buckets[st.session_state.selected_domain]

                        # Initializing the document processor
                        processor = DocumentProcessor(raw_bucket, processed_bucket, s3_client=s3_client)

                        # Uploading document to S3 before processing
                        s3_client.upload_file(temp_path, raw_bucket, uploaded_file.name)
                        # st.info(f"📤 Uploaded {uploaded_file.name} to raw bucket.")

                        # Adding a progress bar for visual feedback
                        progress_bar = st.progress(0)

                        # Helper function for updating progress
                        def update_progress(progress, status=""):
                            progress_bar.progress(progress)
                            # Only displaying major milestone messages, not every document status
                            if "Building vector store" in status or "All documents processed" in status:
                                st.info(f"Status: {status}")
                            # Logging all messages without displaying them
                            logger.info(f"Progress: {progress:.2f} - {status}")

                        # Determining file size and processing method
                        file_size_mb = uploaded_file.size / (1024 * 1024)
                        try:
                            if file_size_mb > 10:
                                # Large document processing
                                st.info(f"📖 Large document detected ({file_size_mb:.1f} MB). Processing in batches...")

                                chunks = processor.process_large_document(
                                    temp_path, uploaded_file.name, max_pages_per_chunk=20, progress_callback=update_progress
                                )
                            else:
                                # Standard document processing
                                chunks = processor.process_with_progress(
                                    temp_path, uploaded_file.name, progress_callback=update_progress
                                )

                            # After processing, trying to load or update FAISS vector store
                            try:
                                update_progress(0.8, "Loading vector store")
                                vector_store = processor.get_vector_store(new_documents=chunks)
                            except Exception as e:
                                st.warning(f"⚠️ Creating new vector store: {str(e)}")
                                # Ensure embeddings model is loaded
                                if 'embeddings_model' not in st.session_state:
                                    with st.spinner("🔍 Loading embeddings model..."):
                                        st.session_state.embeddings_model = HuggingFaceEmbeddings(
                                            model_name="sentence-transformers/all-mpnet-base-v2"
                                        )
                                # Creating vector store from processed chunks
                                vector_store = FAISS.from_documents(chunks, st.session_state.embeddings_model)

                            # Saving updated FAISS index to S3
                            update_progress(0.9, "Saving FAISS index to S3...")
                            data = vector_store.serialize_to_bytes()
                            s3_client.put_object(
                                Bucket=processed_bucket,
                                Key="vector_store/faiss_index.pickle",
                                Body=data
                            )
                            st.success("✅ FAISS index updated successfully!")

                            # Loading API keys for LLM interaction
                            ANTHROPIC_API_KEY = creds.get("anthropic_api_key") if creds else os.getenv('ANTHROPIC_API_KEY')
                            HUGGINGFACE_API_KEY = creds.get("huggingface_api_key") if creds else os.getenv('HUGGINGFACE_API_KEY')
                            TOGETHER_API_KEY = creds.get("together_api_key") if creds else os.getenv('TOGETHER_API_KEY')

                            # Ensuring required API keys are available
                            if not ANTHROPIC_API_KEY:
                                st.error("❌ Anthropic API key missing. Please add it to your secrets or environment variables.")
                                os.unlink(temp_path)
                                st.stop()

                            # Initializing knowledge system
                            st.session_state.knowledge_system = KnowledgeSystem(
                                vector_store=vector_store,
                                anthropic_api_key=ANTHROPIC_API_KEY,
                                huggingface_api_key=HUGGINGFACE_API_KEY,
                                together_api_key=TOGETHER_API_KEY,
                                dynamodb_resource=dynamodb_resource
                            )

                            # Assigning the selected model to the expert agent
                            model_name = st.session_state.expert_model
                            st.session_state.knowledge_system.expert.set_model(
                                model_name=model_name,
                                anthropic_api_key=ANTHROPIC_API_KEY,
                                huggingface_api_key=HUGGINGFACE_API_KEY,
                                together_api_key=TOGETHER_API_KEY
                            )

                            # Cleaning up temporary file
                            os.unlink(temp_path)

                            update_progress(1.0, "✅ Processing complete")
                            st.session_state.processing_complete = True

                            # Logging the successful processing event
                            session_manager.log_interaction("📄 Document processed successfully!", "success")

                            # Displaying success message and instructions
                            # st.success("✅ Document processed successfully!")
                            st.info("Go to the **Knowledge Transfer** tab to begin the session.")

                        except Exception as e:
                            st.error(f"⚠️ Error processing document: {str(e)}")
                            os.unlink(temp_path)  # Cleanup on error
                            st.stop()

                if process_all_button:
                    if process_all_raw_documents(s3_client, dynamodb_resource, creds):
                        st.info("Go to the **Knowledge Transfer** tab to begin the session.")

                        
    
    # ===== KNOWLEDGE TRANSFER TAB ===== #
    with tab2:
        # Knowledge transfer session
        st.header("Knowledge Transfer Session")
        
        if not st.session_state.processing_complete:
            # Showing informative message if document not processed yet
            st.info("Please upload and process a document in the Document Upload tab first.")
            col1, col2 = st.columns([3, 1])
            with col2:
                # Adding convenience button to go to upload tab
                if st.button("Go to Document Upload", use_container_width=True):
                    st.session_state.active_tab = "Document Upload"
                    st.rerun()
        else:
            # Section for starting knowledge transfer or continuing session
            if not st.session_state.inquiry_started:
                # Session start section with some explanation
                st.markdown("""
                    ### How Knowledge transfer works
                    
                    The knowledge transfer session will help extract critical information from your document.
                    The AI expert agent will ask targeted questions based on the document content, and your responses
                    will be stored in the knowledge base.
                    
                    - You'll answer up to 10 questions about the document
                    - Your responses will be saved to the knowledge base
                    - You can stop the session at any time by clicking on the stop button or typing 'stop and exit'
                """)
                
                # Displaying domain and model in a nice info box
                st.markdown(f"""
                <div class="info-box">
                    <h4>Session Configuration</h4>
                    <p><strong>Domain:</strong> {st.session_state.selected_domain}<br>
                    <strong>Expert Model:</strong> {st.session_state.expert_model}</p>
                </div>
                """, unsafe_allow_html=True)

                # Styling the start button
                st.markdown("""
                <style>
                [data-testid="element-container"]:has([data-testid="stButton"] > button:contains("Start Knowledge Transfer")) button {
                    background-color: #4CAF50;
                    color: white;
                    font-size: 1.1rem;
                    padding: 0.6rem 1.2rem;
                }
                [data-testid="element-container"]:has([data-testid="stButton"] > button:contains("Start Knowledge Transfer")) button:hover {
                    background-color: #3d8b40;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Start button with clear visual emphasis
                start_col1, start_col2, start_col3 = st.columns([1, 2, 1])
                with start_col2:
                    if st.button("🚀 Start Knowledge Transfer Session", use_container_width=True):
                        start_time = time.time()
                        initial_query = "Identify knowledge gaps based on transition document."
                        
                        try:
                            first_question = st.session_state.knowledge_system.expert.ask_questions(
                                initial_query,
                                st.session_state.known_info
                            )
                            retrieval_time = time.time() - start_time
                            
                            st.session_state.last_question = first_question
                            st.session_state.inquiry_started = True
                            
                            # Updating metrics
                            if 'metrics_tracker' not in st.session_state:
                                st.session_state.metrics_tracker = StreamlitMetricsTracker()

                            # Initialize qa_pairs if not present
                            if 'qa_pairs' not in st.session_state:
                                st.session_state.qa_pairs = []
                                
                            st.session_state.metrics_tracker.update_metrics(
                                initial_query,
                                first_question,
                                retrieval_time,
                                0
                            )
                            
                            session_manager.log_interaction(
                                "Knowledge transfer session started",
                                "success"
                            )
                            
                            st.rerun()
                        except Exception as e:
                            session_manager.handle_error(e, "starting_session")
            
            # Active knowledge transfer session
            if st.session_state.inquiry_started:
                # Displaying question counter and progress bar
                # question_count = len(st.session_state.known_info) + 1
                synchronize_question_counts()

                # Get and validate the correct question count
                if 'qa_pairs' in st.session_state:
                    # Current question number is the next one after existing qa_pairs
                    question_count = len(st.session_state.qa_pairs) + 1
                else:
                    question_count = 1
                    
                # Clamp to valid range
                question_count = min(question_count, 10)

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h3>Knowledge Transfer Progress</h3>
                    <span style="font-size: 1.2rem; font-weight: bold;">Question {question_count}/10</span>
                </div>
                """, unsafe_allow_html=True)
                
                progress_value = min(question_count / 10, 1.0)
                st.progress(progress_value)
                
                # Displaying the question in a nice formatted box
                st.markdown(f"""
                <div class="question-box">
                    <h3>Expert Question:</h3>
                    <p style="font-size: 1.1rem;">{st.session_state.last_question}</p>
                </div>
                """, unsafe_allow_html=True)

                # Simple CSS for the user feedback styling
                st.markdown("""
                <style>
                /* Simple button styling */
                [data-testid="element-container"]:has(button[key*="up_feedback"]) button {
                    background-color: #E8F5E9;
                    color: #4CAF50;
                    border: 1px solid #4CAF50;
                }

                [data-testid="element-container"]:has(button[key*="down_feedback"]) button {
                    background-color: #FFEBEE;
                    color: #F44336;
                    border: 1px solid #F44336;
                }
                </style>
                """, unsafe_allow_html=True)

                # Creating a simple feedback key for this question
                feedback_key = f"feedback_{len(st.session_state.get('qa_pairs', []))}"

                # Create a container for the feedback display that we can update
                feedback_container = st.empty()

                # Check if the action was just performed (without refreshing)
                action_just_performed = False

                # Display current feedback if it exists in session state
                if feedback_key in st.session_state:
                    feedback_value = st.session_state[feedback_key]
                    if feedback_value == 1:
                        feedback_container.markdown("""
                        <div style="background-color:#E8F5E9; padding:10px; border-radius:5px; 
                                border-left:4px solid #4CAF50; margin-bottom:15px; text-align:center;">
                        <span style="color:#4CAF50; font-weight:bold;">✓ Upvoted</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        feedback_container.markdown("""
                        <div style="background-color:#FFEBEE; padding:10px; border-radius:5px; 
                                border-left:4px solid #F44336; margin-bottom:15px; text-align:center;">
                        <span style="color:#F44336; font-weight:bold;">✗ Downvoted</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Add custom styling for button container
                    st.markdown("""
                    <style>
                    .feedback-button-container {
                        display: flex;
                        justify-content: center;
                        gap: 30px;
                        margin: 0 auto;
                        max-width: 400px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Create a container div with our custom class
                    st.markdown('<div class="feedback-button-container">', unsafe_allow_html=True)
                    
                    # Create columns with better proportions for button spacing
                    cols = st.columns([1, 1])
                    
                    # Place buttons in columns
                    with cols[0]:
                        if st.button("👍 Upvote", key=f"up_{feedback_key}"):
                            # Update session state without rerunning
                            st.session_state[feedback_key] = 1
                            # Update the qa_pairs entry if it exists
                            if st.session_state.get('qa_pairs') and len(st.session_state['qa_pairs']) > 0:
                                st.session_state['qa_pairs'][-1]["User Rating"] = 1
                                st.session_state['qa_pairs'][-1]["Has User Feedback"] = True
                            
                            # Update the feedback container without refresh
                            feedback_container.markdown("""
                            <div style="background-color:#E8F5E9; padding:10px; border-radius:5px; 
                                    border-left:4px solid #4CAF50; margin-bottom:15px; text-align:center;">
                            <span style="color:#4CAF50; font-weight:bold;">✓ Upvoted</span>
                            </div>
                            """, unsafe_allow_html=True)
                            action_just_performed = True
                            
                    with cols[1]:
                        if st.button("👎 Downvote", key=f"down_{feedback_key}"):
                            # Update session state without rerunning
                            st.session_state[feedback_key] = -1
                            if st.session_state.get('qa_pairs') and len(st.session_state['qa_pairs']) > 0:
                                st.session_state['qa_pairs'][-1]["User Rating"] = -1
                                st.session_state['qa_pairs'][-1]["Has User Feedback"] = True
                            
                            # Update the feedback container without refresh
                            feedback_container.markdown("""
                            <div style="background-color:#FFEBEE; padding:10px; border-radius:5px; 
                                    border-left:4px solid #F44336; margin-bottom:15px; text-align:center;">
                            <span style="color:#F44336; font-weight:bold;">✗ Downvoted</span>
                            </div>
                            """, unsafe_allow_html=True)
                            action_just_performed = True
                    
                    # Close the container div
                    st.markdown('</div>', unsafe_allow_html=True)

                # If an action was just performed, update any displays as needed without refresh
                if action_just_performed:
                    # This will run AFTER the button click but BEFORE any rerun
                    pass
                 
                # Adding help text for stop command
                st.caption("Type 'stop and exit' to end the session")
                
                # User response section
                response_col1, response_col2 = st.columns([3, 1])
                
                with response_col1:
                    # Adding a unique key for the text input that changes with each response
                    input_key = f"user_response_{st.session_state.response_counter}"
                    user_response = st.text_area(
                        "Your response:",
                        key=input_key,
                        height=150,
                        placeholder="Enter your response here..."
                    )
                
                with response_col2:
                    # Submit button with clear visual emphasis
                    submit_response = st.button("📤 Submit Response", use_container_width=True)
                    
                    st.write("")  # Spacer
                    
                    # Adding a stop button
                    if st.button("⏹️ Stop Session", use_container_width=True):
                        st.session_state.inquiry_started = False
                        st.session_state.session_stopped = True
                        st.session_state.last_question = None
                        st.success("Session ended by user.")
                        
                        # Logging final metrics
                        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                            st.session_state.metrics_tracker.log_metrics()
                        
                        session_manager.log_interaction(
                            "Session ended by user command",
                            "info"
                        )
                        
                        time.sleep(1)
                        st.rerun()
                
                # Processing response when submitted
                if submit_response:
                    if user_response.strip():
                        # Checking for stop command
                        if user_response.lower().strip() == 'stop and exit':
                            st.success("🛑 Session ended by user command.")
                            st.session_state.inquiry_started = False
                            st.session_state.session_stopped = True  # ✅ Flag to prevent further submission

                            if len(st.session_state.get("qa_pairs", [])) >= 10:
                                st.warning("You have reached the maximum of 10 questions for this session.")
                                st.session_state.session_stopped = True
                                return
                            
                            # Logging final metrics
                            if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                st.session_state.metrics_tracker.log_metrics()
                            
                            # Display final metrics
                            st.subheader("Final Session Summary")
                            st.write(f"Total questions answered: {len(st.session_state.known_info)}")
                            
                            if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                st.session_state.metrics_tracker.display_metrics_dashboard()
                            
                            session_manager.log_interaction(
                                "Session ended by user command",
                                "info"
                            )
                            
                            # Adding option to start new session
                            if st.button("Start New Session"):
                                st.session_state.inquiry_started = False
                                st.session_state.known_info = []
                                st.session_state.last_question = None
                                st.session_state.response_counter = 0
                                st.rerun()
                        
                        else:
                            try:
                                with st.spinner("Processing your response..."):
                                    response_start_time = time.time()
                                    
                                    # ✅ Prevent continuing after 10 questions
                                    if len(st.session_state.known_info) >= 10:
                                        st.success("✅ You've completed all 10 questions. Knowledge transfer session ended.")
                                        st.session_state.inquiry_started = False

                                        # Log final metrics
                                        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                            st.session_state.metrics_tracker.log_metrics()
                                            st.session_state.metrics_tracker.display_metrics_dashboard()

                                        # Optional restart
                                        if st.button("Start New Session"):
                                            st.session_state.inquiry_started = False
                                            st.session_state.known_info = []
                                            st.session_state.last_question = None
                                            st.session_state.response_counter = 0
                                            st.rerun()

                                        st.stop()  # ✅ clean halt without breaking Streamlit
                                    
                                    # Using this approach for more efficient batch processing:
                                    try:
                                        # Creating a batch of items to update
                                        current_model = st.session_state.expert_model
                                        batch_items = [(st.session_state.last_question, user_response)]
                                        
                                        # Adding any pending items from previous sessions if available
                                        if 'pending_kb_updates' in st.session_state:
                                            batch_items.extend(st.session_state.pending_kb_updates)
                                            st.session_state.pending_kb_updates = []
                                        
                                        # Batch update the knowledge base
                                        kb_update_status = st.session_state.knowledge_system.knowledge_manager.batch_update_knowledge_base(batch_items)
                                    except Exception as e:
                                        # Fall back to single update if batch fails
                                        kb_update_status = st.session_state.knowledge_system.knowledge_manager.update_knowledge_base(
                                            st.session_state.last_question,
                                            user_response, 
                                            current_model
                                        )
                                        logger.warning(f"Falling back to single item update: {str(e)}")

                                    # === Initialize eval_summary before first metric logging ===
                                    if "eval_summary" not in st.session_state and "knowledge_system" in st.session_state and st.session_state.knowledge_system:
                                        selected_model_name = st.session_state.knowledge_system.expert.model_name
                                        st.session_state.eval_summary = {
                                            "model": selected_model_name,
                                            "total_questions": 0,
                                            "duplicates": 0,
                                            "relevance_scores": [],
                                            "diversity_scores": [],
                                            "user_ratings": []  # Added for user feedback
                                        }
                                    # # ⬇️ Append current Q&A and model
                                    # st.session_state.known_info.append(st.session_state.last_question)
                                    # st.session_state.known_info.append(user_response)
                                    # st.session_state.models_used_per_step.append(current_model)

                                    # 🔍 Debug log
                                    print("🔍 known_info:", st.session_state.known_info)
                                    print("🔍 models_used_per_step:", st.session_state.models_used_per_step)
                                    
                                    # Getting next question from the knowledge system
                                    next_question = st.session_state.knowledge_system.expert.ask_questions(
                                        st.session_state.last_question,
                                        st.session_state.known_info + [user_response]
                                    )
                                    
                                    # # ADD THE EVALUATION DATA RETRIEVAL HERE
                                    # if hasattr(st.session_state.knowledge_system.expert, "last_question_evaluation"):
                                    #     evaluation_data = st.session_state.knowledge_system.expert.last_question_evaluation
                                    #     print(f"Retrieved evaluation data: {evaluation_data}")
                                    # else:
                                    #     evaluation_data = {"score": 0}
                                    #     print("No last_question_evaluation attribute found")
                                        
                                    evaluation_data = getattr(st.session_state.knowledge_system.expert, "last_question_evaluation", {"score": 0})
                                    print(f"Retrieved evaluation data: {evaluation_data}")
                                    
                                    # Evaluation of RAG-generated questions: wandb log metrics
                                    from metrics_tracker import log_rag_question_metrics

                                    if 'rag_question_embeddings' not in st.session_state:
                                        st.session_state.rag_question_embeddings = []

                                    # Get the context used by the expert agent for this question
                                    context_used = "\n".join(st.session_state.knowledge_system.expert.last_retrieved_context)
                                    model_name = st.session_state.knowledge_system.expert.model_name if st.session_state.knowledge_system else "Unknown"
                                    # Log evaluation and store embedding for diversity computation
                                    metrics = log_rag_question_metrics(
                                    question=next_question,
                                    context=context_used,
                                    step=st.session_state.metrics['questions_asked'],
                                    previous_embeddings=st.session_state.rag_question_embeddings,
                                    #previous_questions=st.session_state.known_info # to track all the asked questions
                                    #model_used=st.session_state.model_name
                                    previous_questions=[qap["Question"] for qap in st.session_state.qa_pairs] if "qa_pairs" in st.session_state else []
                                    )   
                                    # update_model_eval_leaderboard(summary)
                                    # Initialize qa_pairs if not present
                                    if 'qa_pairs' not in st.session_state:
                                        st.session_state.qa_pairs = []

                                    # Appending full Q&A with metadata and metrics
                                    st.session_state.qa_pairs.append({
                                        "Step": len(st.session_state.qa_pairs) + 1, 
                                        "Model Used": current_model,
                                        "Question": st.session_state.last_question,
                                        "Answer": user_response,
                                        "Context Relevance": metrics["context_relevance"],
                                        "Diversity": metrics["diversity"],
                                        "Specificity": metrics["specificity"],
                                        "Final Score": round(
                                            0.4 * (metrics.get("context_relevance", 0) or 0) + 
                                            0.3 * (metrics.get("diversity", 0) or 0) + 
                                            0.3 * (metrics.get("specificity", 0) or 0), 3
                                        ), 
                                        "User Rating": 0,  # New field to store the rating (-1, 0, +1) 
                                        "Has User Feedback": False  # Flag to track if user has provided feedback

                                    })

                                    # Explicitly set the questions_asked metric to match qa_pairs length
                                    #st.session_state.metrics['questions_asked'] = len(st.session_state.qa_pairs)
                                    # Explicitly synchronize all question counts based on qa_pairs
                                    synchronize_question_counts()
                                    
                                    # Keep both qa_pairs and known_info in sync
                                    st.session_state.known_info.append(st.session_state.last_question)
                                    st.session_state.known_info.append(user_response)

                                    # === Update or Reset eval_summary if model switched ===
                                    current_model = st.session_state.knowledge_system.expert.model_name
                                    existing_summary_model = st.session_state.eval_summary.get("model") if "eval_summary" in st.session_state else None

                                    if current_model != existing_summary_model:
                                        st.session_state.eval_summary = {
                                            "model": current_model,
                                            "total_questions": 0,
                                            "duplicates": 0,
                                            "relevance_scores": [],
                                            "diversity_scores": [],
                                            "user_ratings": []  # Added for user feedback
                                        }


                                    # Update evaluation summary
                                    summary = st.session_state.eval_summary
                                    summary["total_questions"] += 1
                                    summary["relevance_scores"].append(metrics["context_relevance"])
                                    summary["diversity_scores"].append(metrics["diversity"])

                                    # Collecting any user rating from the previous question
                                    if len(st.session_state.qa_pairs) > 0 and st.session_state.qa_pairs[-1].get("Has User Feedback", False):
                                        summary["user_ratings"].append(st.session_state.qa_pairs[-1].get("User Rating", 0))

                                    if metrics["is_duplicate"]:
                                        summary["duplicates"] += 1
                                        # st.warning(f"⚠️ Repeated question detected:\n**{next_question}**")
                                    
                                    # === Push to leaderboard
                                    update_model_eval_leaderboard(summary)

                                    # Storing embedding for diversity in next steps
                                    st.session_state.rag_question_embeddings.append(metrics["embedding"])
                                    
                                    # Calculating response time
                                    response_time = time.time() - response_start_time
                                    
                                    #Including evaluation data in metrics
                                    evaluation_data = getattr(st.session_state.knowledge_system.expert, "last_question_evaluation", None)
                                    if evaluation_data is None:  
                                        evaluation_data = {"score": 0, "feedback": "No evaluation available"}
                                    
                                    # Updating metrics
                                    if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                        # Get context data if available
                                        context_data = { 
                                            "relevant_chunks": st.session_state.knowledge_system.expert.last_retrieved_context 
                                        }

                                                
                                        st.session_state.metrics_tracker.update_metrics(
                                            st.session_state.last_question,
                                            user_response,
                                            retrieval_time=0.0,
                                            response_time=response_time,
                                            evaluation_data=evaluation_data,  # Use the evaluation data we retrieved
                                            context_data=context_data, 
                                            user_rating=st.session_state.qa_pairs[-1].get("User Rating") if len(st.session_state.qa_pairs) > 0 and st.session_state.qa_pairs[-1].get("Has User Feedback", False) else None
                                        )
                                    
                                    # Checking if knowledge transfer is complete
                                    if next_question == "No further questions." or "No further questions" in next_question:
                                        # Visual celebration
                                        st.balloons()  
                                        st.success("Knowledge transfer complete!")
                                        st.session_state.inquiry_started = False
                                        
                                        # Logging final metrics
                                        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                            st.session_state.metrics_tracker.log_metrics()
                                        
                                        session_manager.log_interaction(
                                            "Knowledge transfer session completed successfully",
                                            "success"
                                        )
                                        
                                        # Displaying final metrics
                                        st.subheader("Final Session Metrics")
                                        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                            st.session_state.metrics_tracker.display_metrics_dashboard()
                                        
                                        # Add option to start new session
                                        if st.button("Start New Session"):
                                            st.session_state.inquiry_started = False
                                            st.session_state.known_info = []
                                            st.session_state.last_question = None
                                            st.session_state.response_counter = 0
                                            st.rerun()
                                    else:
                                        # Updating session state for next iteration
                                        st.session_state.last_question = next_question
                                        st.session_state.known_info.append(user_response)
                                        
                                        session_manager.log_interaction(
                                            f"Response recorded and new question generated. {kb_update_status}",
                                            "info"
                                        )
                                        
                                        # Forcing streamlit to rerun to show the new question
                                        st.rerun()
                                
                            except Exception as e:
                                session_manager.handle_error(e, "processing_response")
                    else:
                        st.warning("Please provide a response before submitting.")
    
    with tab3:
        # Results and metrics tab
        st.header("⭐ Qualitative Metrics")
            
        # First display avg metrics side by side
        col1, col2 = st.columns(2)
        
        with col1:
            if ('evaluation_scores' in st.session_state.metrics
                and st.session_state.metrics['evaluation_scores']):
                avg_score = sum(st.session_state.metrics['evaluation_scores']) \
                            / len(st.session_state.metrics['evaluation_scores'])
                st.metric("Average Question Quality", f"{avg_score:.1f}/10")
            else:
                st.metric("Average Question Quality", "N/A", "No evaluations yet")
                
        with col2:
            # User Feedback Section
            ratings = [qa.get("User Rating") for qa in st.session_state.qa_pairs if qa.get("Has User Feedback")]
            if ratings:
                avg = sum(ratings)/len(ratings)
                st.metric("Average User Rating", f"{avg:.2f} (-1 to +1)")
            else:
                st.metric("Average User Rating", "N/A")


        # Viewing knowledge base section
        st.subheader("Knowledge Base")
        st.markdown("View the knowledge captured during this session and previous sessions.")
        
        # View knowledge base button
        if st.button("View Knowledge Base", key="view_kb_results"):
            view_knowledge_base()

    # === Tab 4: Evaluation Dashboard ===
    with tab4:
        st.header("Quantitative Metrics")

        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
            st.session_state.metrics_tracker.display_metrics_dashboard()
        else:
            st.warning("No evaluation metrics available yet. Run a knowledge transfer session first.")


        if "qa_pairs" in st.session_state and st.session_state.qa_pairs:
            # Converting qa_pairs to a dataframe for display
            question_logs = []
            for qa_pair in st.session_state.qa_pairs:
                # Formatting data for display
                entry = {
                    "Step": qa_pair.get("Step", 0),
                    "Model Used": qa_pair.get("Model Used", "Unknown"),
                    "Question": qa_pair.get("Question", ""),
                    "Context Relevance %": qa_pair.get("Context Relevance", 0) * 100,
                    "Diversity %": qa_pair.get("Diversity", 0) * 100,
                    "Specificity": qa_pair.get("Specificity", 0) * 100,
                    "Question Length": len(qa_pair.get("Question", "").split()),
                    "Final Score %": qa_pair.get("Final Score", 0) * 100
                }
                question_logs.append(entry)
                
            # Storing for potential future use
            st.session_state.question_logs = question_logs
            
            df = pd.DataFrame(question_logs)
            
            st.subheader("📈 Session-Level Metric Averages")
            
            if not df.empty:
                # Calculating averages - make sure all values are proper percentages
                avg_context = df["Context Relevance %"].mean() if "Context Relevance %" in df.columns and len(df) > 0 else 0
                avg_diversity = df["Diversity %"].mean() if "Diversity %" in df.columns and len(df) > 0 else 0
                avg_specificity = df["Specificity"].mean() if "Specificity" in df.columns and len(df) > 0 else 0
                avg_q_len = df["Question Length"].mean() if "Question Length" in df.columns else 0
                avg_final_score = df["Final Score %"].mean() if "Final Score %" in df.columns else 0

                # Displaying metrics in rows
                # Row 1: Four main metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Context Relevance", f"{avg_context:.2f}%")
                with col2:
                    st.metric("Question Specificity", f"{avg_specificity:.2f}%")
                with col3:
                    st.metric("Question Length", f"{avg_q_len:.1f} words")
                with col4:
                    st.metric("Diversity", f"{avg_diversity:.2f}%")
                
                # Row 2: Final score centered and larger
                st.markdown(f"""
                <div style="text-align:center; margin-top:10px;">
                    <h3>Overall Final Score</h3>
                    <p style="font-size:26px; font-weight:bold; color:#1f77b4;">{avg_final_score:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Combined metrics chart
                st.subheader("🔄 Combined Metrics Visualization")
                
                # Creating a radar chart with all metrics
                # Create normalized versions of metrics for consistent visualization
                metrics_to_include = ["Context Relevance %", "Diversity %", "Specificity"]
                
                # Add information density to radar chart if available
                if 'metrics' in st.session_state and 'info_density' in st.session_state.metrics and st.session_state.metrics['info_density']: 
                    df["Information Density"] = sum(st.session_state.metrics['info_density']) / len(st.session_state.metrics['info_density']) * 100
                    metrics_to_include.append("Information Density")
                
                # Final score should already be 0-100 
                metrics_to_include.append("Final Score %")
                
                # Ensuring metrics are present in df
                metrics_to_include = [m for m in metrics_to_include if m in df.columns]
                
                if metrics_to_include:
                    # Single aggregate "system" performance 
                    avg_values = [df[m].mean() if m in df.columns else 0 for m in metrics_to_include] 
                    fig = go.Figure(go.Scatterpolar( 
                        r=avg_values, 
                        theta=metrics_to_include, 
                        fill='toself', 
                        name="Overall Performance" 
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        title="Metric Performance Comparison"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)


            # Tag showing total answered and total models used
            st.markdown("---")
            # Using questions_asked from metrics to ensure consistency
            questions_asked = st.session_state.metrics.get('questions_asked', 0)
            st.markdown(f"✅ **Total Questions Answered**: `{questions_asked}`")
            if st.button("🧹 Clear Evaluation Logs"):
                st.session_state.question_logs = []
                st.session_state.qa_pairs = []
                st.success("Cleared question evaluations")
                time.sleep(1)
                st.rerun()
        else:
            st.info("⚠️ No evaluation data available. Run a session first.")

    # Displaying interaction logs
    if 'agent_logs' in st.session_state and st.session_state.agent_logs:
        with st.expander("Session Logs", expanded=False):
            for log in st.session_state.agent_logs:
                st.markdown(log)
    
    # Displaying error logs if any
    if 'error_log' in st.session_state and st.session_state.error_log:
        with st.expander("Error Logs", expanded=False):
            for error in st.session_state.error_log:
                st.error(error)

# Getting secrets from Streamlit if available
def get_streamlit_secrets():
    try:
        return {
            "aws_access_key_id": st.secrets["aws"]["aws_access_key_id"],
            "aws_secret_access_key": st.secrets["aws"]["aws_secret_access_key"],
            "region_name": st.secrets["aws"]["region_name"],
            "anthropic_api_key": st.secrets["apis"]["anthropic_api_key"],
            "huggingface_api_key": st.secrets["apis"]["huggingface_api_key"],
            "together_api_key": st.secrets["apis"]["together_api_key"]
        }
    except Exception as e:
        # Not using st.error or st.warning here to avoid error
        logger.warning(f"Error accessing secrets: {e}")
        return None
    
# Function to process all documents in the raw bucket
def process_all_raw_documents(s3_client, dynamodb_resource, creds):
    with st.spinner("Processing all documents in bucket..."):
        domain_buckets = {
            "Data Science": ("final-team1-raw", "final-team1-processed"),
            "Software Engineering": ("se-knowledge-raw-team1", "se-knowledge-processed-team1"),
            "Electrical Engineering": ("ee-knowledge-raw-team1", "ee-knowledge-processed-team1")
        }
        
        # Get appropriate S3 buckets for the selected domain
        raw_bucket, processed_bucket = domain_buckets[st.session_state.selected_domain]
        
        # Initialize the document processor
        processor = DocumentProcessor(raw_bucket, processed_bucket, s3_client=s3_client)
        
        # Adding a progress bar for visual feedback
        progress_bar = st.progress(0)
        
        # Helper function for updating progress
        def update_progress(progress, status=""):
            progress_bar.progress(progress)
            # Only showing important milestone messages
            if progress > 0.7 and ("Loading vector store" in status or "Processing complete" in status):
                st.info(f"Status: {status}")
        
        # Process all documents
        try:
            chunks = processor.process_all_raw_documents(progress_callback=update_progress)
            st.success(f"✅ Successfully processed {len(chunks)} chunks from all documents")
            
            # After processing, ensure we have the vector store loaded
            vector_store = processor.get_vector_store()
            
            # Initialize the knowledge system with this vector store
            ANTHROPIC_API_KEY = creds.get("anthropic_api_key") if creds else os.getenv('ANTHROPIC_API_KEY')
            HUGGINGFACE_API_KEY = creds.get("huggingface_api_key") if creds else os.getenv('HUGGINGFACE_API_KEY')
            TOGETHER_API_KEY = creds.get("together_api_key") if creds else os.getenv('TOGETHER_API_KEY')
            
            st.session_state.knowledge_system = KnowledgeSystem(
                vector_store=vector_store,
                anthropic_api_key=ANTHROPIC_API_KEY,
                huggingface_api_key=HUGGINGFACE_API_KEY,
                together_api_key=TOGETHER_API_KEY,
                dynamodb_resource=dynamodb_resource
            )
            
            # Assigning the selected model to the expert agent - using the session state value or maintain the routed model
            model_name = st.session_state.expert_model
            if not model_name:
                # If no model set yet, use the default
                model_name = "claude-3-haiku-20240307"
                st.session_state.expert_model = model_name

            st.session_state.knowledge_system.expert.set_model(
                model_name=model_name,
                anthropic_api_key=ANTHROPIC_API_KEY,
                huggingface_api_key=HUGGINGFACE_API_KEY,
                together_api_key=TOGETHER_API_KEY
            )
            
            st.session_state.processing_complete = True
            return True
            
        except Exception as e:
            st.error(f"⚠️ Error processing documents: {str(e)}")
            return False

# ===== SESSION STATE MANAGEMENT ===== #
# Initializing all session state variables
def initialize_session_state(dynamodb_resource=None, cloudwatch_client=None):
    if 'metrics_tracker' not in st.session_state:
        st.session_state.metrics_tracker = StreamlitMetricsTracker(
            dynamodb_resource=dynamodb_resource, 
            cloudwatch_client=cloudwatch_client
        )

    # Initializing session state variables with defaults
    session_vars = {
        'uploaded_file': None,
        'knowledge_system': None,
        'inquiry_started': False,
        'processing_complete': False,
        'last_question': None,
        'known_info': [],
        'agent_logs': [],
        'error_log': [],
        'response_counter': 0,
        'selected_domain': "Data Science",
        'models_used_per_step': [],
        'pending_kb_updates': [],
        'embeddings_model': None,
        'session_stopped': False,
        'qa_pairs': [],
        'rag_question_embeddings': []
    }
    
    # Setting all variables that don't exist yet
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
            
    if 'qa_pairs' in st.session_state:
        # If qa_pairs exists, use its length for questions_asked
        st.session_state.metrics['questions_asked'] = len(st.session_state.qa_pairs)
    else:
        # Otherwise reset to 0
        st.session_state.metrics['questions_asked'] = 0
            
    logger.info("Session state initialized")


# ===== DOCUMENT PROCESSING FUNCTIONS ===== #
# Caching results for 5 minutes
@st.cache_data(ttl=300)  
# Getting knowledge base items with caching for better performance
def get_knowledge_base_items(table_name, credentials=None):
    try:
        # Using credentials if provided, otherwise use environment variables
        if credentials:
            kb_table = boto3.resource(
                'dynamodb', 
                region_name='us-east-1',
                aws_access_key_id=credentials.get("aws_access_key_id"),
                aws_secret_access_key=credentials.get("aws_secret_access_key")
            ).Table(table_name)
        else:
            kb_table = boto3.resource(
                'dynamodb', 
                region_name='us-east-1',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            ).Table(table_name)
        
        response = kb_table.scan()
        return response.get('Items', [])
    except Exception as e:
        logger.error(f"Failed to access table {table_name}: {str(e)}")
        return []

# Displaying contents of the knowledge base
def view_knowledge_base():
    try:
        # Getting domain-specific table name based on the selected domain
        domain_tables = {
            "Data Science": "knowledge_base",
            "Software Engineering": "se_knowledge_base",
            "Electrical Engineering": "ee_knowledge_base"
        }

        # Defaulting to the base table if none found
        table_name = domain_tables.get(st.session_state.selected_domain, "knowledge_base")

        # Getting credentials
        creds = None
        try:
            creds = {
                "aws_access_key_id": st.secrets["aws"]["aws_access_key_id"],
                "aws_secret_access_key": st.secrets["aws"]["aws_secret_access_key"], 
                "region_name": os.getenv('AWS_REGION', 'us-east-1')
            }
        except:
            # Fallback to environment variables
            pass
        
        # Using cached function to get items
        with st.spinner("Loading knowledge base..."):
            items = get_knowledge_base_items(table_name, creds)
            
            # Trying default table if domain table is empty
            if not items and table_name != "knowledge_base":
                st.info(f"No entries found in {table_name}. Checking default knowledge base...")
                items = get_knowledge_base_items("knowledge_base", creds)
    
        if items:
            st.subheader(f"Knowledge Base: {st.session_state.selected_domain}")
            # Creating a df for easier viewing
            kb_data = []
            for item in items: 
                kb_data.append({
                    "ID": item.get('id','N/A'), 
                    "Question": item.get('question', item.get('content', 'N/A')), 
                    "Answer": item.get('answer', item.get('analysis', 'N/A')), 
                    "Timestamp": item.get('timestamp', 'N/A')
                })
            
            # Showing as a dataframe 
            kb_df = pd.DataFrame(kb_data) 
            st.dataframe(kb_df, use_container_width=True)

            # Also showing detailed view with expandable sections
            st.subheader("Detailed View")
            # Limiting number of items shown for better performance
            max_items = 50
            if len(items) > max_items:
                st.info(f"Showing {max_items} most recent entries out of {len(items)} total entries")
                items = sorted(items, key=lambda x: x.get('timestamp', ''), reverse=True)[:max_items]
                
            for item in items:
                question_text = item.get('question', item.get('content', 'N/A'))
                # Truncating long questions for the expander header
                display_text = question_text[:50] + "..." if len(question_text) > 50 else question_text
                
                with st.expander(f"{item.get('id', 'Entry')}: {display_text}", expanded=False):
                    st.markdown("#### Question:")
                    st.write(question_text)
                    
                    st.markdown("#### Answer:")
                    st.write(item.get('answer', item.get('analysis', 'N/A')))
                    
                    st.markdown("#### Metadata:")
                    st.write(f"Timestamp: {item.get('timestamp', 'N/A')}")
                    if 'metadata' in item:
                        st.json(item['metadata'])

        else:
            st.info(f"Knowledge base for {st.session_state.selected_domain} is empty")
    except Exception as e:
        st.error(f"Error accessing knowledge base: {str(e)}")
        # Showing detailed error for debugging
        st.exception(e)  

if __name__ == "__main__":
    main()
