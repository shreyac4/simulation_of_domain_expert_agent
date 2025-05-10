import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import boto3
from decimal import Decimal
import logging
import uuid
import os
import textstat
import re
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

# Loading sentence transformer model - doing this once at module level
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {str(e)}")
    embedder = None


# Evaluation

# Calculating semantic similarity between context and question
def compute_context_relevance(context, question):
    if not embedder:
        logger.warning("Embedder not available for context relevance computation")
        return 0.5
        
    try:
        embeddings = embedder.encode([context, question], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return round(similarity, 4)
    except Exception as e:
        logger.error(f"Error computing context relevance: {str(e)}")
        return 0.0

# Calculating specificity ratio of unique words to total words
def compute_specificity(question):
    try:
        words = question.split()
        if not words:
            return 0.0
        return min(1.0, len(set(words)) / len(words))
    except Exception as e:
        logger.error(f"Error computing specificity: {str(e)}")
        return 0.0

# Counting words in question
def compute_question_length(question):
    return len(question.split())

# Calculating semantic diversity compared to previous questions
def compute_diversity(current_embedding, previous_embeddings):
    if not previous_embeddings or len(previous_embeddings) == 0:
        return 1.0
    try:
        similarities = [util.pytorch_cos_sim(current_embedding, prev).item() for prev in previous_embeddings]
        if not similarities:  # Added check for empty list after calculation
            return 1.0
        max_sim = max(similarities)
        return 1 - max_sim  # Safer calculation
    except Exception as e:
        logger.error(f"Error computing diversity: {str(e)}")
        return 1.0

# Detecting if question is semantically similar to previous ones
def is_semantically_duplicate(current_embedding, previous_embeddings, threshold=0.92):
    if not previous_embeddings:
        return False
    try:
        similarities = [util.pytorch_cos_sim(current_embedding, prev).item() for prev in previous_embeddings]
        return max(similarities) > threshold if similarities else False
    except Exception as e:
        logger.error(f"Error checking semantic duplicates: {str(e)}")
        return False
    
# Logging comprehensive metrics for a RAG-generated question
def log_rag_question_metrics(question, context, step, previous_embeddings=[], previous_questions=[]):
    try:
        if not embedder:
            logger.warning("Embedder not available for question metrics")
            return {
                "embedding": None,
                "context_relevance": 0.5,
                "diversity": 1.0,
                "is_duplicate": False,
                "question_length": len(question.split()),
                "specificity": 0.5,
                "final_score": 0.5
            }
            
        current_embedding = embedder.encode(question, convert_to_tensor=True)

        # Computing all metrics
        context_relevance = compute_context_relevance(context, question)
        question_length = compute_question_length(question)
        specificity = compute_specificity(question)
        diversity = compute_diversity(current_embedding, previous_embeddings)

        # Checking for duplicate questions (exact and semantic)
        is_exact_duplicate = question in previous_questions
        is_semantic_duplicate_result = is_semantically_duplicate(current_embedding, previous_embeddings)
        is_duplicate = is_exact_duplicate or is_semantic_duplicate_result

        # Calculating final weighted score
        final_score = (
            0.4 * context_relevance +
            0.3 * diversity +
            0.3 * specificity
        )

        # Returning comprehensive metrics
        return {
            "embedding": current_embedding,
            "context_relevance": context_relevance,
            "diversity": diversity,
            "is_duplicate": is_duplicate,
            "question_length": question_length,
            "specificity": specificity,
            "final_score": final_score
        }
    except Exception as e:
        logger.error(f"Error logging RAG question metrics: {str(e)}")
        return {
            "embedding": None,
            "context_relevance": 0,
            "diversity": 0,
            "is_duplicate": False,
            "question_length": 0,
            "specificity": 0,
            "final_score": 0
        }

# Function to initialize and return a DynamoDB resource 
# Using AWS credentials from environment variables or Streamlit secrets
def get_dynamodb_resource():
    try:
        # If using Streamlit secrets (recommended)
        import streamlit as st
        if "aws" in st.secrets:
            return boto3.resource(
                "dynamodb",
                region_name=st.secrets["aws"]["region_name"],
                aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
                aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
            )

        # Fallback: Use environment variables
        return boto3.resource("dynamodb", region_name="us-east-1")
    
    except Exception as e:
        logger.error(f"❌ Failed to initialize DynamoDB resource: {str(e)}")
        return None

# StreamlitMetricsTracker: Tracks, logs, and visualizes knowledge transfer metrics

# This module handles:
    # 1. Tracking knowledge transfer performance metrics
    # 2. Persisting metrics to DynamoDB and CloudWatch
    # 3. Visualizing metrics in the Streamlit interface
    # 4. Comparing current metrics with historical performance

class StreamlitMetricsTracker:
    def __init__(self, dynamodb_resource=None, cloudwatch_client=None):
        try:
            self._initialize_session_metrics()
            
            # Using provided clients or create default ones
            self.dynamodb = dynamodb_resource or get_dynamodb_resource()
            self.cloudwatch = cloudwatch_client or boto3.client('cloudwatch', region_name='us-east-1')
            self.metrics_table = None  # Lazy initialization
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        except Exception as e:
                logger.error(f"❌ Error initializing MetricsTracker: {str(e)}")
                raise

    # Initializing metrics in session state
    def _initialize_session_metrics(self):
        try:
            if 'metrics' not in st.session_state:
                st.session_state.metrics = {
                'retrieval_times': [],
                'response_times': [],
                'context_utilization': [],
                'questions_asked': 0,
                'info_density': [],
                'response_specificity': [],
                'response_lengths': [],
                'evaluation_scores': [],
                'evaluations': [],
                # Question quality metrics
                'context_relevance': [],
                'question_specificity': [], 
                'question_length': [],
                'diversity': [],
                'duplicate_count': 0,
                'question_quality_scores': [],
                'evaluation_metrics': {
                    'questions': [],
                    'context_relevance': [],
                    'answer_quality': []
                }
            }
            logger.info("Session metrics initialized")
        except Exception as e:
            logger.error(f"Error initializing session metrics: {str(e)}")
            raise

    # Function to get or create DynamoDB table for metrics
    def get_or_create_metrics_table(self):
        if self.metrics_table:
            return self.metrics_table
        try:
            table_name = 'rag_evaluation_metrics'
            existing_tables = self.dynamodb.meta.client.list_tables()["TableNames"]
            if table_name not in existing_tables:
                self.metrics_table = self.dynamodb.create_table(
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
                self.metrics_table.wait_until_exists()
            else:
                self.metrics_table = self.dynamodb.Table(table_name)
            return self.metrics_table
        except Exception as e:
            logger.error(f"⚠️ DynamoDB table issue: {str(e)}")
            return None

    # Logging metrics to CloudWatch
    def log_to_cloudwatch(self, metrics_data):
        try:
            metric_items = [
                {
                    'MetricName': key, 
                    'Value': float(value), 
                    'Unit': 'Count' if key == 'total_questions' else 'Seconds',  
                    'Dimensions': [{'Name': 'SessionId', 'Value': self.session_id}]
                }
                for key, value in metrics_data['interaction_metrics'].items()
            ]
            self.cloudwatch.put_metric_data(Namespace='RAGSystem', MetricData=metric_items)
            logger.info("✅ Metrics logged to CloudWatch.")
        except Exception as e:
            logger.error(f"❌ Failed to log to CloudWatch: {str(e)}")

    # Logging metrics to DynamoDB and CloudWatch
    def log_metrics(self):
        # Getting metrics table, creating it if needed
        self.metrics_table = self.get_or_create_metrics_table()
        
        # Handling case where DynamoDB is unavailable
        if not self.metrics_table:
            logger.warning("DynamoDB unavailable. Metrics will not be persisted.")
            return False
            
        try:
            # Safety checks for metrics data
            retrieval_times = st.session_state.metrics.get('retrieval_times', [])
            response_times = st.session_state.metrics.get('response_times', [])
            
            # Preparing metrics item with safe calculations
            metrics_item = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'interaction_metrics': {
                    'average_retrieval_time': Decimal(str(sum(retrieval_times) / max(len(retrieval_times), 1))),
                    'average_response_time': Decimal(str(sum(response_times) / max(len(response_times), 1))),
                    'total_questions': Decimal(str(st.session_state.metrics['questions_asked']))
                }
            }
            
            # Addng additional content metrics if available
            if 'info_density' in st.session_state.metrics and st.session_state.metrics['info_density']:
                density_values = st.session_state.metrics['info_density']
                metrics_item['interaction_metrics']['average_info_density'] = Decimal(str(sum(density_values) / len(density_values)))
                
            if 'context_utilization' in st.session_state.metrics and st.session_state.metrics['context_utilization']:
                util_values = st.session_state.metrics['context_utilization']
                metrics_item['interaction_metrics']['average_context_utilization'] = Decimal(str(sum(util_values) / len(util_values)))
            
            # Adding question quality metrics if available
            if 'context_relevance' in st.session_state.metrics and st.session_state.metrics['context_relevance']:
                relevance_values = st.session_state.metrics['context_relevance']
                metrics_item['interaction_metrics']['average_context_relevance'] = Decimal(str(sum(relevance_values) / len(relevance_values)))
                
            if 'diversity' in st.session_state.metrics and st.session_state.metrics['diversity']:
                diversity_values = st.session_state.metrics['diversity']
                metrics_item['interaction_metrics']['average_diversity'] = Decimal(str(sum(diversity_values) / len(diversity_values)))
            
            if 'question_quality_scores' in st.session_state.metrics and st.session_state.metrics['question_quality_scores']:
                quality_scores = st.session_state.metrics['question_quality_scores']
                metrics_item['interaction_metrics']['average_question_quality'] = Decimal(str(sum(quality_scores) / len(quality_scores)))
                
            # Include duplicate question count
            if 'duplicate_count' in st.session_state.metrics:
                metrics_item['interaction_metrics']['duplicate_questions'] = Decimal(str(st.session_state.metrics['duplicate_count']))

            # Writing to DynamoDB
            response = self.metrics_table.put_item(Item=metrics_item)
            logger.info(f"Metrics logged to DynamoDB for session {self.session_id}")
            
            # Also logging to CloudWatch if available
            self.log_to_cloudwatch(metrics_item)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            return False

    # Updating metrics with new interaction data
    def update_metrics(self, question, response, retrieval_time, response_time, context_data=None, evaluation_data=None, user_rating=None):
        # Initializing metrics if not present
        if 'metrics' not in st.session_state:
            self._initialize_session_metrics()
        try:
            # Updating basic metrics
            st.session_state.metrics['retrieval_times'].append(retrieval_time)
            st.session_state.metrics['response_times'].append(response_time)
            # Synchronize with qa_pairs instead of incrementing
            if 'qa_pairs' in st.session_state:
                st.session_state.metrics['questions_asked'] = len(st.session_state.qa_pairs)
            else:
                # Only increment if we don't have qa_pairs as source of truth
                st.session_state.metrics['questions_asked'] += 1

            # Syncing with known_info if available
            if 'known_info' in st.session_state:
                st.session_state.metrics['questions_asked'] = len(st.session_state.known_info)
            
            # Calculating information density (response length vs information content)
            if response:
                response_words = response.lower().split()
                unique_words = set(response_words)
                info_density = len(unique_words) / len(response_words) if response_words else 0
            else:
                info_density = 0
            
            # Calculating response specificity (how specific vs generic the response is)
                # Specificity calculation: 
                # 1. Remove common English words that don't carry domain-specific meaning
                # 2. Calculate ratio of specific words to total unique words
                # Higher values indicate more specific, technical responses
            common_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'for', 'with'}
            if response:
                unique_words = set(response.lower().split())
                specific_words = len(unique_words - common_words)
                total_unique_words = len(unique_words)
                specificity = specific_words / total_unique_words if total_unique_words > 0 else 0
            else:
                specificity = 0

            # Storing calculated metrics
            st.session_state.metrics['info_density'].append(info_density)
            st.session_state.metrics['response_specificity'].append(specificity)
            st.session_state.metrics['response_lengths'].append(len(response.split()))
            
            # Calculating context utilization if context data available
            if context_data and 'relevant_chunks' in context_data and response:
                # Get context words
                context_text = ' '.join(context_data['relevant_chunks'])
                # Clean and normalize the context words
                context_words = set(re.sub(r'[^\w\s]', '', context_text.lower()).split())
                
                # Getting response words
                response_words = set(re.sub(r'[^\w\s]', '', response.lower()).split())

                # Removing common stopwords from both sets to focus on meaningful overlap
                stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or',
                            'but', 'for', 'with', 'in', 'to', 'of', 'that', 'this'}
                context_words = context_words - stopwords
                response_words = response_words - stopwords
                
                # Calculating overlap and preventing division by zero
                if context_words and len(context_words) > 0:  # Explicit check for empty sets
                    utilization = len(context_words.intersection(response_words)) / len(context_words)
                    st.session_state.metrics['context_utilization'].append(utilization)
                    print(f"Context utilization calculated: {utilization:.2f}")
                else:
                    # If no meaningful context words, set a default value
                    st.session_state.metrics['context_utilization'].append(0.0)
            
            # Add evaluation metrics if available
            if evaluation_data is None:
                print("No evaluation data received")
                evaluation_data = {"score": 0}

            # Extract and store the score
            print(f"Raw evaluation data: {evaluation_data}")
            try:
                if isinstance(evaluation_data, dict) and "score" in evaluation_data:
                    score_value = evaluation_data["score"]
                    print(f"Successfully extracted score value: {score_value}")
                else:
                    print(f"Invalid evaluation data format: {evaluation_data}")
                    score_value = 0
                    
            except Exception as e:
                print(f"Error accessing score: {e}")
                score_value = 0

            # Store the score in the metrics
            if 'evaluation_scores' not in st.session_state.metrics:
                st.session_state.metrics['evaluation_scores'] = []

            st.session_state.metrics['evaluation_scores'].append(score_value)
            print(f"Added score: {score_value}")
            print(f"Current scores: {st.session_state.metrics['evaluation_scores']}")

            # Store minimal data for average calculation
            if 'evaluations' not in st.session_state.metrics:
                st.session_state.metrics['evaluations'] = []

            st.session_state.metrics['evaluations'].append({
                'score': score_value
            })
            
            # Calculating question quality metrics
            if context_data:
                context_text = ' '.join(context_data.get('relevant_chunks', []))
                
                # Calculating question quality metrics
                question_metrics = log_rag_question_metrics(
                    question, 
                    context_text, 
                    st.session_state.metrics['questions_asked'],
                    st.session_state.get('previous_question_embeddings', []),
                    st.session_state.get('previous_questions', [])
                )
                
                # Storing individual metrics
                st.session_state.metrics['context_relevance'].append(question_metrics['context_relevance'])
                st.session_state.metrics['question_specificity'].append(question_metrics['specificity'])
                st.session_state.metrics['question_length'].append(question_metrics['question_length'])
                st.session_state.metrics['diversity'].append(question_metrics['diversity'])
                st.session_state.metrics['question_quality_scores'].append(question_metrics['final_score'])
                
                # Tracking duplicate questions
                if question_metrics['is_duplicate']:
                    st.session_state.metrics['duplicate_count'] += 1
                
                # Storing embedding and question for future comparison
                if question_metrics['embedding'] is not None:
                    if 'previous_question_embeddings' not in st.session_state:
                        st.session_state.previous_question_embeddings = []
                    st.session_state.previous_question_embeddings.append(question_metrics['embedding'])
                
                if 'previous_questions' not in st.session_state:
                    st.session_state.previous_questions = []
                st.session_state.previous_questions.append(question)

            # Adding user feedback metrics if available
            if user_rating is not None:
                if 'user_ratings' not in st.session_state.metrics:
                    st.session_state.metrics['user_ratings'] = []
                
                st.session_state.metrics['user_ratings'].append(user_rating)
                
                # Calculate percentage of positively rated questions
                positive_ratings = sum(1 for rating in st.session_state.metrics['user_ratings'] if rating > 0)
                total_ratings = len(st.session_state.metrics['user_ratings'])
                
                if total_ratings > 0:
                    st.session_state.metrics['positive_rating_percentage'] = positive_ratings / total_ratings
                else:
                    st.session_state.metrics['positive_rating_percentage'] = 0
            
            # Logging metrics to persistent storage
            self.log_metrics()
            
            logger.info(f"Metrics updated for question: '{question[:30]}...'")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            return False
           
    # Displaying metrics dashboard in Streamlit
    # Dashboard layout:
        # - Top row: Basic metrics (questions, retrieval time, response time)
        # - Middle row: Quality metrics (density, specificity, utilization)
        # - Bottom: Trend visualization showing response length over time
    def display_metrics_dashboard(self):
        try:
            if 'metrics' not in st.session_state:
                self._initialize_session_metrics()
            st.subheader("Knowledge Transfer Metrics")
            
            # Basic metrics row
            col1, col2, col3 = st.columns(3)
            
            # Questions asked
            with col1:
                st.metric(
                    "Questions Asked", 
                    st.session_state.metrics['questions_asked'], 
                    f"{10 - st.session_state.metrics['questions_asked']} remaining"
                )
            
            # Retrieval time
            with col2:
                retrieval_times = st.session_state.metrics.get('retrieval_times', [])
                if retrieval_times and len(retrieval_times) > 0:  # Check if both non-empty and length > 0
                    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
                    st.metric("Avg Retrieval Time (s)", f"{avg_retrieval:.2f}")
                else:
                    st.metric("Avg Retrieval Time (s)", "N/A")
            
            # Response time
            with col3:
                response_times = st.session_state.metrics['response_times']
                if response_times:
                    avg_response = sum(response_times) / len(response_times)
                    st.metric("Avg Response Time (s)", f"{avg_response:.2f}")
                else:
                    st.metric("Avg Response Time (s)", "N/A")
            
            
            # Response quality metrics row - synced with first row
            col1, col2, col3 = st.columns(3)
            # Information density
            with col1:
                if 'info_density' in st.session_state.metrics and st.session_state.metrics['info_density']:
                    density_values = st.session_state.metrics['info_density']
                    avg_density = sum(density_values) / len(density_values)
                    st.metric("Avg Information Density", f"{avg_density:.1%}")
                else:
                    st.metric("Avg Information Density", "N/A")
            
            # Response specificity
            with col2:
                if 'response_specificity' in st.session_state.metrics and st.session_state.metrics['response_specificity']:
                    specificity_values = st.session_state.metrics['response_specificity']
                    avg_specificity = sum(specificity_values) / len(specificity_values)
                    st.metric("Response Specificity", f"{avg_specificity:.1%}")
                else:
                    st.metric("Response Specificity", "N/A")
            
            # Context utilization
            with col3:
                utilization_values = st.session_state.metrics.get('context_utilization', [])
                if utilization_values:
                    avg_utilization = sum(utilization_values) / len(utilization_values)
                    st.metric("Context Utilization", f"{avg_utilization:.1%}")
                else:
                    st.metric("Context Utilization", "N/A")
            
            # Question quality metrics
            if ('context_relevance' in st.session_state.metrics and 
                st.session_state.metrics['context_relevance'] and 
                'diversity' in st.session_state.metrics):
                
                st.subheader("Question Quality Metrics")
                
                    
            #st.subheader("Question Quality Metrics")
            
            # First row of question quality metrics - 4 columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                relevance_values = st.session_state.metrics['context_relevance']
                avg_relevance = sum(relevance_values) / len(relevance_values)
                st.metric("Context Relevance", f"{avg_relevance*100:.1f}%")
            
            with col2:
                diversity_values = st.session_state.metrics['diversity']
                avg_diversity = sum(diversity_values) / len(diversity_values)
                st.metric("Question Diversity", f"{avg_diversity*100:.1f}%")
            
            with col3:
                if 'question_specificity' in st.session_state.metrics and st.session_state.metrics['question_specificity']:
                    q_specificity_values = st.session_state.metrics['question_specificity']
                    avg_q_specificity = sum(q_specificity_values) / len(q_specificity_values)
                    st.metric("Question Specificity", f"{avg_q_specificity*100:.1f}%")
                else:
                    st.metric("Question Specificity", "N/A")
            
            with col4:
                if 'duplicate_count' in st.session_state.metrics:
                    duplicate_count = st.session_state.metrics['duplicate_count']
                    questions_asked = st.session_state.metrics['questions_asked']
                    if questions_asked > 0:
                        duplicate_pct = (duplicate_count / questions_asked) * 100
                        st.metric("Duplicate Questions", f"{duplicate_count} ({duplicate_pct:.1f}%)")
                    else:
                        st.metric("Duplicate Questions", "0 (0.0%)")
                else:
                    st.metric("Duplicate Questions", "N/A")

                # Second row - overall quality centered and larger
            if 'question_quality_scores' in st.session_state.metrics and st.session_state.metrics['question_quality_scores']:
                quality_scores = st.session_state.metrics['question_quality_scores']
                avg_quality = sum(quality_scores) / len(quality_scores)
                
                st.markdown(f"""
                <div style="text-align:center; margin-top:10px;">
                    <h3>Overall Question Quality</h3>
                    <p style="font-size:26px; font-weight:bold; color:#1f77b4;">{avg_quality*10:.2f}/10</p>
                </div>
                """, unsafe_allow_html=True)
                
                
            # Response length trend visualization
            if 'response_lengths' in st.session_state.metrics and st.session_state.metrics['response_lengths']:
                # Create unique key for chart to prevent re-rendering issues
                chart_key = f"length_trend_{self.session_id}_{uuid.uuid4().hex[:6]}"
                
                # Creating figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.metrics['response_lengths'],
                    mode='lines+markers',
                    name='Response Length (words)'
                ))
                
                # Setting layout
                fig.update_layout(
                    title='Response Length Trend',
                    xaxis_title='Question Number',
                    yaxis_title='Word Count',
                    height=300
                )
                
                # Displaying chart
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
        except Exception as e:
            logger.error(f"Error displaying metrics dashboard: {str(e)}")
            st.error("Error displaying metrics dashboard")


    # New method for sidebar that only shows knowledge transfer metrics
    def display_sidebar_metrics(self):
        try:
            if 'metrics' not in st.session_state:
                self._initialize_session_metrics()
                
            st.sidebar.subheader("Session Metrics")
            
            # Displaying questions asked and response time side by side
            col1, col2 = st.sidebar.columns(2)
            
            # Questions asked in first column
            with col1:
                st.metric(
                    "Questions Asked", 
                    st.session_state.metrics['questions_asked'], 
                    f"{10 - st.session_state.metrics['questions_asked']} remaining"
                )
            
            # Response time in second column
            with col2:
                response_times = st.session_state.metrics.get('response_times', [])
                if response_times:
                    avg_response = sum(response_times) / len(response_times)
                    st.metric("Avg Response (s)", f"{avg_response:.1f}")
                else:
                    st.metric("Avg Response (s)", "N/A")
         
        except Exception as e:
            logger.error(f"Error displaying sidebar metrics: {str(e)}")
    
    # Returning current session metrics for external use
    def get_session_metrics(self):
        try:
            return {
                'session_id': self.session_id,
                'metrics': st.session_state.metrics.copy() if hasattr(st.session_state, 'metrics') else {}
            }
        except Exception as e:
            logger.error(f"Error getting session metrics: {str(e)}")
            return {'session_id': self.session_id, 'metrics': {}, 'error': str(e)}