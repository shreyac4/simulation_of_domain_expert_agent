import streamlit as st
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

"""
SessionManager: Manages application state and user interaction logging

This module handles Streamlit session state management and provides a central
place for logging user interactions and errors. 
It also ensures consistent state persistence across Streamlit reruns.
"""

# SessionManager class handles:
    # 1. Session state initialization and persistence
    # 2. Error handling and logging
    # 3. User interaction logging with appropriate UI feedback
class SessionManager:
    def __init__(self):
        self.initialize_session_state()
        
    # Initializing all required session state variables
    # This function creates default values for all session state variables if they don't exist.
    # This ensures the application maintains its state across Streamlit reruns and prevents KeyError exceptions when accessing state variables.
    def initialize_session_state(self):
        default_states = {
            'uploaded_file': None,
            'knowledge_system': None,
            'agent_logs': [],
            'inquiry_started': False,
            'last_question': None,
            'known_info': [],
            'error_log': [],
            'metrics_tracker': None,
            'qa_pairs': [],
            'processing_complete': False, 
            'expert_model': None
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    # Reset of all session state variables safely
    def reset_session(self):
        try:
            keys_to_remove = list(st.session_state.keys())
            for key in keys_to_remove:
                del st.session_state[key]
            self.initialize_session_state()
            return True
        except Exception as e:
            logger.error(f"❌ Error resetting session: {str(e)}")
            return False
    
    # Handling and logging errors with contextual information
    # Displays error in the UI and stores it in the error log for debugging
    def handle_error(self, error: Exception, context: str):    
        error_message = f"⚠️ Error in {context}: {str(error)}"
        if 'error_log' in st.session_state:
            st.session_state.error_log.append(error_message)
        logger.error(error_message)
        st.error(error_message)
    
    # Logging user interaction with appropriate UI feedback
    def log_interaction(self, message: str, message_type: str = "info"):
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            if 'agent_logs' in st.session_state:
                st.session_state.agent_logs.append(log_entry)

            message_types = {
                "error": st.error,
                "success": st.success,
                "warning": st.warning,
                "info": st.info
            }
            message_types.get(message_type, st.info)(log_entry)

        except Exception as e:
            logger.error(f"❌ Error logging interaction: {str(e)}")
    
    # Retrieves essential session state variables
    def get_session_state(self):
        try:
            return {key: st.session_state.get(key) for key in ['inquiry_started', 'processing_complete', 'agent_logs', 'error_log']}
        except Exception as e:
            logger.error(f"❌ Error getting session state: {str(e)}")
            return {}
