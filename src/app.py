import streamlit as st
import os
from model import (create_conversational_rag_chain, get_session_history, load_documents)
import pandas as pd
from langchain_community.chat_message_histories import ChatMessageHistory
import shutil

# Configure Streamlit for larger file uploads
st.set_page_config(
    page_title="Document QA Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FAISS_INDEX_DIR = os.path.join(DATA_DIR, 'faiss_index')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(FAISS_INDEX_DIR):
    os.makedirs(FAISS_INDEX_DIR)

# Load external CSS
def load_css(css_file):
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the CSS file path
        css_path = os.path.join(current_dir, 'static', 'css', 'style.css')
        
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS file: {str(e)}")
        # Fallback minimal CSS for critical functionality
        st.markdown("""
        <style>
        .stApp { background-color: #343541; color: #ECECF1; }
        .stSidebar { background-color: #202123; }
        </style>
        """, unsafe_allow_html=True)

# Function to save FAISS index
def save_faiss_index(vectorstore, session_id):
    index_path = os.path.join(FAISS_INDEX_DIR, f"index_{session_id}")
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    vectorstore.save_local(index_path)
    return index_path

# Function to load FAISS index
def load_faiss_index(session_id, embeddings):
    from langchain_community.vectorstores import FAISS
    index_path = os.path.join(FAISS_INDEX_DIR, f"index_{session_id}")
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings)
    return None

# Load the CSS file
load_css('style.css')

# Increase memory limits for file processing
if not os.environ.get("STREAMLIT_SERVER_MAX_UPLOAD_SIZE"):
    os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2048"  # Size in MB (2GB)

# Increase timeout for longer processing
if not os.environ.get("STREAMLIT_SERVER_TIMEOUT"):
    os.environ["STREAMLIT_SERVER_TIMEOUT"] = "300"  # Timeout in seconds

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "documents" not in st.session_state:
    st.session_state.documents = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "process_input" not in st.session_state:
    st.session_state.process_input = False
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"

# Initialize session history for current session if not exists
if st.session_state.session_id not in st.session_state.store:
    st.session_state.store[st.session_state.session_id] = {
        "messages": [],
        "history": get_session_history(st.session_state.session_id)
    }

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>📚 Document QA Bot</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # File upload section
    st.markdown("### 📎 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "xlsx", "xls", "ppt", "pptx", "png", "jpg", "jpeg", "zip", "scorm", "mp4", "webm", "avi", "mov"],
        accept_multiple_files=True,
        help="Upload documents in PDF, Excel, PowerPoint, SCORM, video, or image formats"
    )
    
    # Process documents immediately after upload
    if uploaded_files:
        current_files = {f.name for f in uploaded_files}
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:  # Only process if there are new files
            with st.spinner(f"Processing {len(new_files)} new document(s)..."):
                try:
                    # Process only new documents
                    new_documents = load_documents(new_files)
                    
                    if st.session_state.documents:
                        # Merge new documents with existing ones
                        documents = st.session_state.documents
                        
                        # Merge vectorstore
                        if new_documents.get("vectorstore"):
                            documents["vectorstore"].merge_from(new_documents["vectorstore"])
                        
                        # Merge tables
                        if new_documents.get("tables"):
                            documents["tables"].extend(new_documents["tables"])
                        
                        # Merge images
                        if new_documents.get("images"):
                            documents["images"].extend(new_documents["images"])
                        
                        # Merge processing errors
                        if new_documents.get("processing_errors"):
                            documents["processing_errors"].extend(new_documents["processing_errors"])
                        
                        # Update retriever with merged vectorstore
                        documents["retriever"] = documents["vectorstore"].as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": 6,
                                "fetch_k": 30,
                                "lambda_mult": 0.7
                            }
                        )
                        
                        st.session_state.documents = documents
                    else:
                        # If no existing documents, use new documents directly
                        st.session_state.documents = new_documents
                    
                    # Update processed files set
                    st.session_state.processed_files.update(f.name for f in new_files)
                    st.success(f"Successfully processed {len(new_files)} new document(s)!")
                    
                    # Display processing information if documents exist
                    if st.session_state.documents:
                        st.markdown("### 📋 Processing Information")
                        documents = st.session_state.documents
                        
                        # Display processing errors if any
                        if documents.get("processing_errors"):
                            with st.expander("⚠️ Processing Issues", expanded=False):
                                for error in documents["processing_errors"]:
                                    st.warning(error)

                        # Display tables
                        if documents.get("tables"):
                            with st.expander("📊 Extracted Tables", expanded=False):
                                for table in documents["tables"]:
                                    if table.get("raw_data"):
                                        st.markdown(f"**Table from {table['source']} (Page/Sheet {table.get('page', 'N/A')})**")
                                        df = pd.DataFrame(table["raw_data"])
                                        st.dataframe(df, use_container_width=True)
                                        st.markdown("---")

                        # Display images
                        if documents.get("images"):
                            with st.expander("🖼️ Extracted Images", expanded=False):
                                for img in documents["images"]:
                                    st.markdown(f"**Image from {img['source']}**")
                                    st.image(img["image"])
                                    if img.get("text"):
                                        st.markdown("**Extracted Text:**")
                                        st.text(img["text"])
                                    st.markdown("---")
                            
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    print(f"Detailed error: {str(e)}")  # For debugging
        
        # Handle removed files
        removed_files = st.session_state.processed_files - current_files
        if removed_files:
            st.session_state.processed_files -= removed_files
            if not current_files:  # All files were removed
                st.session_state.documents = None
                st.session_state.processed_files = set()
                # Clear FAISS index
                index_path = os.path.join(FAISS_INDEX_DIR, f"index_{st.session_state.session_id}")
                if os.path.exists(index_path):
                    shutil.rmtree(index_path)
                st.rerun()

    # Session management
    st.markdown("### 🔑 Session Management")
    new_session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        help="Enter a unique session ID to maintain conversation history"
    ).strip() or "default_session"
    
    # Update session if changed
    if new_session_id != st.session_state.session_id:
        st.session_state.session_id = new_session_id
        if new_session_id not in st.session_state.store:
            st.session_state.store[new_session_id] = {}
            st.session_state.messages = []
        else:
            st.session_state.messages = st.session_state.store[new_session_id].get("messages", [])
    
    # Clear chat history button
    if st.button("Clear Chat History", type="secondary"):
        # Clear messages
        st.session_state.messages = []
        # Clear document context
        st.session_state.documents = None
        st.session_state.uploaded_files = []
        # Clear session store
        st.session_state.store[st.session_state.session_id] = {
            "messages": [],
            "history": ChatMessageHistory()
        }
        # Remove FAISS index for this session
        index_path = os.path.join(FAISS_INDEX_DIR, f"index_{st.session_state.session_id}")
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        st.rerun()

# Track if files were removed
if "previous_files" not in st.session_state:
    st.session_state.previous_files = set()

current_files = {f.name for f in uploaded_files} if uploaded_files else set()
if current_files != st.session_state.previous_files:
    # Files have changed
    if len(current_files) < len(st.session_state.previous_files):
        # Files were removed
        st.session_state.documents = None
        st.session_state.uploaded_files = []
        # Clear FAISS index
        index_path = os.path.join(FAISS_INDEX_DIR, f"index_{st.session_state.session_id}")
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
    st.session_state.previous_files = current_files

# Main content area
# Chat interface - Moved outside the document condition
chat_container = st.container()
input_container = st.container()

# Display messages in the chat container
with chat_container:
    st.markdown("<h3 style='text-align: center;'>💬 Chat with your Documents</h3>", unsafe_allow_html=True)
    
    if not uploaded_files:
        # Welcome message when no documents are uploaded
        st.markdown("""
        <div class='welcome-container'>
            <h1 style='text-align: center; color: #E0E0E0;'>👋 Welcome to Document QA Bot!</h1>
            <p style='font-size: 1.2rem; color: #BBBBBB; text-align: center;'>Your intelligent assistant for document analysis and interaction.</p>
        </div>
        
        <div class='features-container'>
            <h3 style='color: #E0E0E0;'>🌟 Features</h3>
            <ul style='color: #E0E0E0;'>
                <li>📄 Support for multiple document formats (PDF, Excel, PowerPoint, etc.)</li>
                <li>🎥 Video content analysis with audio transcription</li>
                <li>📊 Table extraction and analysis</li>
                <li>🖼️ Image processing with text extraction</li>
                <li>💬 Natural conversation interface</li>
                <li>🔄 Session management for continuous conversations</li>
            </ul>
        </div>
        
        <div class='features-container' style='margin-top: 2rem;'>
            <h3 style='color: #E0E0E0;'>🚀 Getting Started</h3>
            <ol style='color: #E0E0E0;'>
                <li>Upload your documents using the sidebar</li>
                <li>Wait for the processing to complete</li>
                <li>Start asking questions about your documents</li>
            </ol>
            <p style='color: #BBBBBB; font-style: italic;'>Tip: You can use the Session ID to maintain conversation history across sessions!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="message-container user-message">
                        <div class="message-content">
                            <div class="avatar user-avatar">U</div>
                            <div class="message-text">{message["content"]}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="message-container assistant-message">
                        <div class="message-content">
                            <div class="avatar assistant-avatar">A</div>
                            <div class="message-text">{message["content"]}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Add some space at the bottom for the input
    st.markdown("<div style='height: 100px'></div>", unsafe_allow_html=True)

# Input at the bottom - Always visible
with input_container:
    # Create a column for centering the input
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        def submit():
            st.session_state.process_input = True
            st.session_state.user_input = st.session_state.widget
            st.session_state.widget = ""

        st.text_input(
            "Chat Input",
            placeholder="Upload documents and ask questions..." if not uploaded_files else "Ask me anything about your documents...",
            key="widget",
            on_change=submit,
            label_visibility="collapsed",
            disabled=not uploaded_files  # Disable input if no documents are uploaded
        )

if st.session_state.process_input and uploaded_files:
    user_input = st.session_state.user_input
    st.session_state.process_input = False

    if user_input:
        try:
            # Process documents if not already processed
            if not st.session_state.documents:
                with st.spinner("Processing documents..."):
                    documents = load_documents(uploaded_files)
                    st.session_state.documents = documents
            else:
                documents = st.session_state.documents

            vectorstore, retriever, extracted_data = (
                documents["vectorstore"],
                documents["retriever"],
                documents["extracted_data"]
            )

            # Initialize RAG chain
            conversational_rag_chain = create_conversational_rag_chain(retriever)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Update store
            if st.session_state.session_id not in st.session_state.store:
                st.session_state.store[st.session_state.session_id] = {
                    "messages": [],
                    "history": get_session_history(st.session_state.session_id)
                }
            st.session_state.store[st.session_state.session_id]["messages"] = st.session_state.messages.copy()

            if not documents["vectorstore"].docstore._dict:
                response_text = "No document content was found to analyze. Please make sure your document contains readable text."
            else:
                with st.spinner("Thinking..."):
                    try:
                        # Get session history
                        session_history = get_session_history(st.session_state.session_id)
                        
                        # Add the current message to session history
                        session_history.add_user_message(user_input)
                        
                        # Get response from the chain
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            {
                                "default": {
                                    "id": st.session_state.session_id,
                                    "history": None,
                                    "annotation": "chat_history"
                                }
                            }
                        )
                        
                        if not response:
                            raise ValueError("No response received from the chain")
                            
                        response_text = response.get('answer', 'I apologize, but I was unable to process your request.')
                        
                        # Add assistant response to session history
                        session_history.add_ai_message(response_text)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        st.session_state.store[st.session_state.session_id]["messages"] = st.session_state.messages.copy()
                        
                    except Exception as e:
                        error_msg = f"Error processing your request: {str(e)}"
                        st.error(error_msg)
                        print(f"Detailed error: {str(e)}")  # For debugging
                        response_text = error_msg

            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            print(f"Detailed error: {str(e)}")  # For debugging
