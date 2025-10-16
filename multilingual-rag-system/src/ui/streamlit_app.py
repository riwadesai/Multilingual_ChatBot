"""
Streamlit UI for Multilingual RAG System
Main application interface
"""

import streamlit as st
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Streamlit app"""
    
    # Page configuration
    st.set_page_config(
        page_title="Multilingual RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'vector_manager' not in st.session_state:
        st.session_state.vector_manager = None
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 Multilingual RAG System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Language selection
        language = st.selectbox(
            "Select Language",
            ["en", "hi", "bn", "zh", "es", "fr", "de"],
            format_func=lambda x: {
                "en": "English", "hi": "Hindi", "bn": "Bengali", 
                "zh": "Chinese", "es": "Spanish", "fr": "French", "de": "German"
            }[x]
        )
        
        # Document filter
        st.subheader("📚 Document Filter")
        if st.session_state.vector_manager:
            available_docs = st.session_state.vector_manager.list_documents()
            if available_docs:
                selected_docs = st.multiselect(
                    "Select documents to search in:",
                    available_docs,
                    default=available_docs
                )
            else:
                selected_docs = None
                st.info("No documents available. Please upload some PDFs first.")
        else:
            selected_docs = None
            st.info("Vector manager not initialized.")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            use_reranking = st.checkbox("Use Reranking", value=True)
            max_context_length = st.slider("Max Context Length", 1000, 4000, 2000)
            retrieval_k = st.slider("Retrieval Count", 5, 20, 10)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Upload", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        render_chat_interface(language, selected_docs, use_reranking, max_context_length)
    
    with tab2:
        render_upload_interface()
    
    with tab3:
        render_analytics_interface()
    
    with tab4:
        render_settings_interface()

def render_chat_interface(language: str, selected_docs: List[str], 
                         use_reranking: bool, max_context_length: int):
    """Render the chat interface"""
    
    st.header("💬 Ask Questions")
    
    # Initialize RAG pipeline if not already done
    if st.session_state.rag_pipeline is None:
        with st.spinner("Initializing RAG pipeline..."):
            try:
                initialize_rag_pipeline()
                st.success("RAG pipeline initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize RAG pipeline: {e}")
                return
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get answer from RAG pipeline
                    result = st.session_state.rag_pipeline.answer_question(
                        question=prompt,
                        language=language,
                        document_filter=selected_docs,
                        use_reranking=use_reranking,
                        max_context_length=max_context_length
                    )
                    
                    # Display answer
                    st.write(result['answer'])
                    
                    # Display confidence
                    confidence = result['confidence']
                    if confidence > 0.7:
                        confidence_class = "confidence-high"
                        confidence_text = "High"
                    elif confidence > 0.4:
                        confidence_class = "confidence-medium"
                        confidence_text = "Medium"
                    else:
                        confidence_class = "confidence-low"
                        confidence_text = "Low"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>Confidence:</strong> <span class="{confidence_class}">{confidence_text} ({confidence:.2f})</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources
                    if result['sources']:
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result['sources'][:3]):  # Show top 3 sources
                            with st.expander(f"Source {i+1} (Relevance: {source.get('relevance_score', 'N/A')})"):
                                st.write(source['content'])
                                if 'metadata' in source:
                                    st.json(source['metadata'])
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result['answer'],
                        "sources": result['sources'],
                        "confidence": result['confidence']
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    logger.error(f"Error in chat interface: {e}")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"][:3]):
                        st.write(f"**Source {i+1}:**")
                        st.write(source['content'])
                        st.write("---")

def render_upload_interface():
    """Render the document upload interface"""
    
    st.header("📄 Upload Documents")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF documents to add to the knowledge base"
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s)")
        
        # Process files
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    process_uploaded_files(uploaded_files)
                    st.success("Documents processed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                    logger.error(f"Error processing files: {e}")
    
    # Document management
    if st.session_state.vector_manager:
        st.subheader("📚 Document Management")
        
        # List documents
        documents = st.session_state.vector_manager.list_documents()
        if documents:
            st.write(f"**Available Documents ({len(documents)}):**")
            
            for doc_name in documents:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"📄 {doc_name}")
                
                with col2:
                    if st.button("View", key=f"view_{doc_name}"):
                        view_document(doc_name)
                
                with col3:
                    if st.button("Delete", key=f"delete_{doc_name}"):
                        if st.session_state.vector_manager.remove_document(doc_name):
                            st.success(f"Deleted {doc_name}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {doc_name}")
        else:
            st.info("No documents available. Upload some PDFs to get started.")

def render_analytics_interface():
    """Render the analytics interface"""
    
    st.header("📊 Analytics")
    
    if st.session_state.vector_manager:
        # Get store statistics
        stats = st.session_state.vector_manager.get_store_stats()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", stats.get('total_documents', 0))
        
        with col2:
            st.metric("Total Chunks", stats.get('total_chunks', 0))
        
        with col3:
            st.metric("Vector Store Size", f"{stats.get('index_info', {}).get('ntotal', 0)}")
        
        with col4:
            st.metric("Embedding Dimension", f"{stats.get('index_info', {}).get('d', 0)}")
        
        # Language distribution
        if 'language_distribution' in stats:
            st.subheader("🌍 Language Distribution")
            lang_dist = stats['language_distribution']
            
            if lang_dist:
                lang_df = pd.DataFrame(list(lang_dist.items()), columns=['Language', 'Count'])
                st.bar_chart(lang_df.set_index('Language'))
            else:
                st.info("No language distribution data available")
        
        # Document list
        st.subheader("📚 Document Details")
        documents = st.session_state.vector_manager.list_documents()
        
        if documents:
            doc_data = []
            for doc_name in documents:
                doc_info = st.session_state.vector_manager.get_document_info(doc_name)
                if doc_info:
                    doc_data.append({
                        'Document': doc_name,
                        'Chunks': doc_info.get('chunk_count', 0),
                        'Added': doc_info.get('added_at', 'Unknown')
                    })
            
            if doc_data:
                df = pd.DataFrame(doc_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No documents available")
    
    else:
        st.warning("Vector manager not initialized. Please upload some documents first.")

def render_settings_interface():
    """Render the settings interface"""
    
    st.header("⚙️ Settings")
    
    # Model settings
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Embedding Model:**")
        st.code("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    with col2:
        st.write("**LLM Model:**")
        st.code("microsoft/Phi-3-mini-4k-instruct")
    
    # Pipeline settings
    st.subheader("🔧 Pipeline Settings")
    
    if st.session_state.rag_pipeline:
        # Get current settings
        current_stats = st.session_state.rag_pipeline.get_pipeline_stats()
        
        st.json(current_stats)
        
        # Update settings
        st.subheader("Update Settings")
        
        new_retrieval_k = st.number_input("Retrieval Count", min_value=1, max_value=50, value=10)
        new_rerank_k = st.number_input("Rerank Count", min_value=1, max_value=20, value=5)
        
        if st.button("Update Pipeline Settings"):
            st.session_state.rag_pipeline.update_parameters(
                retrieval_k=new_retrieval_k,
                rerank_k=new_rerank_k
            )
            st.success("Settings updated!")
    
    else:
        st.info("RAG pipeline not initialized")

def initialize_rag_pipeline():
    """Initialize the RAG pipeline"""
    # This would be implemented to set up the actual pipeline
    # For now, we'll create a placeholder
    st.session_state.rag_pipeline = "placeholder"
    st.session_state.vector_manager = "placeholder"

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files"""
    # This would be implemented to actually process the files
    # For now, we'll just simulate processing
    time.sleep(2)  # Simulate processing time
    logger.info(f"Processed {len(uploaded_files)} files")

def view_document(doc_name: str):
    """View document details"""
    st.write(f"Viewing document: {doc_name}")
    # This would be implemented to show document details

if __name__ == "__main__":
    create_app()
