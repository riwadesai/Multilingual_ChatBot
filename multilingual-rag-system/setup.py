"""
Setup script for Multilingual RAG System
Initializes the system and downloads required models
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages"""
    logger.info("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False
    return True

def download_models():
    """Download required models"""
    logger.info("Downloading models...")
    
    try:
        # Import required modules
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Download embedding model
        logger.info("Downloading embedding model...")
        embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Download LLM model
        logger.info("Downloading LLM model...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        
        # Download reranker model
        logger.info("Downloading reranker model...")
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        logger.info("All models downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")
    
    directories = [
        "data/pdfs",
        "data/processed", 
        "models",
        "models/vector_db",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def setup_environment():
    """Set up environment variables"""
    logger.info("Setting up environment...")
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_CACHE"] = str(Path("models").absolute())
    os.environ["HF_HOME"] = str(Path("models").absolute())
    
    logger.info("Environment variables set")

def main():
    """Main setup function"""
    logger.info("Starting Multilingual RAG System setup...")
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Install requirements
    if not install_requirements():
        logger.error("Setup failed at requirements installation")
        return False
    
    # Download models
    if not download_models():
        logger.error("Setup failed at model download")
        return False
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run the application with: streamlit run main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
