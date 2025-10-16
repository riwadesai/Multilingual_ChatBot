"""
Configuration file for Multilingual RAG System
Based on the 72-hour assignment requirements
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDFS_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # Free, small model
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Vector database settings
VECTOR_DB_TYPE = "faiss"  # or "chroma"
VECTOR_DB_PATH = MODELS_DIR / "vector_db"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ".", "!", "?", " ", ""]

# Retrieval settings
TOP_K_RETRIEVAL = 5
TOP_K_RERANK = 3
SIMILARITY_THRESHOLD = 0.7

# Language settings
SUPPORTED_LANGUAGES = ["en", "hi", "bn", "zh", "es", "fr", "de"]
DEFAULT_LANGUAGE = "en"

# UI settings
STREAMLIT_CONFIG = {
    "page_title": "Multilingual RAG System",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Performance settings
MAX_DOCUMENTS = 10
MAX_CHUNKS_PER_DOC = 100
BATCH_SIZE = 32

# Evaluation settings
EVALUATION_DATASET_PATH = DATA_DIR / "evaluation_qa.json"
METRICS_TO_TRACK = [
    "retrieval_precision",
    "retrieval_recall", 
    "answer_relevance",
    "answer_correctness",
    "latency"
]

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create directories if they don't exist
for directory in [DATA_DIR, PDFS_DIR, PROCESSED_DIR, MODELS_DIR, VECTOR_DB_PATH]:
    directory.mkdir(parents=True, exist_ok=True)
