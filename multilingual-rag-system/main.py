"""
Main Application for Multilingual RAG System
Entry point for the application
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ui.streamlit_app import create_app

if __name__ == "__main__":
    create_app()
