"""
Document Processing Module for Multilingual RAG System
Handles PDF extraction, OCR, and text preprocessing
"""

from .pdf_processor import PDFProcessor
from .text_processor import TextProcessor
from .language_detector import LanguageDetector

__all__ = ["PDFProcessor", "TextProcessor", "LanguageDetector"]
