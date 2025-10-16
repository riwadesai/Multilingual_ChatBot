"""
Text Processing Module
Handles text cleaning, chunking, and preprocessing for multilingual content
"""

import re
import nltk
import spacy
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Processes and chunks text for multilingual RAG system"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize text processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
    
    def clean_text(self, text: str, language: str = 'en') -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """
        Split text into chunks
        
        Args:
            text: Input text
            metadata: Additional metadata for chunks
            
        Returns:
            List of Document objects
        """
        if not text.strip():
            return []
        
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Create base metadata
        base_metadata = metadata or {}
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only include non-empty chunks
                doc_metadata = {
                    **base_metadata,
                    'chunk_index': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
        
        logger.info(f"Created {len(documents)} chunks from text")
        return documents
    
    def extract_sentences(self, text: str, language: str = 'en') -> List[str]:
        """
        Extract sentences from text
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of sentences
        """
        try:
            # Use NLTK for sentence tokenization
            sentences = nltk.sent_tokenize(text, language=language)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"Failed to tokenize sentences: {e}")
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def extract_keywords(self, text: str, language: str = 'en', max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            language: Language code
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        try:
            # Simple keyword extraction using word frequency
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Remove common stopwords (basic English list)
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }
            
            # Filter out stopwords and short words
            filtered_words = [
                word for word in words 
                if word not in stopwords and len(word) > 2
            ]
            
            # Count word frequency
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in keywords[:max_keywords]]
            
        except Exception as e:
            logger.warning(f"Failed to extract keywords: {e}")
            return []
    
    def preprocess_for_embedding(self, text: str) -> str:
        """
        Preprocess text for embedding generation
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove very short sentences (likely noise)
        sentences = self.extract_sentences(text)
        filtered_sentences = [s for s in sentences if len(s) > 10]
        
        return ' '.join(filtered_sentences)
    
    def create_summary(self, text: str, max_length: int = 200) -> str:
        """
        Create a simple extractive summary
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        sentences = self.extract_sentences(text)
        
        if not sentences:
            return ""
        
        # Simple extractive summary: take first few sentences
        summary_sentences = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) <= max_length:
                summary_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        return ' '.join(summary_sentences)
    
    def detect_language_patterns(self, text: str) -> Dict[str, float]:
        """
        Detect language patterns in text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with language confidence scores
        """
        # Simple language detection based on character patterns
        patterns = {
            'en': r'[a-zA-Z]',
            'hi': r'[\u0900-\u097F]',  # Devanagari
            'bn': r'[\u0980-\u09FF]',  # Bengali
            'zh': r'[\u4e00-\u9fff]',  # Chinese
            'ar': r'[\u0600-\u06FF]',  # Arabic
        }
        
        scores = {}
        total_chars = len(text)
        
        if total_chars == 0:
            return scores
        
        for lang, pattern in patterns.items():
            matches = len(re.findall(pattern, text))
            scores[lang] = matches / total_chars
        
        return scores
