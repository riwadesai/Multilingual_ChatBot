"""
Reranker Module
Implements reranking using cross-encoder models for better retrieval quality
"""

import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class Reranker:
    """Reranks retrieved documents using cross-encoder models"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker
        
        Args:
            model_name: Name of the cross-encoder model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model {self.model_name}: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], 
               top_k: int = 5, batch_size: int = 32) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Query string
            documents: List of document texts
            top_k: Number of top documents to return
            batch_size: Batch size for processing
            
        Returns:
            List of (document_index, relevance_score) tuples
        """
        if not documents:
            return []
        
        try:
            # Create query-document pairs
            pairs = [(query, doc) for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs, batch_size=batch_size)
            
            # Create index-score pairs
            scored_docs = [(i, float(score)) for i, score in enumerate(scores)]
            
            # Sort by relevance score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            # Return original order as fallback
            return [(i, 0.0) for i in range(min(len(documents), top_k))]
    
    def rerank_with_metadata(self, query: str, documents: List[dict], 
                            top_k: int = 5, batch_size: int = 32) -> List[dict]:
        """
        Rerank documents with metadata
        
        Args:
            query: Query string
            documents: List of document dictionaries with 'text' and metadata
            top_k: Number of top documents to return
            batch_size: Batch size for processing
            
        Returns:
            List of reranked document dictionaries with relevance scores
        """
        if not documents:
            return []
        
        try:
            # Extract texts
            texts = [doc.get('text', '') for doc in documents]
            
            # Rerank
            ranked_indices = self.rerank(query, texts, top_k, batch_size)
            
            # Create result documents with scores
            result_docs = []
            for idx, score in ranked_indices:
                doc = documents[idx].copy()
                doc['relevance_score'] = score
                doc['rank'] = len(result_docs) + 1
                result_docs.append(doc)
            
            return result_docs
            
        except Exception as e:
            logger.error(f"Failed to rerank documents with metadata: {e}")
            # Return original documents as fallback
            return documents[:top_k]
    
    def rerank_langchain_docs(self, query: str, documents: List[dict], 
                             top_k: int = 5, batch_size: int = 32) -> List[dict]:
        """
        Rerank LangChain documents
        
        Args:
            query: Query string
            documents: List of LangChain document dictionaries
            top_k: Number of top documents to return
            batch_size: Batch size for processing
            
        Returns:
            List of reranked document dictionaries
        """
        if not documents:
            return []
        
        try:
            # Extract page_content from documents
            texts = [doc.get('page_content', '') for doc in documents]
            
            # Rerank
            ranked_indices = self.rerank(query, texts, top_k, batch_size)
            
            # Create result documents
            result_docs = []
            for idx, score in ranked_indices:
                doc = documents[idx].copy()
                doc['relevance_score'] = score
                doc['rank'] = len(result_docs) + 1
                result_docs.append(doc)
            
            return result_docs
            
        except Exception as e:
            logger.error(f"Failed to rerank LangChain documents: {e}")
            return documents[:top_k]
    
    def compute_relevance_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute relevance scores for documents without ranking
        
        Args:
            query: Query string
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        if not documents:
            return []
        
        try:
            # Create query-document pairs
            pairs = [(query, doc) for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            return [float(score) for score in scores]
            
        except Exception as e:
            logger.error(f"Failed to compute relevance scores: {e}")
            return [0.0] * len(documents)
    
    def filter_by_threshold(self, documents: List[dict], 
                           threshold: float = 0.5) -> List[dict]:
        """
        Filter documents by relevance threshold
        
        Args:
            documents: List of documents with relevance scores
            threshold: Minimum relevance score
            
        Returns:
            Filtered list of documents
        """
        return [doc for doc in documents if doc.get('relevance_score', 0.0) >= threshold]
    
    def get_model_info(self) -> dict:
        """Get information about the reranker model"""
        return {
            'model_name': self.model_name,
            'model_type': 'cross-encoder'
        }
