"""
Embedding Generator Module
Generates embeddings using sentence-transformers for multilingual support
"""

import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import logging
import torch

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for multilingual text using sentence-transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return [np.zeros(self.model.get_sentence_embedding_dimension()) for _ in texts]
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings)
            
            # Handle empty texts in original list
            result_embeddings = []
            valid_idx = 0
            
            for text in texts:
                if text and text.strip():
                    result_embeddings.append(embeddings[valid_idx])
                    valid_idx += 1
                else:
                    result_embeddings.append(np.zeros(self.model.get_sentence_embedding_dimension()))
            
            return result_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return zero vectors as fallback
            return [np.zeros(self.model.get_sentence_embedding_dimension()) for _ in texts]
    
    def generate_document_embeddings(self, documents: List[Document], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for LangChain documents
        
        Args:
            documents: List of Document objects
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not documents:
            return []
        
        # Extract text content from documents
        texts = [doc.page_content for doc in documents]
        return self.generate_embeddings(texts, batch_size)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray], 
                         top_k: int = 5) -> List[tuple]:
        """
        Find most similar embeddings to query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if not candidate_embeddings:
            return []
        
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings generated by this model"""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dimension': self.get_embedding_dimension(),
            'max_sequence_length': getattr(self.model, 'max_seq_length', 512)
        }
    
    def encode_queries(self, queries: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Encode queries for retrieval (optimized for queries)
        
        Args:
            queries: List of query strings
            batch_size: Batch size for processing
            
        Returns:
            List of query embeddings
        """
        return self.generate_embeddings(queries, batch_size)
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Encode documents for indexing (optimized for documents)
        
        Args:
            documents: List of document strings
            batch_size: Batch size for processing
            
        Returns:
            List of document embeddings
        """
        return self.generate_embeddings(documents, batch_size)
    
    def batch_encode(self, texts: List[str], batch_size: int = 32, 
                    show_progress: bool = False) -> List[np.ndarray]:
        """
        Batch encode texts with progress tracking
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
        except Exception as e:
            logger.error(f"Failed to batch encode: {e}")
            return [np.zeros(self.get_embedding_dimension()) for _ in texts]
