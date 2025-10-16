"""
FAISS Vector Store Module
Implements vector database operations using FAISS with LangChain integration
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import logging

logger = logging.getLogger(__name__)

class FAISSStore:
    """FAISS-based vector store for multilingual RAG system"""
    
    def __init__(self, embedding_model: Embeddings, index_path: Optional[str] = None):
        """
        Initialize FAISS store
        
        Args:
            embedding_model: Embedding model for generating vectors
            index_path: Path to save/load FAISS index
        """
        self.embedding_model = embedding_model
        self.index_path = Path(index_path) if index_path else None
        self.vectorstore = None
        self.metadata_store = {}
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize or load FAISS store"""
        if self.index_path and self.index_path.exists():
            try:
                self._load_store()
                logger.info(f"Loaded existing FAISS store from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load existing store: {e}. Creating new store.")
                self._create_new_store()
        else:
            self._create_new_store()
    
    def _create_new_store(self):
        """Create new FAISS store"""
        # Create empty vectorstore
        self.vectorstore = FAISS.from_texts(
            texts=[""], 
            embedding=self.embedding_model,
            metadatas=[{}]
        )
        # Remove the dummy document
        self.vectorstore.delete([0])
        logger.info("Created new FAISS store")
    
    def _load_store(self):
        """Load existing FAISS store"""
        self.vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Load metadata if available
        metadata_path = self.index_path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata_store = pickle.load(f)
    
    def _save_store(self):
        """Save FAISS store to disk"""
        if self.index_path:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(self.index_path))
            
            # Save metadata
            metadata_path = self.index_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            logger.info(f"Saved FAISS store to {self.index_path}")
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects
            batch_size: Batch size for processing
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        try:
            # Add documents to vectorstore
            doc_ids = self.vectorstore.add_documents(documents)
            
            # Store metadata
            for i, doc in enumerate(documents):
                doc_id = doc_ids[i] if i < len(doc_ids) else f"doc_{len(self.metadata_store)}"
                self.metadata_store[doc_id] = {
                    'metadata': doc.metadata,
                    'content': doc.page_content,
                    'added_at': str(pd.Timestamp.now())
                }
            
            # Save store
            self._save_store()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return []
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """
        Add texts to the vector store
        
        Args:
            texts: List of text strings
            metadatas: List of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        # Create documents
        metadatas = metadatas or [{}] * len(texts)
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        return self.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_dict: Dict = None) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Query string
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        try:
            if filter_dict:
                results = self.vectorstore.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5,
                                   filter_dict: Dict = None) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores
        
        Args:
            query: Query string
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search with score: {e}")
            return []
    
    def max_marginal_relevance_search(self, query: str, k: int = 5, 
                                    fetch_k: int = 20, 
                                    lambda_mult: float = 0.5) -> List[Document]:
        """
        Perform MMR (Maximal Marginal Relevance) search
        
        Args:
            query: Query string
            k: Number of results to return
            fetch_k: Number of documents to fetch before reranking
            lambda_mult: Lambda parameter for MMR
            
        Returns:
            List of diverse documents
        """
        try:
            results = self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform MMR search: {e}")
            return []
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents from the store
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from vectorstore
            self.vectorstore.delete(doc_ids)
            
            # Remove from metadata store
            for doc_id in doc_ids:
                self.metadata_store.pop(doc_id, None)
            
            # Save store
            self._save_store()
            
            logger.info(f"Deleted {len(doc_ids)} documents from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Get document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        if doc_id in self.metadata_store:
            metadata = self.metadata_store[doc_id]
            return Document(
                page_content=metadata['content'],
                metadata=metadata['metadata']
            )
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with store statistics
        """
        try:
            total_docs = len(self.metadata_store)
            
            # Get index info
            index_info = {}
            if hasattr(self.vectorstore, 'index'):
                index_info = {
                    'ntotal': self.vectorstore.index.ntotal,
                    'd': self.vectorstore.index.d,
                    'is_trained': self.vectorstore.index.is_trained
                }
            
            return {
                'total_documents': total_docs,
                'index_info': index_info,
                'store_path': str(self.index_path) if self.index_path else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def clear_store(self):
        """Clear all documents from the store"""
        try:
            # Create new empty store
            self._create_new_store()
            self.metadata_store = {}
            
            # Save empty store
            self._save_store()
            
            logger.info("Cleared vector store")
            
        except Exception as e:
            logger.error(f"Failed to clear store: {e}")
    
    def search_by_metadata(self, filter_dict: Dict, k: int = 5) -> List[Document]:
        """
        Search documents by metadata filter
        
        Args:
            filter_dict: Metadata filter dictionary
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            # Get all documents matching filter
            matching_docs = []
            for doc_id, metadata in self.metadata_store.items():
                doc_metadata = metadata['metadata']
                if self._matches_filter(doc_metadata, filter_dict):
                    doc = Document(
                        page_content=metadata['content'],
                        metadata=doc_metadata
                    )
                    matching_docs.append(doc)
            
            return matching_docs[:k]
            
        except Exception as e:
            logger.error(f"Failed to search by metadata: {e}")
            return []
    
    def _matches_filter(self, doc_metadata: Dict, filter_dict: Dict) -> bool:
        """Check if document metadata matches filter"""
        for key, value in filter_dict.items():
            if key not in doc_metadata:
                return False
            if doc_metadata[key] != value:
                return False
        return True
    
    def export_documents(self, output_path: str) -> bool:
        """
        Export all documents to file
        
        Args:
            output_path: Path to save exported documents
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export metadata
            with open(output_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            logger.info(f"Exported documents to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export documents: {e}")
            return False
