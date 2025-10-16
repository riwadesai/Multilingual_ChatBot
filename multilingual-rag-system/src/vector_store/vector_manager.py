"""
Vector Manager Module
Manages vector store operations and provides high-level interface
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import logging

from .faiss_store import FAISSStore

logger = logging.getLogger(__name__)

class VectorManager:
    """High-level manager for vector store operations"""
    
    def __init__(self, embedding_model: Embeddings, store_path: str = "models/vector_db"):
        """
        Initialize vector manager
        
        Args:
            embedding_model: Embedding model for generating vectors
            store_path: Path to vector store
        """
        self.embedding_model = embedding_model
        self.store_path = Path(store_path)
        self.vector_store = FAISSStore(embedding_model, str(self.store_path))
        self.document_index = {}  # Maps document names to IDs
        self._load_document_index()
    
    def _load_document_index(self):
        """Load document index from file"""
        index_file = self.store_path / "document_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.document_index = json.load(f)
                logger.info(f"Loaded document index with {len(self.document_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load document index: {e}")
                self.document_index = {}
    
    def _save_document_index(self):
        """Save document index to file"""
        index_file = self.store_path / "document_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.document_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save document index: {e}")
    
    def add_document(self, document_name: str, chunks: List[Document], 
                    metadata: Dict = None) -> bool:
        """
        Add a document to the vector store
        
        Args:
            document_name: Name of the document
            chunks: List of document chunks
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add metadata to each chunk
            base_metadata = metadata or {}
            base_metadata['document_name'] = document_name
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update(base_metadata)
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(chunks)
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents(chunks)
            
            # Update document index
            self.document_index[document_name] = {
                'doc_ids': doc_ids,
                'chunk_count': len(chunks),
                'metadata': base_metadata,
                'added_at': str(pd.Timestamp.now())
            }
            
            # Save index
            self._save_document_index()
            
            logger.info(f"Added document '{document_name}' with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document '{document_name}': {e}")
            return False
    
    def remove_document(self, document_name: str) -> bool:
        """
        Remove a document from the vector store
        
        Args:
            document_name: Name of the document to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if document_name not in self.document_index:
                logger.warning(f"Document '{document_name}' not found in index")
                return False
            
            # Get document IDs
            doc_ids = self.document_index[document_name]['doc_ids']
            
            # Remove from vector store
            success = self.vector_store.delete_documents(doc_ids)
            
            if success:
                # Remove from index
                del self.document_index[document_name]
                self._save_document_index()
                logger.info(f"Removed document '{document_name}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to remove document '{document_name}': {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5, 
                        document_filter: List[str] = None,
                        language_filter: str = None) -> List[Document]:
        """
        Search documents with optional filters
        
        Args:
            query: Search query
            k: Number of results to return
            document_filter: List of document names to search in
            language_filter: Language to filter by
            
        Returns:
            List of relevant documents
        """
        try:
            # Build filter dictionary
            filter_dict = {}
            if document_filter:
                filter_dict['document_name'] = {'$in': document_filter}
            if language_filter:
                filter_dict['language'] = language_filter
            
            # Perform search
            if filter_dict:
                results = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5,
                          document_filter: List[str] = None,
                          language_filter: str = None) -> List[Tuple[Document, float]]:
        """
        Search documents with relevance scores
        
        Args:
            query: Search query
            k: Number of results to return
            document_filter: List of document names to search in
            language_filter: Language to filter by
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Build filter dictionary
            filter_dict = {}
            if document_filter:
                filter_dict['document_name'] = {'$in': document_filter}
            if language_filter:
                filter_dict['language'] = language_filter
            
            # Perform search with scores
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents with scores: {e}")
            return []
    
    def get_document_chunks(self, document_name: str) -> List[Document]:
        """
        Get all chunks for a specific document
        
        Args:
            document_name: Name of the document
            
        Returns:
            List of document chunks
        """
        try:
            if document_name not in self.document_index:
                logger.warning(f"Document '{document_name}' not found")
                return []
            
            doc_ids = self.document_index[document_name]['doc_ids']
            chunks = []
            
            for doc_id in doc_ids:
                doc = self.vector_store.get_document_by_id(doc_id)
                if doc:
                    chunks.append(doc)
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x.metadata.get('chunk_index', 0))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return []
    
    def get_document_info(self, document_name: str) -> Optional[Dict]:
        """
        Get information about a document
        
        Args:
            document_name: Name of the document
            
        Returns:
            Document information dictionary
        """
        return self.document_index.get(document_name)
    
    def list_documents(self) -> List[str]:
        """
        List all document names in the store
        
        Returns:
            List of document names
        """
        return list(self.document_index.keys())
    
    def get_store_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive store statistics
        
        Returns:
            Dictionary with store statistics
        """
        try:
            stats = self.vector_store.get_stats()
            
            # Add document-level stats
            stats['total_documents'] = len(self.document_index)
            stats['document_names'] = list(self.document_index.keys())
            
            # Calculate total chunks
            total_chunks = sum(
                doc_info['chunk_count'] 
                for doc_info in self.document_index.values()
            )
            stats['total_chunks'] = total_chunks
            
            # Language distribution
            language_dist = {}
            for doc_info in self.document_index.values():
                lang = doc_info['metadata'].get('language', 'unknown')
                language_dist[lang] = language_dist.get(lang, 0) + 1
            stats['language_distribution'] = language_dist
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get store stats: {e}")
            return {}
    
    def export_document(self, document_name: str, output_path: str) -> bool:
        """
        Export a specific document
        
        Args:
            document_name: Name of the document to export
            output_path: Path to save the exported document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            chunks = self.get_document_chunks(document_name)
            if not chunks:
                logger.warning(f"No chunks found for document '{document_name}'")
                return False
            
            # Combine chunks into full document
            full_text = '\n\n'.join(chunk.page_content for chunk in chunks)
            
            # Save to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            logger.info(f"Exported document '{document_name}' to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export document: {e}")
            return False
    
    def clear_all(self):
        """Clear all documents from the store"""
        try:
            self.vector_store.clear_store()
            self.document_index = {}
            self._save_document_index()
            logger.info("Cleared all documents from vector store")
            
        except Exception as e:
            logger.error(f"Failed to clear store: {e}")
    
    def backup_store(self, backup_path: str) -> bool:
        """
        Create a backup of the vector store
        
        Args:
            backup_path: Path to save the backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy vector store files
            import shutil
            shutil.copytree(self.store_path, backup_path / "vector_store")
            
            # Save document index
            with open(backup_path / "document_index.json", 'w') as f:
                json.dump(self.document_index, f, indent=2)
            
            logger.info(f"Created backup at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
