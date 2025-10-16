"""
RAG Pipeline Module
Main RAG pipeline that orchestrates retrieval, reranking, and generation
"""

from typing import List, Dict, Optional, Tuple, Any
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import logging

from .llm_manager import LLMManager
from .query_processor import QueryProcessor
from ..vector_store.vector_manager import VectorManager
from ..embeddings.reranker import Reranker

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline for multilingual question answering"""
    
    def __init__(self, vector_manager: VectorManager, embedding_model: Embeddings,
                 llm_manager: LLMManager, reranker: Reranker = None):
        """
        Initialize RAG pipeline
        
        Args:
            vector_manager: Vector store manager
            embedding_model: Embedding model
            llm_manager: LLM manager
            reranker: Optional reranker for better retrieval
        """
        self.vector_manager = vector_manager
        self.embedding_model = embedding_model
        self.llm_manager = llm_manager
        self.reranker = reranker
        self.query_processor = QueryProcessor()
        
        # Default parameters
        self.default_retrieval_k = 10
        self.default_rerank_k = 5
        self.default_generation_k = 3
        
        # Prompt templates
        self.prompt_templates = {
            'en': """Context: {context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so.""",
            
            'hi': """संदर्भ: {context}

प्रश्न: {question}

कृपया उपरोक्त संदर्भ के आधार पर एक व्यापक उत्तर दें। यदि संदर्भ में प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है, तो कृपया यह बताएं।""",
            
            'bn': """প্রসঙ্গ: {context}

প্রশ্ন: {question}

অনুগ্রহ করে উপরের প্রসঙ্গের ভিত্তিতে একটি বিস্তৃত উত্তর দিন। যদি প্রসঙ্গে প্রশ্নের উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য না থাকে, অনুগ্রহ করে তা বলুন।"""
        }
    
    def answer_question(self, question: str, language: str = 'en',
                       document_filter: List[str] = None,
                       use_reranking: bool = True,
                       max_context_length: int = 2000) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline
        
        Args:
            question: User question
            language: Language code
            document_filter: Optional list of document names to search in
            use_reranking: Whether to use reranking
            max_context_length: Maximum context length for generation
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Step 1: Process and expand query
            search_queries = self.query_processor.create_search_queries(question, language)
            logger.info(f"Generated {len(search_queries)} search query variations")
            
            # Step 2: Retrieve relevant documents
            all_documents = []
            for query in search_queries:
                docs = self.vector_manager.search_documents(
                    query=query,
                    k=self.default_retrieval_k,
                    document_filter=document_filter,
                    language_filter=language
                )
                all_documents.extend(docs)
            
            # Remove duplicates while preserving order
            seen_content = set()
            unique_documents = []
            for doc in all_documents:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_documents.append(doc)
            
            logger.info(f"Retrieved {len(unique_documents)} unique documents")
            
            if not unique_documents:
                return {
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'sources': [],
                    'confidence': 0.0,
                    'metadata': {
                        'retrieval_count': 0,
                        'reranking_used': False,
                        'language': language
                    }
                }
            
            # Step 3: Rerank documents if reranker is available
            if use_reranking and self.reranker:
                logger.info("Reranking documents...")
                reranked_docs = self.reranker.rerank_langchain_docs(
                    question, unique_documents, top_k=self.default_rerank_k
                )
                final_documents = reranked_docs
                reranking_used = True
            else:
                final_documents = unique_documents[:self.default_rerank_k]
                reranking_used = False
            
            # Step 4: Prepare context
            context = self._prepare_context(final_documents, max_context_length)
            
            # Step 5: Generate answer
            answer = self._generate_answer(question, context, language)
            
            # Step 6: Prepare sources
            sources = self._prepare_sources(final_documents)
            
            # Step 7: Calculate confidence
            confidence = self._calculate_confidence(final_documents, answer)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'metadata': {
                    'retrieval_count': len(unique_documents),
                    'final_documents': len(final_documents),
                    'reranking_used': reranking_used,
                    'language': language,
                    'context_length': len(context)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'answer': "I apologize, but I encountered an error while processing your question.",
                'sources': [],
                'confidence': 0.0,
                'metadata': {
                    'error': str(e),
                    'language': language
                }
            }
    
    def _prepare_context(self, documents: List[Document], max_length: int) -> str:
        """
        Prepare context from documents
        
        Args:
            documents: List of documents
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            doc_text = doc.page_content
            doc_metadata = doc.metadata
            
            # Add document source info
            source_info = f"[Source {i+1}"
            if 'document_name' in doc_metadata:
                source_info += f" from {doc_metadata['document_name']}"
            if 'chunk_index' in doc_metadata:
                source_info += f", chunk {doc_metadata['chunk_index']}"
            source_info += "]: "
            
            # Check if adding this document would exceed length limit
            if current_length + len(source_info) + len(doc_text) > max_length:
                # Truncate the last document if needed
                remaining_length = max_length - current_length - len(source_info)
                if remaining_length > 100:  # Only add if there's meaningful content
                    truncated_text = doc_text[:remaining_length] + "..."
                    context_parts.append(source_info + truncated_text)
                break
            
            context_parts.append(source_info + doc_text)
            current_length += len(source_info) + len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, language: str) -> str:
        """
        Generate answer using LLM
        
        Args:
            question: User question
            context: Retrieved context
            language: Language code
            
        Returns:
            Generated answer
        """
        try:
            # Get appropriate prompt template
            template = self.prompt_templates.get(language, self.prompt_templates['en'])
            
            # Format prompt
            prompt = template.format(context=context, question=question)
            
            # Generate response
            answer = self.llm_manager.generate_response(prompt)
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I couldn't generate a proper answer."
    
    def _prepare_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Prepare source information for documents
        
        Args:
            documents: List of documents
            
        Returns:
            List of source dictionaries
        """
        sources = []
        
        for i, doc in enumerate(documents):
            source_info = {
                'rank': i + 1,
                'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'metadata': doc.metadata
            }
            
            # Add relevance score if available
            if hasattr(doc, 'relevance_score'):
                source_info['relevance_score'] = doc.relevance_score
            
            sources.append(source_info)
        
        return sources
    
    def _calculate_confidence(self, documents: List[Document], answer: str) -> float:
        """
        Calculate confidence score for the answer
        
        Args:
            documents: Retrieved documents
            answer: Generated answer
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Base confidence on number of documents and their relevance
            base_confidence = min(len(documents) / 5.0, 1.0)  # More documents = higher confidence
            
            # Adjust based on answer length (longer answers might be more confident)
            if len(answer) > 50:
                length_factor = min(len(answer) / 500.0, 1.0)
                base_confidence = (base_confidence + length_factor) / 2
            
            # Check if answer indicates uncertainty
            uncertainty_indicators = [
                "i don't know", "i'm not sure", "i can't find", "no information",
                "not available", "unclear", "uncertain"
            ]
            
            answer_lower = answer.lower()
            for indicator in uncertainty_indicators:
                if indicator in answer_lower:
                    base_confidence *= 0.5  # Reduce confidence if uncertain
                    break
            
            return min(max(base_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default moderate confidence
    
    def batch_answer_questions(self, questions: List[str], language: str = 'en',
                             document_filter: List[str] = None) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch
        
        Args:
            questions: List of questions
            language: Language code
            document_filter: Optional document filter
            
        Returns:
            List of answer dictionaries
        """
        results = []
        
        for question in questions:
            try:
                result = self.answer_question(
                    question=question,
                    language=language,
                    document_filter=document_filter
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                results.append({
                    'answer': f"Error processing question: {str(e)}",
                    'sources': [],
                    'confidence': 0.0,
                    'metadata': {'error': str(e)}
                })
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics
        
        Returns:
            Dictionary with pipeline statistics
        """
        try:
            vector_stats = self.vector_manager.get_store_stats()
            llm_info = self.llm_manager.get_model_info()
            
            return {
                'vector_store': vector_stats,
                'llm_model': llm_info,
                'reranker_available': self.reranker is not None,
                'supported_languages': list(self.prompt_templates.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {}
    
    def update_parameters(self, **kwargs):
        """
        Update pipeline parameters
        
        Args:
            **kwargs: Parameters to update
        """
        if 'retrieval_k' in kwargs:
            self.default_retrieval_k = kwargs['retrieval_k']
        
        if 'rerank_k' in kwargs:
            self.default_rerank_k = kwargs['rerank_k']
        
        if 'generation_k' in kwargs:
            self.default_generation_k = kwargs['generation_k']
        
        logger.info(f"Updated pipeline parameters: {kwargs}")
    
    def add_prompt_template(self, language: str, template: str):
        """
        Add or update prompt template for a language
        
        Args:
            language: Language code
            template: Prompt template with {context} and {question} placeholders
        """
        self.prompt_templates[language] = template
        logger.info(f"Added prompt template for language: {language}")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages
        
        Returns:
            List of language codes
        """
        return list(self.prompt_templates.keys())
