"""
Query Processor Module
Handles query processing, expansion, and optimization for multilingual RAG
"""

import re
from typing import List, Dict, Optional, Tuple
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Processes and optimizes queries for multilingual RAG system"""
    
    def __init__(self):
        """Initialize query processor"""
        self.stop_words = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
            'hi': {'और', 'या', 'लेकिन', 'में', 'पर', 'को', 'के', 'लिए', 'से', 'द्वारा'},
            'bn': {'এবং', 'বা', 'কিন্তু', 'এ', 'এতে', 'কো', 'এর', 'জন্য', 'থেকে', 'দ্বারা'},
            'zh': {'的', '了', '在', '是', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        }
    
    def preprocess_query(self, query: str, language: str = 'en') -> str:
        """
        Preprocess query for better retrieval
        
        Args:
            query: Input query
            language: Language code
            
        Returns:
            Preprocessed query
        """
        if not query or not query.strip():
            return ""
        
        # Clean and normalize
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)  # Remove extra whitespace
        
        # Remove special characters but keep important punctuation
        query = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', query)
        
        # Convert to lowercase for consistency
        query = query.lower()
        
        return query
    
    def expand_query(self, query: str, language: str = 'en') -> List[str]:
        """
        Expand query with synonyms and related terms
        
        Args:
            query: Input query
            language: Language code
            
        Returns:
            List of expanded query variations
        """
        expanded_queries = [query]
        
        # Simple expansion based on common patterns
        if language == 'en':
            # Add question variations
            if query.endswith('?'):
                expanded_queries.append(query[:-1])  # Remove question mark
            else:
                expanded_queries.append(query + '?')
            
            # Add "what is" variations
            if not query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
                expanded_queries.append(f"what is {query}")
                expanded_queries.append(f"how to {query}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in expanded_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries
    
    def extract_keywords(self, query: str, language: str = 'en', 
                        max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from query
        
        Args:
            query: Input query
            language: Language code
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        if not query:
            return []
        
        # Clean query
        cleaned_query = self.preprocess_query(query, language)
        
        # Extract words
        words = re.findall(r'\b\w+\b', cleaned_query.lower())
        
        # Remove stop words
        stop_words = self.stop_words.get(language, self.stop_words['en'])
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:max_keywords]
    
    def create_search_queries(self, query: str, language: str = 'en') -> List[str]:
        """
        Create multiple search query variations
        
        Args:
            query: Input query
            language: Language code
            
        Returns:
            List of search query variations
        """
        queries = []
        
        # Original query
        queries.append(query)
        
        # Preprocessed query
        preprocessed = self.preprocess_query(query, language)
        if preprocessed != query:
            queries.append(preprocessed)
        
        # Expanded queries
        expanded = self.expand_query(query, language)
        queries.extend(expanded)
        
        # Keyword-based queries
        keywords = self.extract_keywords(query, language)
        if keywords:
            queries.append(' '.join(keywords))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query intent and type
        
        Args:
            query: Input query
            
        Returns:
            Dictionary with intent analysis
        """
        query_lower = query.lower()
        
        # Question type detection
        question_types = {
            'what': query_lower.startswith('what'),
            'how': query_lower.startswith('how'),
            'why': query_lower.startswith('why'),
            'when': query_lower.startswith('when'),
            'where': query_lower.startswith('where'),
            'who': query_lower.startswith('who'),
            'which': query_lower.startswith('which')
        }
        
        # Intent detection
        intents = {
            'definition': any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']),
            'how_to': any(word in query_lower for word in ['how to', 'how do', 'how can', 'steps', 'process']),
            'comparison': any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']),
            'explanation': any(word in query_lower for word in ['explain', 'why', 'reason', 'cause']),
            'factual': any(word in query_lower for word in ['when', 'where', 'who', 'which']),
            'procedural': any(word in query_lower for word in ['steps', 'process', 'procedure', 'method'])
        }
        
        # Complexity analysis
        word_count = len(query.split())
        has_question_mark = '?' in query
        has_multiple_questions = query.count('?') > 1
        
        return {
            'question_types': question_types,
            'intents': intents,
            'complexity': {
                'word_count': word_count,
                'has_question_mark': has_question_mark,
                'has_multiple_questions': has_multiple_questions,
                'is_complex': word_count > 10 or has_multiple_questions
            },
            'primary_intent': max(intents.items(), key=lambda x: x[1])[0] if any(intents.values()) else 'general'
        }
    
    def optimize_for_retrieval(self, query: str, language: str = 'en') -> str:
        """
        Optimize query for better retrieval performance
        
        Args:
            query: Input query
            language: Language code
            
        Returns:
            Optimized query
        """
        # Preprocess
        optimized = self.preprocess_query(query, language)
        
        # Remove very common words that might hurt retrieval
        stop_words = self.stop_words.get(language, self.stop_words['en'])
        words = optimized.split()
        filtered_words = [word for word in words if word not in stop_words or len(word) > 3]
        
        if filtered_words:
            optimized = ' '.join(filtered_words)
        
        # Ensure minimum length
        if len(optimized.split()) < 2:
            return query  # Return original if optimization makes it too short
        
        return optimized
    
    def create_rerank_queries(self, query: str, retrieved_docs: List[Document]) -> List[Tuple[str, Document]]:
        """
        Create query-document pairs for reranking
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents
            
        Returns:
            List of (query, document) pairs
        """
        pairs = []
        
        for doc in retrieved_docs:
            # Create different query variations for reranking
            variations = [
                query,  # Original query
                self.optimize_for_retrieval(query),  # Optimized query
                ' '.join(self.extract_keywords(query))  # Keyword-only query
            ]
            
            for variation in variations:
                if variation and variation.strip():
                    pairs.append((variation, doc))
        
        return pairs
    
    def extract_entities(self, query: str) -> List[str]:
        """
        Extract potential entities from query
        
        Args:
            query: Input query
            
        Returns:
            List of potential entities
        """
        entities = []
        
        # Simple entity extraction based on patterns
        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized)
        
        # Quoted strings
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        # Numbers and dates
        numbers = re.findall(r'\b\d+\b', query)
        entities.extend(numbers)
        
        # Remove duplicates
        return list(set(entities))
    
    def create_metadata_filter(self, query: str, language: str = 'en') -> Dict[str, Any]:
        """
        Create metadata filter based on query analysis
        
        Args:
            query: Input query
            language: Language code
            
        Returns:
            Metadata filter dictionary
        """
        filter_dict = {}
        
        # Language filter
        if language and language != 'en':
            filter_dict['language'] = language
        
        # Extract entities for potential filtering
        entities = self.extract_entities(query)
        if entities:
            # Could filter by document names containing entities
            pass
        
        return filter_dict
