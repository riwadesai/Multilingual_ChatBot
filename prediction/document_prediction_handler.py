
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
import re
from collections import Counter
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPredictionHandler:
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.embedding_model = None
        self.chat_history = []  # Store chat history
        
        # Create .env file if it doesn't exist
        self.create_env_file_if_needed()
        
        # Direct transformers configuration (memory-efficient Gemma-2B)
        self.model_name = "google/gemma-2b"
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
        logger.info(f"DocumentPredictionHandler initialized")
    
    
    def create_env_file_if_needed(self):
        """Create .env file with Hugging Face API token placeholder if it doesn't exist."""
        env_file = Path(".env")
        if not env_file.exists():
            env_content = """# Hugging Face API Configuration
# Get your API token from: https://huggingface.co/settings/tokens
HUGGINGFACE_API_TOKEN=your_token_here
"""
            try:
                with open(env_file, 'w') as f:
                    f.write(env_content)
                logger.info("Created .env file with Hugging Face API token placeholder")
                logger.info("Update .env file with your actual Hugging Face API token")
            except Exception as e:
                logger.warning(f"Could not create .env file: {str(e)}")
        else:
            logger.info(".env file already exists")
    
    
    def load_model(self):
        if self.pipeline is None:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                import torch
                
                logger.info(f"Loading model: {self.model_name}")
                
                # Load tokenizer and model with memory optimization
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=os.getenv("HUGGINGFACE_API_TOKEN")
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    token=os.getenv("HUGGINGFACE_API_TOKEN")
                )
                
                # Create pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu"
                )
                
                logger.info("Model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise Exception(f"Model loading failed: {str(e)}")
    
    def generate_text(self, messages: list, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using the loaded model."""
        try:
            # Load model if not already loaded
            self.load_model()
            
            # Check if tokenizer has chat template
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                # Use chat template if available
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)
            else:
                # Convert messages to simple text prompt for models without chat template
                prompt = self._messages_to_text(messages)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise Exception(f"Text generation failed: {str(e)}")
    
    def _messages_to_text(self, messages: list) -> str:
        """Convert messages to simple text prompt for models without chat template."""
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant:"
        return prompt
    
    def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:

        try:
            # Extract parameters from event
            query = event.get("query")
            language = event.get("language")
            chat_history = event.get("chat_history", [])
            
            # Validate event
            if not query or not language:
                return {
                    'success': False,
                    'error': "Event must contain 'query' and 'language'"
                }
            
            logger.info(f"Processing query: '{query}' in language: {language}")
            
            # Step 1: Load metadata and FAISS index
            metadata, faiss_index = self.load_data(language)
            
            # Step 2: Embed query using BAAI/bge-m3
            query_embedding = self.query_embedding(query, language)
            
            # Step 3: MMR Search (top 10 chunks)
            mmr_results = self.faiss_search(query_embedding, faiss_index, metadata, top_k=10)
            
            # Step 4: Keyword Search (top 5 chunks)
            keyword_results = self.keyword_search(query, metadata, top_k=5)
            
            # Step 5: Deduplication
            deduplicated_results = self.deduplicate_chunks(mmr_results, keyword_results)
            
            # Step 6: Reranking with LLM
            reranked_results = self.rerank_chunks(deduplicated_results, query, language)
            
            # Step 7: Filter by rerank score (>= 2)
            filtered_results = [chunk for chunk in reranked_results if chunk.get('rerank_score', 0) >= 2]
            
            # Step 8: Generate answer
            answer = self.generate_answer(query, filtered_results, chat_history, language)
            
            # Update chat history
            self.chat_history.append({"query": query, "answer": answer})
            
            return {
                'success': True,
                'query': query,
                'answer': answer,
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    
    def load_data(self, language: str) -> tuple:
        logger.info(f"Loading data for language: {language}")
        
        # Load metadata
        metadata_path = self.data_dir / f"metadata_{language}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load FAISS index
        faiss_path = self.data_dir / f"faiss_index_{language}.pkl"
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {faiss_path}")
        
        with open(faiss_path, 'rb') as f:
            faiss_index = pickle.load(f)
        
        logger.info(f"Loaded metadata and faiss index")
        return metadata, faiss_index
    
    def query_embedding(self, query: str, language: str) -> List[float]:

        logger.info(f"Generating embedding for query: '{query}'")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load BAAI/bge-m3 model if not already loaded
            if self.embedding_model is None:
                logger.info("Loading BAAI/bge-m3 model...")
                self.embedding_model = SentenceTransformer('BAAI/bge-m3')
            
            # Generate embedding
            embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
            
            logger.info(f"Generated embedding with dimension {len(embedding)}")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise Exception(f"Failed to generate query embedding: {str(e)}")
    
    def faiss_search(self, query_embedding: List[float], faiss_index: List[List[float]], 
                   metadata: Dict, top_k: int = 10, lambda_param: float = 0.7) -> List[Dict]:

        logger.info(f"Performing MMR search with lambda={lambda_param}")
        
        # Convert to numpy arrays
        query_vec = np.array(query_embedding).reshape(1, -1)
        doc_embeddings = np.array(faiss_index)
        
        # Calculate similarities
        similarities = np.dot(query_vec, doc_embeddings.T).flatten()
        
        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(doc_embeddings)))
        
        # Select first document (highest similarity)
        first_idx = np.argmax(similarities)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select documents with highest MMR score
        for _ in range(min(top_k - 1, len(remaining_indices))):
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance score
                relevance = similarities[idx]
                
                # Diversity score (max similarity to already selected docs)
                if selected_indices:
                    diversity = max([
                        np.dot(doc_embeddings[idx], doc_embeddings[sel_idx])
                        for sel_idx in selected_indices
                    ])
                else:
                    diversity = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Prepare results
        results = []
        for i, idx in enumerate(selected_indices):
            chunk_id = str(idx)
            if chunk_id in metadata:
                chunk_data = metadata[chunk_id]
                results.append({
                    'chunk_id': chunk_id,
                    'content': chunk_data['content'],
                    'lang': chunk_data['lang'],
                    'mmr_score': similarities[idx],
                    'rank': i + 1
                })
        
        logger.info(f"MMR search returned {len(results)} results")
        return results
    
    def keyword_search(self, query: str, metadata: Dict, top_k: int = 5) -> List[Dict]:

        logger.info(f"Performing keyword search for: '{query}'")
        
        # Extract keywords from query
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Score each chunk
        chunk_scores = []
        for chunk_id, chunk_data in metadata.items():
            content = chunk_data['content'].lower()
            content_words = set(re.findall(r'\b\w+\b', content))
            
            # Calculate keyword match score
            matches = query_words.intersection(content_words)
            match_score = len(matches) / len(query_words) if query_words else 0
            
            if match_score > 0:
                chunk_scores.append({
                    'chunk_id': chunk_id,
                    'content': chunk_data['content'],
                    'lang': chunk_data['lang'],
                    'keyword_score': match_score,
                    'matches': list(matches)
                })
        
        # Sort by keyword score and return top_k
        chunk_scores.sort(key=lambda x: x['keyword_score'], reverse=True)
        results = chunk_scores[:top_k]
        
        logger.info(f"Keyword search returned {len(results)} results")
        return results
    
    def deduplicate_chunks(self, mmr_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:

        logger.info("Deduplicating search results...")
        
        seen_chunks = set()
        deduplicated = []
        
        # Add MMR results first (they have better semantic relevance)
        for chunk in mmr_results:
            chunk_id = chunk['chunk_id']
            if chunk_id not in seen_chunks:
                chunk['search_type'] = 'mmr'
                deduplicated.append(chunk)
                seen_chunks.add(chunk_id)
        
        # Add keyword results that aren't already included
        for chunk in keyword_results:
            chunk_id = chunk['chunk_id']
            if chunk_id not in seen_chunks:
                chunk['search_type'] = 'keyword'
                deduplicated.append(chunk)
                seen_chunks.add(chunk_id)
        
        logger.info(f"Deduplication: {len(mmr_results)} MMR + {len(keyword_results)} keyword = {len(deduplicated)} unique chunks")
        return deduplicated
    
    def rerank_chunks(self, chunks: List[Dict], query: str, language: str) -> List[Dict]:
        
        try:
            # Limit chunks for faster processing (take top 6 instead of all 12)
            chunks_to_process = chunks[:6]
            logger.info(f"Reranking top {len(chunks_to_process)} chunks for speed...")
            
            # Rerank each chunk using the loaded model
            for i, chunk in enumerate(chunks_to_process):
                content = chunk['content']
                
                # Create reranking messages with shorter content for speed
                messages = [
                    {"role": "system", "content": "Score relevance 1-10. Respond with just the number."},
                    {"role": "user", "content": f"Query: {query}\n\nText: {content[:300]}...\n\nScore (1-10):"}
                ]
                
                # Generate response with minimal tokens for speed
                response = self.generate_text(
                    messages=messages,
                    max_new_tokens=3,
                    temperature=0.1
                )
                
                # Extract score from response
                rerank_score = self.rerank_score(response)
                chunk['rerank_score'] = rerank_score
                
                logger.info(f"Chunk {i+1}/{len(chunks_to_process)}: Score {rerank_score}")
            
            # Sort by rerank score (descending)
            chunks_to_process.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top 3 reranked chunks
            top_chunks = chunks_to_process[:3]
            logger.info(f"Reranking completed. Top 3 scores: {[c['rerank_score'] for c in top_chunks]}")
            return top_chunks
            
        except Exception as e:
            logger.error(f"Error in LLM reranking: {str(e)}")
            raise Exception(f"Reranking failed: {str(e)}")
    
    
    def rerank_score(self, response: str) -> int:
        # Extract number from response
        import re
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        
        if numbers:
            score = int(numbers[0])
            return max(1, min(10, score))  # Clamp to 1-10
        else:
            return 5  # Default score if extraction fails
    
    def generate_answer(self, query: str, chunks: List[Dict], chat_history: List[Dict], language: str) -> str:

        
        try:
            # Create answer generation messages
            messages = self.create_answer_messages(query, chunks, chat_history, language)
            
            # Generate answer using the loaded model
            answer = self.generate_text(
                messages=messages,
                max_new_tokens=512,
                temperature=0.7
            )
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error in LLM answer generation: {str(e)}")
            raise Exception(f"Answer generation failed: {str(e)}")
    
    def create_answer_messages(self, query: str, chunks: List[Dict], chat_history: List[Dict], language: str) -> list:

        # Prepare context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"Document {i}:\n{chunk['content']}")
        context = "\n\n".join(context_parts)
        
        # Prepare chat history
        history_text = ""
        if chat_history:
            history_parts = []
            for h in chat_history[-3:]:  # Last 3 exchanges
                history_parts.append(f"User: {h['query']}\nAssistant: {h['answer']}")
            history_text = "\n\n".join(history_parts)
        
        # Language-specific instructions
        lang_instructions = {
            'ben': 'বাংলা ভাষায় উত্তর দিন',
            'hin': 'हिंदी भाषा में उत्तर दें',
            'eng': 'Answer in English',
            'urd': 'اردو زبان میں جواب دیں',
            'chi_sim': '用中文回答',
            'chi_tra': '用繁體中文回答',
            'ara': 'أجب باللغة العربية',
            'fra': 'Répondez en français',
            'deu': 'Antworten Sie auf Deutsch',
            'spa': 'Responde en español',
            'por': 'Responda em português',
            'rus': 'Ответьте на русском языке',
            'jpn': '日本語で答えてください',
            'kor': '한국어로 답변하세요',
            'tha': 'ตอบเป็นภาษาไทย',
            'vie': 'Trả lời bằng tiếng Việt'
        }
        
        lang_instruction = lang_instructions.get(language, f'Answer in {language}')
        
        # Create messages for the chat template
        messages = [
            {
                "role": "system", 
                "content": f"""You are a helpful multilingual assistant that answers questions based on provided document context. Your task is to provide accurate, comprehensive, and well-structured answers.

Guidelines:
- Use only information from the provided documents
- If the documents don't contain enough information, say so clearly
- Be concise but comprehensive
- Maintain the same language as the question
- Cite relevant document sections when appropriate
- If you're unsure about something, express that uncertainty

{lang_instruction}"""
            },
            {
                "role": "user",
                "content": f"""Previous Conversation:
{history_text}

Relevant Documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the documents above:"""
            }
        ]
        
        return messages
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Cleanup embedding model if it exists
            if self.embedding_model is not None:
                del self.embedding_model
                self.embedding_model = None
            
            # Cleanup main model if it exists
            if self.model is not None:
                del self.model
                self.model = None
                
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
    
