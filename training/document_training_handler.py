import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentTrainingHandler:

    
    def __init__(self, output_dir: str = "../data/processed"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.document_parser = None
        self.text_chunker = None
        self.embedding_generator = None
        self.vector_store = None
        
    
    def train_document(self, event: Dict[str, Any]) -> Dict[str, Any]:

        try:
            document_path = event.get("document_path")
            language = event.get("document_language")
            
            if not document_path or not language:
                return {
                    'success': False,
                    'error': "Event must contain 'document_path' and 'document_language'"
                }
            
            logger.info(f"Starting training for document: {document_path}")
            logger.info(f"Event: {event}")
            
            # Step 1: Parse document
            logger.info("Extracting data from pdf")
            parsed_content = self.data_extractor(document_path, language)
            logger.info(f"data extracted from pdf")
            
            # Step 2: Chunk the text
            logger.info("Step 2: Chunking text...")
            chunks = self.chunk_data(parsed_content, language)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            logger.info("Step 3: Generating embeddings...")
            embeddings = self._generate_embeddings(chunks, language)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 4: Save to vector store
            logger.info("Step 4: Saving to vector store...")
            metadata_path, faiss_path = self._save_to_vector_store(
                chunks, embeddings, document_path, language
            )
            
            return {
                'success': True,
                'chunk_count': len(chunks),
                'metadata_path': str(metadata_path),
                'faiss_path': str(faiss_path)
            }
            
        except Exception as e:
            logger.error(f"Error during document training: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def data_extractor(self, document_path: str, language: str = 'eng') -> str:
        logger.info(f"Extracting text from PDF: {document_path}")
        logger.info(f"Using language: {language}")
        
        os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'
        
        return self._tesseract_extraction(document_path, language)
    
    def _tesseract_extraction(self, document_path: str, language: str = 'eng') -> str:

        try:
            import fitz 
            import pytesseract
            from PIL import Image
            import io
            
            logger.info(f"Using Tesseract OCR for {language} text extraction")

            if language.lower() != 'eng':
                tesseract_lang = f"{language.lower()}+eng"
            else:
                tesseract_lang = 'eng'
            
            logger.info(f"Using Tesseract language: {tesseract_lang}")
            
            doc = fitz.open(document_path)
            text_content = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Convert page to image with high resolution for better OCR
                mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Perform OCR with Tesseract using specified language
                page_text = pytesseract.image_to_string(
                    image, 
                    lang=tesseract_lang,  # Use mapped language
                    config='--psm 6'  # Assume uniform block of text
                )
                
                if page_text.strip():
                    text_content += f"\n---PAGE {page_num + 1} (TESSERACT)---\n{page_text.strip()}\n"
            
            doc.close()
            
            lang_chars = sum(1 for char in text_content if '\u0980' <= char <= '\u09FF')
            total_chars = len(text_content.replace(' ', '').replace('\n', ''))
            lang_ratio = lang_chars / total_chars if total_chars > 0 else 0
            
            logger.info(f"Tesseract OCR extraction completed:")
            logger.info(f"  - Total characters: {len(text_content)}")
            logger.info(f"  - Language: {language}")
            logger.info(f"  - lang characters: {lang_chars} ({lang_ratio:.2%})")
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error in Tesseract OCR extraction: {str(e)}")
            return f"Error extracting text from PDF {document_path}: {str(e)}"
        
    def chunk_data(self, text: str, language: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Chunk text using RecursiveCharacterTextSplitter for better segmentation.
        
        Args:
            text: Text content to chunk
            language: Language of the text
            chunk_size: Size of each chunk in characters
            
        Returns:
            List of text chunks with metadata
        """
        logger.info(f"Chunking text using RecursiveCharacterTextSplitter with chunk size {chunk_size}")
        
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Create text splitter with appropriate settings
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 10,  # 10% overlap
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            # Split text into chunks
            text_chunks = text_splitter.split_text(text)
            
            # Create chunk objects
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    'chunk_index': i,  # This will be updated when saving
                    'content': chunk_text.strip(),
                    'language': language
                })
            
            logger.info(f"Created {len(chunks)} chunks using RecursiveCharacterTextSplitter")
            return chunks
            
        except ImportError:
            logger.warning("LangChain not installed. Using simple equal-sized chunking.")
            return self._simple_equal_chunking(text, language, chunk_size)
        except Exception as e:
            logger.error(f"Error in RecursiveCharacterTextSplitter: {str(e)}")
            return self._simple_equal_chunking(text, language, chunk_size)
    
    def _simple_equal_chunking(self, text: str, language: str, chunk_size: int) -> List[Dict[str, Any]]:
        """
        Simple fallback chunking method for equal-sized chunks.
        
        Args:
            text: Text content to chunk
            language: Language of the text
            chunk_size: Size of each chunk in characters
            
        Returns:
            List of text chunks with metadata
        """
        logger.info(f"Using simple equal-sized chunking with {chunk_size} characters per chunk")
        
        chunks = []
        chunk_index = 0
        
        # Split text into equal-sized chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            
            chunks.append({
                'chunk_index': chunk_index,  # This will be updated when saving
                'content': chunk,
                'language': language
            })
            
            chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks using simple chunking")
        return chunks
    
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]], language: str) -> List[List[float]]:
        """
        Generate embeddings for text chunks using BAAI/bge-m3 model.
        
        Args:
            chunks: List of text chunks
            language: Language of the text
            
        Returns:
            List of embeddings for each chunk
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks in language: {language}")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load BAAI/bge-m3 model for multilingual embeddings
            logger.info("Loading BAAI/bge-m3 model...")
            model = SentenceTransformer('BAAI/bge-m3')
            
            # Extract text content from chunks
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            logger.info("Generating embeddings with BAAI/bge-m3...")
            embeddings = model.encode(texts, normalize_embeddings=True)
            
            # Convert to list of lists
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            logger.info(f"Generated {len(embeddings_list)} embeddings with dimension {len(embeddings_list[0])}")
            return embeddings_list
            
        except ImportError:
            logger.error("sentence-transformers not installed. Please install it with: pip install sentence-transformers")
            raise ImportError("sentence-transformers is required for embedding generation. Please install it.")
        except Exception as e:
            logger.error(f"Error generating embeddings with BAAI/bge-m3: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def _save_to_vector_store(self, chunks: List[Dict[str, Any]], 
                            embeddings: List[List[float]], 
                            document_path: str, 
                            language: str) -> tuple:
        """
        Save chunks and embeddings to vector store, appending to existing data if it exists.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embeddings
            document_path: Original document path
            language: Language of the document
            
        Returns:
            Tuple of (metadata_path, faiss_path)
        """
        logger.info("Saving to vector store...")
        
        metadata_path = self.output_dir / f"metadata_{language}.json"
        faiss_path = self.output_dir / f"faiss_index_{language}.pkl"
        
        # Load existing metadata if it exists
        existing_metadata = {}
        existing_embeddings = []
        
        if metadata_path.exists():
            logger.info(f"Loading existing metadata from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
            
            if faiss_path.exists():
                logger.info(f"Loading existing embeddings from {faiss_path}")
                with open(faiss_path, 'rb') as f:
                    existing_embeddings = pickle.load(f)
        
        # Find the next available chunk index
        max_chunk_index = -1
        if existing_metadata:
            for chunk_id in existing_metadata.keys():
                try:
                    chunk_index = int(chunk_id)
                    max_chunk_index = max(max_chunk_index, chunk_index)
                except ValueError:
                    continue
        
        # Add new chunks with updated indices
        new_chunk_start_index = max_chunk_index + 1
        for i, chunk in enumerate(chunks):
            new_chunk_id = str(new_chunk_start_index + i)
            existing_metadata[new_chunk_id] = {
                "content": chunk['content'],
                "lang": chunk['language']
            }
        
        # Combine embeddings
        all_embeddings = existing_embeddings + embeddings
        
        # Save updated metadata (only chunks in the specified format)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(existing_metadata, f, indent=2, ensure_ascii=False)
        
        # Save updated FAISS index
        with open(faiss_path, 'wb') as f:
            pickle.dump(all_embeddings, f)
        
        logger.info(f"Saved metadata to: {metadata_path}")
        logger.info(f"Saved FAISS index to: {faiss_path}")
        logger.info(f"Total chunks now: {len(existing_metadata)}")
        logger.info(f"New chunks added: {len(chunks)}")
        
        return metadata_path, faiss_path
