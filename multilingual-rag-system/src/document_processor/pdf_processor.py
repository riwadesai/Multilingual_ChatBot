"""
PDF Processing Module
Handles digital PDF text extraction and OCR for scanned PDFs
Supports multiple languages using EasyOCR
"""

import fitz  # PyMuPDF
import easyocr
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processes PDF documents with multilingual OCR support"""
    
    def __init__(self, languages: List[str] = None):
        """
        Initialize PDF processor with OCR support for specified languages
        
        Args:
            languages: List of language codes (e.g., ['en', 'hi', 'bn', 'zh'])
        """
        self.languages = languages or ['en']  # Default to English
        self.ocr_reader = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize EasyOCR reader for specified languages"""
        try:
            # Map language codes to EasyOCR format
            lang_map = {
                'en': 'en',
                'hi': 'hi', 
                'bn': 'bn',
                'zh': 'ch_sim',  # Simplified Chinese
                'es': 'es',
                'fr': 'fr',
                'de': 'de'
            }
            
            ocr_langs = [lang_map.get(lang, 'en') for lang in self.languages]
            self.ocr_reader = easyocr.Reader(ocr_langs, gpu=False)
            logger.info(f"Initialized OCR for languages: {ocr_langs}")
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            self.ocr_reader = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """
        Extract text from PDF using digital extraction and OCR
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted text, metadata, and processing info
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Open PDF document
        doc = fitz.open(pdf_path)
        
        result = {
            'file_name': pdf_path.name,
            'total_pages': len(doc),
            'pages': [],
            'full_text': '',
            'metadata': self._extract_metadata(doc),
            'processing_info': {
                'digital_extraction_pages': 0,
                'ocr_pages': 0,
                'languages_detected': set()
            }
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_result = self._process_page(page, page_num)
            result['pages'].append(page_result)
            result['full_text'] += page_result['text'] + '\n'
            
            # Update processing info
            if page_result['method'] == 'digital':
                result['processing_info']['digital_extraction_pages'] += 1
            else:
                result['processing_info']['ocr_pages'] += 1
            
            result['processing_info']['languages_detected'].update(
                page_result.get('languages', [])
            )
        
        doc.close()
        
        # Convert set to list for JSON serialization
        result['processing_info']['languages_detected'] = list(
            result['processing_info']['languages_detected']
        )
        
        logger.info(f"Processed {len(doc)} pages. "
                   f"Digital: {result['processing_info']['digital_extraction_pages']}, "
                   f"OCR: {result['processing_info']['ocr_pages']}")
        
        return result
    
    def _extract_metadata(self, doc) -> Dict:
        """Extract metadata from PDF document"""
        metadata = doc.metadata
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'modification_date': metadata.get('modDate', '')
        }
    
    def _process_page(self, page, page_num: int) -> Dict:
        """
        Process a single PDF page
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            
        Returns:
            Dictionary with page text and processing info
        """
        # Try digital text extraction first
        digital_text = page.get_text()
        
        if len(digital_text.strip()) > 50:  # Sufficient text found
            return {
                'page_number': page_num + 1,
                'text': digital_text,
                'method': 'digital',
                'confidence': 1.0,
                'languages': ['en']  # Assume English for digital text
            }
        
        # Fall back to OCR
        if self.ocr_reader is None:
            return {
                'page_number': page_num + 1,
                'text': '',
                'method': 'none',
                'confidence': 0.0,
                'languages': []
            }
        
        return self._ocr_page(page, page_num)
    
    def _ocr_page(self, page, page_num: int) -> Dict:
        """
        Perform OCR on a PDF page
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to OpenCV format
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Perform OCR
            results = self.ocr_reader.readtext(img)
            
            # Extract text and calculate confidence
            text_parts = []
            confidences = []
            languages = set()
            
            for bbox, text, confidence in results:
                if confidence > 0.5:  # Filter low-confidence results
                    text_parts.append(text)
                    confidences.append(confidence)
                    # Note: EasyOCR doesn't return language info directly
                    # We'll use language detection later
            
            full_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'page_number': page_num + 1,
                'text': full_text,
                'method': 'ocr',
                'confidence': avg_confidence,
                'languages': list(languages) if languages else ['unknown']
            }
            
        except Exception as e:
            logger.error(f"OCR failed for page {page_num + 1}: {e}")
            return {
                'page_number': page_num + 1,
                'text': '',
                'method': 'ocr_failed',
                'confidence': 0.0,
                'languages': []
            }
    
    def extract_images(self, pdf_path: str) -> List[Dict]:
        """
        Extract images from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        images.append({
                            'page_number': page_num + 1,
                            'image_index': img_index,
                            'data': img_data,
                            'format': 'png',
                            'width': pix.width,
                            'height': pix.height
                        })
                    
                    pix = None
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
        
        doc.close()
        return images
    
    def get_page_images(self, pdf_path: str, page_num: int) -> np.ndarray:
        """
        Get page as image array
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            
        Returns:
            Image array
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Convert to image with high resolution
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to OpenCV format
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        doc.close()
        return img
