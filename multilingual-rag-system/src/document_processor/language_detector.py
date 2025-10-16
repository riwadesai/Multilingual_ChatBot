"""
Language Detection Module
Detects language of text using multiple methods
"""

import re
from typing import List, Dict, Optional, Tuple
from langdetect import detect, detect_langs, LangDetectException
import logging

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Detects language of text using multiple methods"""
    
    def __init__(self):
        """Initialize language detector"""
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi', 
            'bn': 'Bengali',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ar': 'Arabic',
            'ja': 'Japanese',
            'ko': 'Korean'
        }
        
        # Character range patterns for different scripts
        self.script_patterns = {
            'latin': r'[a-zA-Z]',
            'devanagari': r'[\u0900-\u097F]',  # Hindi, Sanskrit
            'bengali': r'[\u0980-\u09FF]',     # Bengali
            'chinese': r'[\u4e00-\u9fff]',     # Chinese
            'arabic': r'[\u0600-\u06FF]',      # Arabic
            'cyrillic': r'[\u0400-\u04FF]',    # Russian, etc.
            'japanese': r'[\u3040-\u309F\u30A0-\u30FF]',  # Hiragana, Katakana
            'korean': r'[\uAC00-\uD7AF]',      # Korean
        }
    
    def detect_language(self, text: str, method: str = 'auto') -> Dict:
        """
        Detect language of text
        
        Args:
            text: Input text
            method: Detection method ('auto', 'langdetect', 'pattern', 'hybrid')
            
        Returns:
            Dictionary with detection results
        """
        if not text or len(text.strip()) < 10:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'method': 'insufficient_text',
                'alternatives': []
            }
        
        if method == 'auto':
            # Try langdetect first, fall back to pattern matching
            try:
                result = self._detect_with_langdetect(text)
                if result['confidence'] > 0.5:
                    return result
            except Exception:
                pass
            
            return self._detect_with_patterns(text)
        
        elif method == 'langdetect':
            return self._detect_with_langdetect(text)
        
        elif method == 'pattern':
            return self._detect_with_patterns(text)
        
        elif method == 'hybrid':
            return self._detect_hybrid(text)
        
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _detect_with_langdetect(self, text: str) -> Dict:
        """Detect language using langdetect library"""
        try:
            # Get primary language
            primary_lang = detect(text)
            
            # Get all detected languages with confidence
            all_langs = detect_langs(text)
            
            # Convert to our format
            alternatives = []
            for lang_prob in all_langs:
                alternatives.append({
                    'language': lang_prob.lang,
                    'confidence': lang_prob.prob
                })
            
            return {
                'language': primary_lang,
                'confidence': alternatives[0]['confidence'] if alternatives else 0.0,
                'method': 'langdetect',
                'alternatives': alternatives[1:]  # Exclude primary
            }
            
        except LangDetectException as e:
            logger.warning(f"LangDetect failed: {e}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'method': 'langdetect_failed',
                'alternatives': []
            }
    
    def _detect_with_patterns(self, text: str) -> Dict:
        """Detect language using character pattern matching"""
        script_scores = {}
        total_chars = len(text)
        
        if total_chars == 0:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'method': 'pattern',
                'alternatives': []
            }
        
        # Count characters in each script
        for script, pattern in self.script_patterns.items():
            matches = len(re.findall(pattern, text))
            score = matches / total_chars
            script_scores[script] = score
        
        # Map scripts to languages
        script_to_lang = {
            'latin': 'en',  # Default to English for Latin script
            'devanagari': 'hi',
            'bengali': 'bn', 
            'chinese': 'zh',
            'arabic': 'ar',
            'cyrillic': 'ru',  # Default to Russian
            'japanese': 'ja',
            'korean': 'ko'
        }
        
        # Find dominant script
        dominant_script = max(script_scores.items(), key=lambda x: x[1])
        
        if dominant_script[1] > 0.1:  # At least 10% of characters
            detected_lang = script_to_lang.get(dominant_script[0], 'unknown')
            confidence = min(dominant_script[1] * 2, 1.0)  # Scale confidence
            
            # Create alternatives
            alternatives = []
            for script, score in script_scores.items():
                if script != dominant_script[0] and score > 0.05:
                    lang = script_to_lang.get(script, 'unknown')
                    alternatives.append({
                        'language': lang,
                        'confidence': score
                    })
            
            return {
                'language': detected_lang,
                'confidence': confidence,
                'method': 'pattern',
                'alternatives': alternatives
            }
        
        return {
            'language': 'unknown',
            'confidence': 0.0,
            'method': 'pattern',
            'alternatives': []
        }
    
    def _detect_hybrid(self, text: str) -> Dict:
        """Hybrid detection combining multiple methods"""
        results = []
        
        # Try langdetect
        try:
            langdetect_result = self._detect_with_langdetect(text)
            if langdetect_result['confidence'] > 0.3:
                results.append(langdetect_result)
        except Exception:
            pass
        
        # Try pattern matching
        try:
            pattern_result = self._detect_with_patterns(text)
            if pattern_result['confidence'] > 0.3:
                results.append(pattern_result)
        except Exception:
            pass
        
        if not results:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'method': 'hybrid',
                'alternatives': []
            }
        
        # Combine results
        if len(results) == 1:
            return results[0]
        
        # If both methods agree, use higher confidence
        if results[0]['language'] == results[1]['language']:
            best_result = max(results, key=lambda x: x['confidence'])
            return {
                **best_result,
                'method': 'hybrid_agreement'
            }
        
        # If methods disagree, use langdetect if confidence is high
        langdetect_result = next((r for r in results if r['method'] == 'langdetect'), None)
        if langdetect_result and langdetect_result['confidence'] > 0.7:
            return {
                **langdetect_result,
                'method': 'hybrid_langdetect_preferred'
            }
        
        # Otherwise use pattern matching
        pattern_result = next((r for r in results if r['method'] == 'pattern'), None)
        if pattern_result:
            return {
                **pattern_result,
                'method': 'hybrid_pattern_preferred'
            }
        
        return results[0]
    
    def detect_multiple_languages(self, text: str) -> List[Dict]:
        """
        Detect multiple languages in text (for mixed-language documents)
        
        Args:
            text: Input text
            
        Returns:
            List of language detection results for different segments
        """
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        language_segments = []
        
        current_segment = ""
        current_language = None
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Detect language of current sentence
            sent_detection = self.detect_language(sentence.strip())
            
            # If language changed, save previous segment
            if (current_language is not None and 
                sent_detection['language'] != current_language and
                current_segment.strip()):
                
                segment_detection = self.detect_language(current_segment.strip())
                language_segments.append({
                    'text': current_segment.strip(),
                    'language': segment_detection['language'],
                    'confidence': segment_detection['confidence'],
                    'start_pos': text.find(current_segment),
                    'end_pos': text.find(current_segment) + len(current_segment)
                })
                
                current_segment = sentence
            else:
                current_segment += " " + sentence if current_segment else sentence
            
            current_language = sent_detection['language']
        
        # Add final segment
        if current_segment.strip():
            segment_detection = self.detect_language(current_segment.strip())
            language_segments.append({
                'text': current_segment.strip(),
                'language': segment_detection['language'],
                'confidence': segment_detection['confidence'],
                'start_pos': text.find(current_segment),
                'end_pos': text.find(current_segment) + len(current_segment)
            })
        
        return language_segments
    
    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from code"""
        return self.supported_languages.get(lang_code, lang_code)
    
    def is_supported_language(self, lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in self.supported_languages
