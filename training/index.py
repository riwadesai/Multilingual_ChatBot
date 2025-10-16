#!/usr/bin/env python3
"""
Training Module - Document Training Handler
This module handles document training, parsing, chunking, and embedding.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.document_training_handler import DocumentTrainingHandler


def main():
    """
    Main function to train documents.
    Takes event object and sends to document training handler.
    language code. Supported languages:
                     - eng: English
                     - ben: Bengali
                     - hin: Hindi
                     - urd: Urdu
                     - chi_sim: Chinese Simplified
                     - chi_tra: Chinese Traditional
                     - ara: Arabic
                     - guj: Gujarati
                     - pan: Punjabi
                     - tam: Tamil
                     - tel: Telugu
                     - kan: Kannada
                     - mal: Malayalam
                     - ori: Odia
                     - asm: Assamese
                     - fra: French
                     - deu: German
                     - spa: Spanish
                     - por: Portuguese
                     - rus: Russian
                     - jpn: Japanese
                     - kor: Korean
                     - tha: Thai
                     - vie: Vietnamese
    """
    # Event object with document path and language
    event = {
        "document_path": '/Users/riwadesai/Downloads/Hindi_article_for_Mumbai,_mobile_PDF.pdf',
        "document_language": "hin"
    }
    
    # Initialize training handler
    trainer = DocumentTrainingHandler()
    
    # Send event to training handler
    result = trainer.train_document(event)
    
    # Print results
    if result['success']:
        print("✅ Document training completed successfully!")
        print(f"📁 Metadata saved to: {result['metadata_path']}")
        print(f"📁 FAISS index saved to: {result['faiss_path']}")
        print(f"📊 Total chunks processed: {result['chunk_count']}")
    else:
        print("❌ Document training failed!")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()