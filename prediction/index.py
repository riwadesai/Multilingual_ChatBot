#!/usr/bin/env python3
"""
Prediction Module - Document Prediction Handler
This module handles query processing and document retrieval.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from prediction.document_prediction_handler import DocumentPredictionHandler


def main():
    """
    Main function to query documents.
    Takes event object and sends to document prediction handler.
    """
    # Event object with query and language
    event = {
        "query": "महाराष्ट्र की राजधानी और जिला?",
        "language": "hin"
    }
    
    # Check for Hugging Face API token (needed for Gemma-2B access)
    if not os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN") == "your_token_here":
        print("❌ Error: HUGGINGFACE_API_TOKEN not configured!")
        print("\n📋 Setup Instructions for Gemma-2B:")
        print("1. Go to: https://huggingface.co/google/gemma-2b")
        print("2. Log in and accept the license terms")
        print("3. Get your token from: https://huggingface.co/settings/tokens")
        print("4. Update the .env file with your actual token:")
        print("   HUGGINGFACE_API_TOKEN=your_actual_token_here")
        print("\nOr set it as an environment variable:")
        print("   export HUGGINGFACE_API_TOKEN=your_actual_token_here")
        return
    
    # Initialize prediction handler
    predictor = None
    try:
        predictor = DocumentPredictionHandler()
        
        # Send event to prediction handler
        result = predictor.predict(event)
        
        # Print results
        if result['success']:
            print("✅ Query processed successfully!")
            print(f"🔍 Query: {result['query']}")
            print(f"🌐 Language: {result['language']}")
            print(f"💬 Answer: {result['answer']}")
        else:
            print("❌ Query processing failed!")
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
    
    finally:
        # Clean up resources
        if predictor is not None:
            predictor.cleanup()


if __name__ == "__main__":
    main()