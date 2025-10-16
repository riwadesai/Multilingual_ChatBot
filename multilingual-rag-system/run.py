"""
Run script for Multilingual RAG System
Simplified way to start the application
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit application"""
    
    # Check if setup has been run
    if not Path("models").exists():
        print("❌ Setup not completed. Please run 'python setup.py' first.")
        return
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_CACHE"] = str(Path("models").absolute())
    os.environ["HF_HOME"] = str(Path("models").absolute())
    
    # Run Streamlit
    try:
        print("🚀 Starting Multilingual RAG System...")
        print("📱 Open your browser to http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
