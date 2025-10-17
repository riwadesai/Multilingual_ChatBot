# Multilingual ChatBot System

A sophisticated Retrieval-Augmented Generation (RAG) system supporting multiple languages with advanced document search and AI-powered answer generation.

## 🚀 Quick Start

### 1. Train Documents (First Time Setup)
```bash
# Train documents for the RAG system
python training/index.py
```

### 2. Run the ChatBot
```bash
# 1. Set up your Hugging Face token
export HUGGINGFACE_API_TOKEN=hf_your_token_here

# 2. Run the system
python prediction/index.py
```

## 📋 System Overview

### Supported Languages
- 🇮🇳 **Hindi** (hin) - Active
- 🇧🇩 **Bengali** (ben) - Active  
- 🇨🇳 **Chinese** (chi_sim) - Active
- 🇺🇸 **English** (en) - Active

### Core Features
- **Hybrid Search**: Vector similarity + keyword matching
- **LLM Reranking**: Gemma-2B powered relevance scoring
- **Multilingual Embeddings**: BAAI/bge-m3 model
- **Resource Management**: Automatic cleanup and memory optimization

## 🏗️ Architecture

```
Query → Embedding → Vector Search → MMR → Keyword Search
  ↓
Deduplication → LLM Reranking → Answer Generation → Response
```


## 📚 Documentation

- **[Technical Documentation](TECHNICAL_DOCUMENTATION.md)** - System architecture and components
- **[User Guide](USER_GUIDE.md)** - Operating and maintenance instructions  
- **[Training Guide](TRAINING_GUIDE.md)** - Document training and processing
- **[Training Workflow](TRAINING_WORKFLOW.md)** - Complete training process
- **[Performance Report](PERFORMANCE_REPORT.md)** - Detailed performance analysis

## 🔧 Requirements

- **Python**: 3.8+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB for models and data
- **Internet**: For model downloads

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Documents (First Time)
```bash
# Edit training/index.py to specify your document
# Then run training
python training/index.py
```

### 3. Get Hugging Face Token
1. Visit [https://huggingface.co/google/gemma-2b](https://huggingface.co/google/gemma-2b)
2. Log in and accept the license
3. Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Run the System
```bash
export HUGGINGFACE_API_TOKEN=hf_your_token_here
python prediction/index.py
```

## 🛠️ Customization

### Change Query
Edit `prediction/index.py`:
```python
event = {
    "query": "Your question here",
    "language": "hin"  # hin, ben, chi_sim, en
}
```

### Optimize Performance
- **Faster Model**: Use `microsoft/DialoGPT-small`
- **Better Quality**: Use `google/gemma-2b` (default)
- **Memory**: Clear cache with `rm -rf ~/.cache/huggingface/hub/`

## 📈 System Status

### ✅ Working Components
- Document retrieval and search
- Multilingual embedding generation
- FAISS vector search with MMR
- Keyword-based search
- LLM-powered reranking
- Answer generation
- Resource cleanup

### ⚠️ Performance Notes
- **Reranking**: Takes ~30s (CPU processing)
- **Memory**: Requires 6GB RAM
- **Models**: First run downloads ~3GB

## 🔍 Troubleshooting

### Common Issues

**Memory Error**: `zsh: killed python`
```bash
# Solution: Free up memory or use smaller model
# Option 1: Close other applications
# Option 2: Use DialoGPT-small instead of Gemma-2B
```

**Authentication Error**: `401 Client Error`
```bash
# Solution: Check your token
echo $HUGGINGFACE_API_TOKEN
# Should start with 'hf_'
```

**Slow Performance**: 
- Reranking is CPU-intensive (~30s)
- Consider GPU acceleration for faster processing
- Reduce chunk processing from 6 to 3 for speed

## 📁 Project Structure

```
Multilingual_ChatBot/
├── prediction/           # Main application
│   ├── index.py         # Entry point
│   └── document_prediction_handler.py  # Core logic
├── data/processed/      # FAISS indices and metadata
├── training/           # Training scripts
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🎯 Key Features

### 1. Advanced Search Pipeline
- **Semantic Search**: FAISS vector similarity
- **MMR Search**: Diversity-aware result selection
- **Keyword Search**: TF-IDF based matching
- **Deduplication**: Content-based duplicate removal

### 2. LLM Integration
- **Model**: Google Gemma-2B (3B parameters)
- **Reranking**: Relevance scoring for search results
- **Answer Generation**: Context-aware responses
- **Chat Templates**: Automatic message formatting

### 3. Resource Management
- **Singleton Pattern**: Single model instance
- **Automatic Cleanup**: Memory deallocation
- **Error Handling**: Graceful degradation
- **Logging**: Comprehensive system monitoring

## 📊 Performance Optimization

### Speed Improvements
- ✅ Reduced chunk processing (6 vs 12)
- ✅ Shorter context (300 vs 1000 chars)
- ✅ Minimal token generation (3 vs 10)
- ✅ Batch processing optimization

### Memory Optimization
- ✅ Model caching and reuse
- ✅ Automatic resource cleanup
- ✅ Efficient tensor operations
- ✅ Singleton pattern implementation

