# Multilingual RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built for the 72-hour assignment, supporting multiple languages and using free Hugging Face models.

## 🚀 Features

- **Multilingual Support**: English, Hindi, Bengali, Chinese, Spanish, French, German
- **PDF Processing**: Digital text extraction + OCR for scanned documents
- **Free Models**: Uses Hugging Face models (no API costs)
- **Advanced RAG**: Retrieval, reranking, and generation pipeline
- **Streamlit UI**: User-friendly web interface
- **FAISS Vector Store**: Efficient similarity search
- **Language Detection**: Automatic language identification

## 🏗️ Architecture

```
multilingual-rag-system/
├── src/
│   ├── document_processor/     # PDF processing & OCR
│   ├── embeddings/            # Embedding generation & reranking
│   ├── vector_store/          # FAISS vector database
│   ├── rag_pipeline/          # RAG orchestration
│   └── ui/                    # Streamlit interface
├── data/                      # Document storage
├── models/                    # Model cache
├── tests/                     # Test files
├── main.py                    # Application entry point
├── setup.py                   # Setup script
└── requirements.txt          # Dependencies
```

## 🛠️ Tech Stack

- **PDF Processing**: PyMuPDF, EasyOCR
- **Embeddings**: sentence-transformers
- **LLM**: Microsoft Phi-3-mini-4k-instruct
- **Vector DB**: FAISS
- **Reranking**: Cross-encoder models
- **UI**: Streamlit
- **Framework**: LangChain

## 📦 Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd multilingual-rag-system
```

### 2. Run Setup Script

```bash
python setup.py
```

This will:
- Install all dependencies
- Download required models
- Create necessary directories
- Set up environment variables

### 3. Manual Installation (Alternative)

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/{pdfs,processed} models tests

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=./models
export HF_HOME=./models
```

## 🚀 Usage

### Start the Application

```bash
streamlit run main.py
```

### Upload Documents

1. Go to the "Upload" tab
2. Upload PDF files (supports multiple languages)
3. Click "Process Documents"
4. Wait for processing to complete

### Ask Questions

1. Go to the "Chat" tab
2. Select your language
3. Type your question
4. Get AI-powered answers with sources

## 🔧 Configuration

### Model Settings

- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM Model**: `microsoft/Phi-3-mini-4k-instruct`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Supported Languages

| Code | Language | OCR Support |
|------|----------|-------------|
| en   | English  | ✅ |
| hi   | Hindi    | ✅ |
| bn   | Bengali  | ✅ |
| zh   | Chinese  | ✅ |
| es   | Spanish  | ✅ |
| fr   | French   | ✅ |
| de   | German   | ✅ |

## 📊 Performance

### Model Sizes
- **Embedding Model**: ~420MB
- **LLM Model**: ~2.7B parameters
- **Reranker**: ~80MB

### Performance Metrics
- **Retrieval Speed**: ~100ms per query
- **Generation Speed**: ~2-5s per response
- **Memory Usage**: ~4-6GB RAM
- **Storage**: ~1GB for models

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_pdf_processor.py
```

## 📈 Evaluation

The system includes evaluation capabilities:

- **Retrieval Metrics**: Precision@K, Recall@K
- **Answer Quality**: RAGAS framework
- **Latency**: Query processing time
- **Multilingual**: Cross-language evaluation

## 🔍 API Usage

```python
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorManager
from src.embeddings import EmbeddingGenerator

# Initialize components
embedding_model = EmbeddingGenerator()
vector_manager = VectorManager(embedding_model)
llm_manager = LLMManager()

# Create RAG pipeline
rag = RAGPipeline(vector_manager, embedding_model, llm_manager)

# Answer questions
result = rag.answer_question(
    question="What is machine learning?",
    language="en"
)

print(result['answer'])
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
   ```

2. **Model Download Fails**
   ```bash
   # Clear cache and retry
   rm -rf models/
   python setup.py
   ```

3. **PDF Processing Issues**
   - Ensure PDFs are not password-protected
   - Check file permissions
   - Try with smaller files first

### Logs

Check logs in the `logs/` directory for detailed error information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face** for providing free models
- **LangChain** for the RAG framework
- **Streamlit** for the UI framework
- **Sentence Transformers** for embeddings

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

**Built for the 72-hour RAG assignment with ❤️**
