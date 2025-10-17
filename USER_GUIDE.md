# Multilingual ChatBot - User Guide

## Quick Start Guide

### 1. System Setup

#### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- Internet connection for model downloads
- Tesseract OCR with language packs

#### System Dependencies

**Install Tesseract OCR:**

**Basic Installation:**
- **Windows**: Download from [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

**Install All Languages (Recommended):**
- **Windows**: Download installer with all language packs from [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract --with-all-languages`
- **Linux (Ubuntu/Debian)**: `sudo apt-get install tesseract-ocr-all`
- **Linux (CentOS/RHEL/Fedora)**: `sudo dnf install tesseract-langpack-*`

**Verify Tesseract Installation:**
```bash
tesseract --list-langs
```

#### Python Dependencies Installation
```bash
# Clone the repository
git clone <repository-url>
cd Multilingual_ChatBot

# Install dependencies
pip install -r requirements.txt
```

### 2. Document Training (First Time Setup)

#### Step 1: Prepare Documents
```bash
# Place your documents in a directory
# Supported formats: PDF, TXT, DOCX, HTML
# Example: /path/to/your/document.pdf
```

#### Step 2: Configure Training
Edit `training/index.py`:
```python
event = {
    "document_path": "/path/to/your/document.pdf",
    "document_language": "hin"  # Language code
}
```

#### Step 3: Run Training
```bash
# Train documents
python training/index.py
```

#### Step 4: Verify Training Output
```bash
# Check generated files
ls -la data/processed/
# Should see: metadata_hin.json, faiss_index_hin.pkl
```

### 3. System Configuration

#### Set up environment
```bash
# Set up environment
export HUGGINGFACE_API_TOKEN=your_token_here
```

### 4. Getting Your Hugging Face Token

1. **Create Account**: Visit [https://huggingface.co/](https://huggingface.co/) and sign up
2. **Accept License**: Go to [https://huggingface.co/google/gemma-2b](https://huggingface.co/google/gemma-2b) and accept the license
3. **Generate Token**: Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. **Create Token**: Click "New token" → Select "Read" permissions → Copy token

### 5. Running the System

#### Basic Usage
```bash
# Set your token
export HUGGINGFACE_API_TOKEN=hf_your_token_here

# Run the system
python prediction/index.py
```

#### Custom Queries
Edit `prediction/index.py` to change the query:
```python
event = {
    "query": "Your question here",
    "language": "hin"  # hin, ben, chi_sim, en
}
```

### 5. System Operations

#### 5.1 Starting the System
```bash
# Method 1: Direct execution
python prediction/index.py

# Method 2: With environment file
echo "HUGGINGFACE_API_TOKEN=your_token" > .env
python prediction/index.py
```

#### 5.2 Monitoring Performance
The system provides detailed logging:
```
INFO: Loading model: google/gemma-2b
INFO: Reranking top 6 chunks for speed...
INFO: Chunk 1/6: Score 10
INFO: Reranking completed. Top 3 scores: [10, 10, 10]
```

#### 5.3 Stopping the System
- **Graceful Shutdown**: Ctrl+C (automatic cleanup)
- **Resource Cleanup**: Models are automatically unloaded
- **Memory Release**: All resources are freed

### 6. Maintenance Operations

#### 6.1 Clearing Model Cache
```bash
# Clear all cached models (frees ~10GB)
python -c "
import shutil
from pathlib import Path
cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print('Cache cleared!')
"
```

#### 6.2 Updating Models
```bash
# Force re-download of models
rm -rf ~/.cache/huggingface/hub/models--google--gemma-2b
python prediction/index.py
```

#### 6.3 Memory Management
```bash
# Check memory usage
python -c "
import psutil
print(f'RAM Usage: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / (1024**3):.1f} GB')
"
```

### 7. Troubleshooting

#### 7.1 Common Issues

**Issue**: `401 Client Error: Not Found`
```bash
# Solution: Check your token
echo $HUGGINGFACE_API_TOKEN
# Should start with 'hf_'
```

**Issue**: `zsh: killed python`
```bash
# Solution: Insufficient memory
# Option 1: Use smaller model
# Option 2: Increase system RAM
# Option 3: Close other applications
```

**Issue**: `Cannot use chat template functions`
```bash
# Solution: Already fixed in current version
# The system automatically handles models without chat templates
```

#### 7.2 Performance Issues

**Slow Reranking**:
- The system processes only top 6 chunks for speed
- Each chunk takes ~5 seconds on CPU
- Total reranking time: ~30 seconds

**Memory Issues**:
- Gemma-2B requires ~4GB RAM
- Close other applications
- Consider using a smaller model

### 8. Advanced Configuration

#### 8.1 Custom Model Selection
Edit `prediction/document_prediction_handler.py`:
```python
# For faster processing (smaller model)
self.model_name = "microsoft/DialoGPT-small"

# For better quality (larger model)
self.model_name = "google/gemma-2b"
```

#### 8.2 Search Parameters
```python
# MMR search lambda (0.0 = pure relevance, 1.0 = pure diversity)
mmr_lambda = 0.7

# Number of chunks to rerank
chunks_to_process = 6

# Final results returned
top_k_results = 3
```

#### 8.3 Performance Tuning
```python
# Reduce context length for speed
content[:300]  # Instead of content[:1000]

# Reduce max tokens for faster generation
max_new_tokens=3  # Instead of 10

# Lower temperature for consistent results
temperature=0.1
```

### 9. System Monitoring

#### 9.1 Log Analysis
```bash
# View system logs
python prediction/index.py 2>&1 | tee system.log

# Monitor memory usage
watch -n 1 'ps aux | grep python'
```

### 10. Backup and Recovery

#### 10.1 Data Backup
```bash
# Backup processed data
cp -r data/processed/ backup/processed_$(date +%Y%m%d)/

# Backup configuration
cp .env backup/env_$(date +%Y%m%d)
```

#### 10.2 System Recovery
```bash
# Restore from backup
cp -r backup/processed_YYYYMMDD/* data/processed/

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### 11. Support and Maintenance

#### 11.1 Regular Maintenance
- **Weekly**: Clear model cache if storage is low
- **Monthly**: Update dependencies
- **Quarterly**: Review and update models

#### 11.2 System Health Checks
```bash
# Check system status
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
"
```

#### 11.3 Performance Optimization
- **Use SSD storage** for faster model loading
- **Increase RAM** for better performance
- **Use GPU** if available (requires CUDA setup)
- **Close unnecessary applications** during inference

---

## Quick Reference

### Essential Commands
```bash
# Start system
export HUGGINGFACE_API_TOKEN=your_token && python prediction/index.py

# Clear cache
rm -rf ~/.cache/huggingface/hub/

# Check memory
python -c "import psutil; print(f'{psutil.virtual_memory().percent}%')"

# Monitor processes
ps aux | grep python
```

### Configuration Files
- **`.env`**: Environment variables
- **`requirements.txt`**: Python dependencies
- **`prediction/index.py`**: Main entry point
- **`prediction/document_prediction_handler.py`**: Core logic

### Key Directories
- **`data/processed/`**: FAISS indices and metadata
- **`~/.cache/huggingface/hub/`**: Model cache
- **`prediction/`**: Application code
- **`training/`**: Training scripts
