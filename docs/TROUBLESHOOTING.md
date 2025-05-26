# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using Sifaka.

## Quick Diagnostics

### Check Your Environment
```bash
# Verify Python version (3.11+ required)
python --version

# Check installed packages
pip list | grep -E "(sifaka|anthropic|openai|redis|milvus)"

# Test basic imports
python -c "import sifaka; print('✅ Sifaka imported successfully')"
```

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use Sifaka's logger
from sifaka.utils.logging import get_logger
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
```

## Common Issues

### 1. Model Connection Issues

#### Symptoms
- `ConnectionError` or `NetworkError`
- `TimeoutError` during model calls
- `AuthenticationError` or `PermissionError`

#### Solutions
```python
# Check API keys
import os
print("OpenAI API Key:", "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Missing")
print("Anthropic API Key:", "✅ Set" if os.getenv("ANTHROPIC_API_KEY") else "❌ Missing")

# Test model connectivity
from sifaka.models.openai import OpenAIModel
try:
    model = OpenAIModel("gpt-3.5-turbo")
    result = model.generate("Hello", max_tokens=5)
    print("✅ Model connection successful")
except Exception as e:
    print(f"❌ Model connection failed: {e}")
```

#### Common Causes
- Missing or invalid API keys
- Network connectivity issues
- Rate limiting or quota exceeded
- Incorrect model names

### 2. Storage Connection Issues

#### Redis Connection Problems
```python
# Test Redis connection
import redis
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("✅ Redis connection successful")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
```

**Solutions:**
- Ensure Redis is running: `redis-server`
- Check connection parameters (host, port, password)
- Verify firewall settings

#### Milvus Connection Problems
```python
# Test Milvus connection (if using)
try:
    from pymilvus import connections
    connections.connect("default", host="localhost", port="19530")
    print("✅ Milvus connection successful")
except Exception as e:
    print(f"❌ Milvus connection failed: {e}")
```

**Solutions:**
- Ensure Milvus is running: `docker-compose up -d`
- Check connection parameters
- Verify collection exists

### 3. Import and Dependency Issues

#### Missing Dependencies
```bash
# Install core dependencies
pip install sifaka

# Install optional dependencies
pip install "sifaka[redis]"     # For Redis storage
pip install "sifaka[milvus]"    # For Milvus storage
pip install "sifaka[all]"       # For all optional features
```

#### Version Conflicts
```bash
# Check for conflicts
pip check

# Upgrade packages
pip install --upgrade sifaka anthropic openai
```

### 4. Memory and Performance Issues

#### GPU Memory Issues (Local Models)
```python
# Clear GPU cache
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory}")
```

**Solutions:**
- Use model quantization (4bit/8bit)
- Reduce batch size or sequence length
- Use smaller model variants
- Enable gradient checkpointing

#### High Memory Usage
```python
# Monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Current memory usage: {memory_mb:.1f} MB")
```

### 5. Configuration Issues

#### Environment Variables
```bash
# Check environment variables
env | grep -E "(OPENAI|ANTHROPIC|REDIS|MILVUS)"

# Set environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

#### Configuration Files
```python
# Validate configuration
from sifaka import QuickStart

try:
    chain = QuickStart.basic_chain("openai:gpt-3.5-turbo", "Test prompt")
    print("✅ Configuration valid")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

## Error-Specific Solutions

### `ModuleNotFoundError`
1. Install missing package: `pip install <package-name>`
2. Check virtual environment activation
3. Verify Python path and PYTHONPATH

### `JSONDecodeError`
1. Check API response format
2. Verify network connectivity
3. Check for rate limiting responses

### `ValidationError`
1. Check input format and encoding
2. Verify validation rules
3. Ensure required fields are present

### `TimeoutError`
1. Increase timeout values
2. Check network latency
3. Verify service availability

## Getting Help

### Enable Verbose Logging
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Collect System Information
```python
import sys
import platform
import sifaka

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Sifaka: {sifaka.__version__}")
```

### Report Issues
When reporting issues, please include:
1. Error message and full traceback
2. System information (Python version, OS)
3. Sifaka version and configuration
4. Minimal code example to reproduce
5. Debug logs (if applicable)

## Performance Optimization

### Model Selection
- Use smaller models for faster responses
- Consider local models for privacy/cost
- Use appropriate model for task complexity

### Storage Optimization
- Use memory storage for temporary data
- Use Redis for shared/persistent data
- Use Milvus for vector similarity search

### Caching
```python
# Enable caching for better performance
from sifaka.storage.cached import CachedStorage
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.redis import RedisStorage

cache = MemoryStorage()
persistence = RedisStorage()
storage = CachedStorage(cache=cache, persistence=persistence)
```

## Advanced Debugging

### Network Debugging
```bash
# Test connectivity
ping api.openai.com
curl -I https://api.anthropic.com

# Check DNS resolution
nslookup api.openai.com
```

### Process Monitoring
```bash
# Monitor system resources
top -p $(pgrep -f python)
htop

# Check network connections
netstat -an | grep :6379  # Redis
netstat -an | grep :19530 # Milvus
```

For more specific issues, consult the [API Reference](API_REFERENCE.md) or [Architecture Documentation](ARCHITECTURE.md).
