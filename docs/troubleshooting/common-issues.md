# Common Issues and Solutions

This guide covers the most frequently encountered issues when using Sifaka and provides step-by-step solutions.

## Quick Diagnostics

Run this diagnostic script to check your Sifaka setup:

```python
#!/usr/bin/env python3
"""Sifaka diagnostic script."""

import os
import sys
import platform

def run_diagnostics():
    """Run comprehensive Sifaka diagnostics."""
    print("🔍 Sifaka Diagnostics")
    print("=" * 50)

    # System information
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    # Check Sifaka installation
    try:
        import sifaka
        print(f"✅ Sifaka: {sifaka.__version__}")
    except ImportError as e:
        print(f"❌ Sifaka not installed: {e}")
        return

    # Check API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "HuggingFace": os.getenv("HUGGINGFACE_API_KEY")
    }

    print("\n🔑 API Keys:")
    for service, key in api_keys.items():
        status = "✅ Set" if key else "❌ Missing"
        print(f"  {service}: {status}")

    # Test basic functionality
    print("\n🧪 Basic Tests:")
    try:
        from sifaka.core.thought import Thought
        thought = Thought(prompt="Test")
        print("✅ Core imports working")
    except Exception as e:
        print(f"❌ Core import failed: {e}")

    # Test model creation
    try:
        from sifaka.models import create_model
        model = create_model("mock:test-model")
        print("✅ Model creation working")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")

    print("\n" + "=" * 50)
    print("Diagnostics complete!")

if __name__ == "__main__":
    run_diagnostics()
```

## Installation Issues

### Problem: `pip install sifaka` fails

**Symptoms:**
- Package not found errors
- Permission denied errors
- Dependency conflicts

**Solutions:**

```bash
# 1. Update pip and try again
python -m pip install --upgrade pip
pip install sifaka

# 2. Use virtual environment
python -m venv sifaka-env
source sifaka-env/bin/activate  # Linux/Mac
# OR
sifaka-env\Scripts\activate     # Windows
pip install sifaka

# 3. Install with specific Python version
python3.11 -m pip install sifaka

# 4. Install from source (if package issues)
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka
pip install -e .
```

### Problem: Python version compatibility

**Symptoms:**
- "Python 3.11+ required" errors
- Syntax errors on import

**Solutions:**

```bash
# Check Python version
python --version

# Install Python 3.11+ if needed
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.11

# On macOS with Homebrew:
brew install python@3.11

# On Windows: Download from python.org

# Use uv for Python version management
uv python install 3.11
uv python pin 3.11
```

## API Key Issues

### Problem: Authentication errors

**Symptoms:**
- `AuthenticationError` or `PermissionError`
- "Invalid API key" messages
- 401 Unauthorized responses

**Solutions:**

```python
# 1. Check if API keys are set
import os

def check_api_keys():
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
    }

    for name, key in keys.items():
        if key:
            print(f"✅ {name}: Set (length: {len(key)})")
        else:
            print(f"❌ {name}: Not set")

check_api_keys()

# 2. Set API keys properly
# In .env file:
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Or in shell:
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"

# 3. Test API key validity
from sifaka.models import create_model

try:
    model = create_model("openai:gpt-3.5-turbo")
    result = model.generate("Hello", max_tokens=5)
    print("✅ API key valid")
except Exception as e:
    print(f"❌ API key invalid: {e}")
```

### Problem: Rate limiting

**Symptoms:**
- `RateLimitError` or 429 status codes
- "Too many requests" messages

**Solutions:**

```python
# 1. Add retry logic with backoff
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def generate_with_retry(model, prompt):
    return model.generate(prompt)

# 2. Reduce request frequency
import asyncio

async def batch_with_delay(prompts, model, delay=1.0):
    results = []
    for prompt in prompts:
        result = model.generate(prompt)
        results.append(result)
        await asyncio.sleep(delay)  # Add delay between requests
    return results

# 3. Use different models or tiers
# Switch to higher tier or different provider
model = create_model("anthropic:claude-3-haiku")  # Often has higher limits
```

## Model Issues

### Problem: Model not found or unsupported

**Symptoms:**
- "Model not found" errors
- "Unsupported model" messages

**Solutions:**

```python
# 1. Check available models
from sifaka.models import create_model

# Supported model formats:
models = [
    "openai:gpt-4",
    "openai:gpt-3.5-turbo",
    "anthropic:claude-3-sonnet",
    "anthropic:claude-3-haiku",
    "gemini:gemini-1.5-flash",
    "gemini:gemini-1.5-pro",
    "mock:test-model"  # For testing
]

for model_spec in models:
    try:
        model = create_model(model_spec)
        print(f"✅ {model_spec}: Available")
    except Exception as e:
        print(f"❌ {model_spec}: {e}")

# 2. Use mock model for testing
model = create_model("mock:test-model")  # Always works, no API key needed
```

### Problem: Model timeout or hanging

**Symptoms:**
- Requests hang indefinitely
- Timeout errors after long waits

**Solutions:**

```python
# 1. Set explicit timeouts
model = create_model("openai:gpt-4", timeout=30)  # 30 second timeout

# 2. Use shorter max_tokens
model = create_model("openai:gpt-4", max_tokens=500)  # Faster responses

# 3. Check network connectivity
import requests

def test_connectivity():
    try:
        response = requests.get("https://api.openai.com", timeout=5)
        print(f"✅ OpenAI reachable: {response.status_code}")
    except Exception as e:
        print(f"❌ OpenAI unreachable: {e}")

    try:
        response = requests.get("https://api.anthropic.com", timeout=5)
        print(f"✅ Anthropic reachable: {response.status_code}")
    except Exception as e:
        print(f"❌ Anthropic unreachable: {e}")

test_connectivity()
```

## Storage Issues

> **⚠️ MCP Storage Status**: Redis and Milvus storage backends via MCP are currently experiencing issues and are being actively fixed. For production use, we recommend Memory or File storage until MCP integration is restored.

### Problem: Redis connection failed

**Symptoms:**
- `ConnectionError` to Redis
- "Connection refused" errors

**Solutions:**

```bash
# 1. Check if Redis is running
redis-cli ping  # Should return "PONG"

# 2. Start Redis if not running
# Using Docker:
docker run -d -p 6379:6379 redis:alpine

# Using system package:
sudo systemctl start redis  # Linux
brew services start redis   # macOS

# 3. Check Redis configuration
redis-cli info server
```

```python
# 4. Test Redis connection in Python
import redis

def test_redis():
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connection successful")

        # Test basic operations
        r.set("test_key", "test_value")
        value = r.get("test_key")
        print(f"✅ Redis operations working: {value}")

    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Solutions:")
        print("  1. Start Redis: docker run -d -p 6379:6379 redis:alpine")
        print("  2. Check firewall settings")
        print("  3. Verify Redis configuration")

test_redis()
```

### Problem: Milvus connection issues

**Symptoms:**
- Milvus connection errors
- Collection not found errors

**Solutions:**

```bash
# 1. Start Milvus using installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start

# 2. Check Milvus status
curl http://localhost:9091/health

# 3. Check Milvus logs
docker logs milvus
```

```python
# 4. Test Milvus connection
def test_milvus():
    try:
        from pymilvus import connections, utility

        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus connection successful")

        # List collections
        collections = utility.list_collections()
        print(f"✅ Collections: {collections}")

    except ImportError:
        print("❌ pymilvus not installed: pip install pymilvus")
    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")
        print("Solutions:")
        print("  1. Start Milvus: docker-compose up -d")
        print("  2. Check port 19530 is accessible")
        print("  3. Wait for Milvus to fully start (can take 1-2 minutes)")

test_milvus()
```

## Chain Execution Issues

### Problem: Chain fails to run

**Symptoms:**
- Exceptions during `chain.run()`
- Validation failures
- Empty or None results

**Solutions:**

```python
# 1. Debug chain step by step
from sifaka import Chain
from sifaka.models import create_model

def debug_chain():
    try:
        # Test model creation
        model = create_model("mock:test-model")
        print("✅ Model created")

        # Test chain creation
        chain = Chain(model=model, prompt="Test prompt")
        print("✅ Chain created")

        # Test chain execution
        result = chain.run()
        print(f"✅ Chain executed: {result.text[:50]}...")

    except Exception as e:
        print(f"❌ Chain failed: {e}")
        import traceback
        traceback.print_exc()

debug_chain()

# 2. Check validation issues
def debug_validation():
    from sifaka.validators import LengthValidator

    validator = LengthValidator(min_length=10, max_length=100)
    chain = Chain(model=model, prompt="Test")
    chain.validate_with(validator)

    result = chain.run()

    # Check validation results
    for name, validation in result.validation_results.items():
        if validation.passed:
            print(f"✅ {name}: Passed")
        else:
            print(f"❌ {name}: Failed - {validation.message}")
            print(f"   Issues: {validation.issues}")
            print(f"   Suggestions: {validation.suggestions}")

debug_validation()
```

### Problem: Memory issues with large chains

**Symptoms:**
- Out of memory errors
- Slow performance
- System freezing

**Solutions:**

```python
# 1. Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# 2. Clear thought history for long chains
def memory_efficient_chain():
    chain = Chain(model=model, prompt="Your prompt")
    result = chain.run()

    # Clear history to save memory
    result.history = None

    return result

# 3. Use generators for batch processing
def process_batch_efficiently(prompts):
    for prompt in prompts:
        chain = Chain(model=model, prompt=prompt)
        result = chain.run()

        # Yield result and clear memory
        yield result
        del chain, result

        # Force garbage collection periodically
        if len(prompts) % 10 == 0:
            import gc
            gc.collect()
```

## Import and Dependency Issues

### Problem: Import errors

**Symptoms:**
- `ModuleNotFoundError`
- `ImportError` messages

**Solutions:**

```python
# 1. Check what's installed
def check_installation():
    packages = [
        "sifaka",
        "openai",
        "anthropic",
        "redis",
        "pymilvus"
    ]

    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Not installed")
            print(f"   Install with: pip install {package}")

check_installation()

# 2. Install missing dependencies
# pip install sifaka[all]  # Install everything
# pip install sifaka[openai]  # Just OpenAI
# pip install sifaka[anthropic]  # Just Anthropic
```

## Getting Help

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or just for Sifaka
from sifaka.utils.logging import get_logger
logger = get_logger("sifaka")
logger.setLevel(logging.DEBUG)
```

### Collect System Information

```python
def collect_system_info():
    """Collect system information for bug reports."""
    import sys
    import platform
    import sifaka

    info = {
        "Python": sys.version,
        "Platform": platform.platform(),
        "Sifaka": sifaka.__version__,
        "Environment": dict(os.environ)
    }

    # Remove sensitive information
    sensitive_keys = [k for k in info["Environment"] if "KEY" in k or "TOKEN" in k]
    for key in sensitive_keys:
        info["Environment"][key] = "***REDACTED***"

    return info

# Use when reporting issues
system_info = collect_system_info()
print("System Information:")
for key, value in system_info.items():
    if key != "Environment":
        print(f"  {key}: {value}")
```

### When to Report Issues

Report issues on GitHub when you encounter:
- Bugs in core functionality
- Documentation errors
- Performance problems
- Feature requests

Include:
1. **Error message and full traceback**
2. **System information** (from `collect_system_info()`)
3. **Minimal code example** to reproduce the issue
4. **Expected vs actual behavior**
5. **Debug logs** (if applicable)

For more specific troubleshooting:
- **[Import problems](import-problems.md)** - Detailed import troubleshooting
- **[Configuration errors](configuration-errors.md)** - Configuration-specific issues
- **[Performance tuning](../guides/performance-tuning.md)** - Optimization strategies
