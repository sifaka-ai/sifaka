# Sifaka Troubleshooting Guide

This guide helps you diagnose and fix common issues with Sifaka.

## 🚨 Common Issues

### 1. API Key Errors

**Error**: `Missing or invalid OpenAI API key`

**Symptoms**:
```
ConfigurationError: API key error: Missing or invalid OpenAI API key
Suggestions:
  - Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'
  - Get an API key from openai.com
  - Verify your API key is valid and has necessary permissions
```

**Solutions**:
```bash
# Set your API key
export OPENAI_API_KEY="sk-your-actual-key-here"

# Verify it's set
echo $OPENAI_API_KEY

# For persistent setup, add to ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
```

**Test your setup**:
```python
import os
print(f"API key set: {'OPENAI_API_KEY' in os.environ}")
print(f"Key starts with: {os.environ.get('OPENAI_API_KEY', '')[:7]}...")
```

### 2. Model Connection Issues

**Error**: `Failed to connect to model 'openai:gpt-4'`

**Common Causes**:
- Network connectivity issues
- API service outages
- Rate limiting
- Invalid model names

**Solutions**:
```python
# Try a different model
result = await sifaka.improve(
    "Your prompt",
    model="openai:gpt-4o-mini"  # Faster, more available
)

# Check model availability
from openai import OpenAI
client = OpenAI()
models = client.models.list()
print([m.id for m in models.data if 'gpt' in m.id])
```

### 3. Rate Limiting

**Error**: `Rate limit exceeded for model 'openai:gpt-4'`

**Solutions**:
```python
import asyncio

# Add retry logic with exponential backoff
async def improve_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await sifaka.improve(prompt)
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
```

### 4. Validation Never Passes

**Error**: Iterations hit max limit but validation still fails

**Diagnosis**:
```python
result = await sifaka.improve("Your prompt", max_rounds=5)

# Check what's failing
print(f"Final validation passed: {result.validation_passed()}")
print(f"Iterations used: {result.iteration}")

# Examine validation details
for validation in result.validations:
    if not validation.passed:
        print(f"Failed: {validation.validator}")
        print(f"Message: {validation.message}")
        print(f"Suggestions: {validation.suggestions}")
```

**Solutions**:
```python
# Adjust validation criteria
config = SifakaConfig(
    min_length=50,      # Was too high?
    max_length=1000,    # Was too low?
    max_iterations=5    # Give more attempts
)

# Or use more lenient validation
from sifaka.validators import LengthValidator
validator = LengthValidator(
    min_length=50,
    max_length=2000,    # More generous limit
    unit="words"        # Maybe chars was too restrictive
)
```

### 5. Slow Performance

**Issue**: Sifaka takes too long to complete

**Diagnosis**:
```python
import time
from sifaka.utils.thought_inspector import get_thought_overview

start = time.time()
result = await sifaka.improve("Your prompt")
end = time.time()

print(f"Total time: {end - start:.2f}s")
overview = get_thought_overview(result)
print(f"Iterations: {overview['total_iterations']}")
print(f"Model calls: {overview['total_model_calls']}")
```

**Solutions**:
```python
# Use faster models
result = await sifaka.improve(
    "Your prompt",
    model="groq:llama-3.1-8b-instant",  # Much faster
    max_rounds=2                         # Fewer iterations
)

# Or optimize configuration
config = SifakaConfig(
    model="openai:gpt-4o-mini",  # Faster than GPT-4
    max_iterations=2,            # Limit iterations
    critics=["reflexion"]        # Single critic
)
```

## 🔍 Debugging Tools

### 1. Thought Inspector

Use the built-in debugging tools to understand what's happening:

```python
from sifaka.utils.thought_inspector import (
    print_iteration_details,
    print_all_iterations,
    print_critic_summary,
    print_validation_summary
)

result = await sifaka.improve("Your prompt")

# See what happened in the latest iteration
print_iteration_details(result)

# See all iterations
print_all_iterations(result)

# See critic feedback
print_critic_summary(result)

# See validation results
print_validation_summary(result)
```

### 2. Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("sifaka")
logger.setLevel(logging.DEBUG)

# Now run your code - you'll see detailed logs
result = await sifaka.improve("Your prompt")
```

### 3. Conversation History

Examine the actual model conversations:

```python
from sifaka.utils.thought_inspector import print_conversation_messages

result = await sifaka.improve("Your prompt")

# See the actual conversations with the model
print_conversation_messages(result, full_messages=True)
```

## ⚠️ Error Patterns

### Pattern 1: Import Errors

**Error**: `ModuleNotFoundError: No module named 'sifaka'`

**Solution**:
```bash
# Install Sifaka
pip install sifaka

# Or with specific extras
pip install sifaka[openai,anthropic]

# Verify installation
python -c "import sifaka; print('Sifaka installed successfully')"
```

### Pattern 2: Async/Await Issues

**Error**: `RuntimeError: asyncio.run() cannot be called from a running event loop`

**Solution**:
```python
# In Jupyter notebooks or async environments
import asyncio

# Don't use asyncio.run() - just await directly
result = await sifaka.improve("Your prompt")

# Or create a new event loop
def sync_improve(prompt):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(sifaka.improve(prompt))
    finally:
        loop.close()
```

### Pattern 3: Configuration Errors

**Error**: `ConfigurationError: max_iterations must be positive`

**Solution**:
```python
# Check your configuration values
config = SifakaConfig(
    max_iterations=3,    # Must be > 0
    min_length=50,       # Must be >= 0
    max_length=1000      # Must be >= min_length
)

# Validate before using
print(f"Config valid: {config.max_iterations > 0}")
```

## 🛠️ Advanced Debugging

### Custom Error Handling

```python
from sifaka.utils.errors import (
    ValidationError,
    CritiqueError,
    ConfigurationError
)

async def robust_improve(prompt):
    try:
        return await sifaka.improve(prompt)
    except ValidationError as e:
        print(f"Validation failed: {e}")
        print(f"Suggestions: {e.suggestions}")
        # Maybe retry with different validation
    except CritiqueError as e:
        print(f"Critic failed: {e}")
        print(f"Model: {e.context.get('model', 'unknown')}")
        # Maybe retry with different model
    except ConfigurationError as e:
        print(f"Configuration issue: {e}")
        print(f"Config key: {e.context.get('config_key', 'unknown')}")
        # Fix configuration and retry
```

### Performance Profiling

```python
import time
import asyncio

async def profile_improve(prompt):
    times = {}

    start = time.time()
    result = await sifaka.improve(prompt)
    times['total'] = time.time() - start

    # Analyze the result
    overview = get_thought_overview(result)
    times['per_iteration'] = times['total'] / overview['total_iterations']
    times['per_model_call'] = times['total'] / overview['total_model_calls']

    print(f"Performance Profile:")
    for key, value in times.items():
        print(f"  {key}: {value:.2f}s")

    return result
```

## 📞 Getting Help

### 1. Check the Documentation

- **[Getting Started](GETTING_STARTED.md)** - Basic usage
- **[API Reference](API_REFERENCE.md)** - Complete API docs
- **[Architecture](ARCHITECTURE.md)** - How it works
- **[Performance](PERFORMANCE.md)** - Optimization tips

### 2. Quick Fixes with Presets

**💡 Pro Tip:** Most issues can be avoided by using presets instead of manual configuration!

| Issue | Preset Solution |
|-------|-----------------|
| API key missing | `export OPENAI_API_KEY="your-key"` |
| Too slow | `await sifaka.presets.quick_draft("prompt")` |
| Validation fails | `await sifaka.presets.academic_writing("prompt")` |
| Import error | `pip install "sifaka[all]"` |
| Rate limited | `await sifaka.presets.draft("prompt")` (uses gpt-4o-mini) |
| Need high quality | `await sifaka.presets.premium("prompt")` |
| Complex config | `await sifaka.presets.technical_docs("prompt")` |

### 3. Create a Minimal Reproduction

When reporting issues, include:

```python
import asyncio
import sifaka

async def reproduce_issue():
    # Minimal code that reproduces the problem
    try:
        result = await sifaka.improve("test prompt")
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Type: {type(e)}")
        import traceback
        traceback.print_exc()

asyncio.run(reproduce_issue())
```

### 4. Environment Information

Include this information when reporting issues:

```python
import sys
import sifaka

print(f"Python version: {sys.version}")
print(f"Sifaka version: {sifaka.__version__}")
print(f"Platform: {sys.platform}")

# Check dependencies
try:
    import openai
    print(f"OpenAI version: {openai.__version__}")
except ImportError:
    print("OpenAI not installed")

try:
    import anthropic
    print(f"Anthropic version: {anthropic.__version__}")
except ImportError:
    print("Anthropic not installed")
```

## 🎯 Prevention Tips

1. **Always set API keys** before running Sifaka
2. **Start with simple examples** before complex configurations
3. **Use reasonable iteration limits** (2-5 is usually enough)
4. **Test with fast models first** (`gpt-4o-mini`, `groq:llama-3.1-8b`)
5. **Enable logging** during development
6. **Check validation criteria** if iterations max out
7. **Handle rate limits** with retry logic
8. **Monitor performance** in production

Remember: Sifaka is designed to provide helpful error messages with specific suggestions. When you see an error, read the suggestions carefully - they usually contain the solution! 🎉
