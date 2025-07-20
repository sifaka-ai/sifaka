# Installation

## Requirements

- Python 3.10 or higher
- pip or uv package manager

## Basic Installation

### Install from PyPI

```bash
# Install with pip
pip install sifaka

# Or with uv (recommended)
uv pip install sifaka
```

### Install from Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/sifaka-ai/sifaka
cd sifaka

# Install with uv (recommended)
uv pip install -e .

# Or with standard pip
pip install -e .
```

## Optional Dependencies

Sifaka supports various LLM providers and features through optional dependencies:

### Model Providers

Install optional dependencies for specific features:

```bash
# For Anthropic (Claude)
pip install sifaka[anthropic]

# For Google (Gemini)
pip install sifaka[gemini]

# For all providers
pip install sifaka[anthropic,gemini]
```

### Tool Support

```bash
# For web search and tool capabilities
pip install sifaka[tools]
```

### Development

```bash
# For development and testing (when installing from source)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

## Setting Up API Keys

Sifaka requires API keys for LLM providers. Set them as environment variables:

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google
export GEMINI_API_KEY="your-google-api-key"

# Groq
export GROQ_API_KEY="your-groq-api-key"

# Ollama (local LLMs - no API key required)
export OLLAMA_BASE_URL="http://localhost:11434/v1"  # Optional
```

Or use a `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GEMINI_API_KEY=your-google-api-key
```

## Verify Installation

```python
import sifaka
print(sifaka.__version__)

# Test basic functionality (requires API key)
from sifaka import improve_sync
result = improve_sync("Hello, world!")
print(result.final_text)
```

**Note**: If you haven't set up an API key, you'll get a clear error message:
```
ModelProviderError: No LLM provider available for critic. Please set up at least one provider by configuring the appropriate API key.
Suggestion: Set up at least one API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, or OLLAMA_API_KEY
```

### Using Ollama

To use Ollama for local LLM inference:

1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama3.2`
3. Use in Sifaka:

```python
from sifaka import improve_sync, Config
from sifaka.core.config import LLMConfig

# Use local Ollama
config = Config(llm=LLMConfig(provider="ollama", model="llama3.2"))
result = improve_sync("Hello world", config=config)
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you have the correct optional dependencies:

```bash
# Check installed packages
pip list | grep sifaka

# Reinstall with all dependencies
pip install --upgrade sifaka[all]
```

### API Key Errors

If you get authentication errors:

1. Verify your API key is correct
2. Check environment variables are set
3. Ensure your API key has proper permissions

### Python Version Issues

Sifaka requires Python 3.10+. Check your version:

```bash
python --version
```

## Next Steps

- Follow the [Quickstart Guide](getting-started/quickstart.md)
- Read the [Basic Usage Guide](guide/basic-usage.md)
- Explore [Available Critics](guide/critics.md)
