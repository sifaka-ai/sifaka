# Installation

## Requirements

- Python 3.9 or higher
- pip or uv package manager

## Basic Installation

Install Sifaka using pip:

```bash
pip install sifaka
```

Or using uv (recommended):

```bash
uv pip install sifaka
```

## Optional Dependencies

Sifaka supports various LLM providers and features through optional dependencies:

### Model Providers

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
# For development and testing
pip install sifaka[dev]
```

## Setting Up API Keys

Sifaka requires API keys for LLM providers. Set them as environment variables:

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google
export GOOGLE_API_KEY="your-google-api-key"
```

Or use a `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
```

## Verify Installation

```python
import sifaka
print(sifaka.__version__)

# Test basic functionality
from sifaka import improve_sync
result = improve_sync("Hello, world!")
print(result.final_text)
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

Sifaka requires Python 3.9+. Check your version:

```bash
python --version
```

## Next Steps

- Follow the [Quickstart Guide](quickstart.md)
- Read the [Basic Usage Guide](guide/basic-usage.md)
- Explore [Available Critics](critics/overview.md)
