# Sifaka Installation Guide

This guide provides comprehensive installation instructions for Sifaka with different use cases and dependency requirements.

## Quick Start

### Basic Installation
```bash
pip install sifaka
```

### Development Installation
```bash
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka
pip install -e .[full]
```

## Installation Options

Sifaka uses a modular dependency system. Install only what you need:

### Core Installation
```bash
pip install sifaka
```
Includes: Pydantic, basic utilities, HTTP clients

### Model Providers

#### OpenAI Models
```bash
pip install sifaka[openai]
```
Includes: OpenAI API client, tiktoken for token counting

#### Anthropic Models
```bash
pip install sifaka[anthropic]
```
Includes: Anthropic API client

#### Google Gemini Models
```bash
pip install sifaka[gemini]
```
Includes: Google GenerativeAI client

#### All Model Providers
```bash
pip install sifaka[models]
```

### Retrievers and Vector Databases

#### Redis Retriever
```bash
pip install sifaka[redis]
```

#### Milvus Vector Database
```bash
pip install sifaka[milvus]
```

#### All Retrievers
```bash
pip install sifaka[retrievers]
```

### Machine Learning and Classification

#### Text Classifiers
```bash
pip install sifaka[classifiers]
```
Includes: scikit-learn, TextBlob, langdetect, better-profanity for text classification

### Validation and Guardrails

#### GuardrailsAI Integration
```bash
pip install sifaka[guardrails]
```

### Performance Tools
```bash
pip install sifaka[performance]
```
Includes: NumPy, psutil, tqdm for monitoring and optimization

### Development Tools
```bash
pip install sifaka[dev]
```
Includes: pytest, black, mypy, ruff, type stubs

## Common Installation Patterns

### Production Use (Minimal)
```bash
pip install sifaka[openai,redis]
```

### Research and Development
```bash
pip install sifaka[models,classifiers,guardrails]
```

### Full Installation (Everything)
```bash
pip install sifaka[all]
```

### Development with All Features
```bash
pip install sifaka[full]
```

## Requirements

- Python 3.11 or higher
- pip 21.0 or higher (for modern dependency resolution)

## Verification

Test your installation:

```python
import sifaka
from sifaka.core.thought import Thought

# Create a basic thought
thought = Thought(prompt="Hello, world!")
print(f"Sifaka version: {sifaka.__version__}")
print(f"Thought created: {thought.prompt}")
```

## Troubleshooting

### Common Issues

1. **Python Version**: Ensure you're using Python 3.11+
2. **Dependency Conflicts**: Use a virtual environment
3. **Optional Dependencies**: Install specific extras for features you need

### Virtual Environment Setup
```bash
python -m venv sifaka-env
source sifaka-env/bin/activate  # On Windows: sifaka-env\Scripts\activate
pip install sifaka[full]
```

### Docker Installation
```dockerfile
FROM python:3.11-slim
RUN pip install sifaka[all]
```

## Next Steps

- Check out the [examples/](examples/) directory
- Read the [README.md](README.md) for usage examples
- See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for development guidelines
