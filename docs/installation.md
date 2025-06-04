# Installation

## Requirements

- Python 3.11+ (required for PydanticAI compatibility)
- PydanticAI 0.2.0+
- At least one model provider (OpenAI, Anthropic, Google, Groq, Ollama)

## Install Sifaka

### Basic Installation
```bash
pip install sifaka
```

### With Model Providers
```bash
# OpenAI support
pip install sifaka[openai]

# Anthropic support
pip install sifaka[anthropic]

# Google Gemini support
pip install sifaka[gemini]

# Groq support (included in core)
pip install sifaka

# Ollama support
pip install sifaka[ollama]

# All model providers
pip install sifaka[models]

# Everything (models + storage + classifiers)
pip install sifaka[all]
```

### Development Installation
```bash
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka
pip install -e .[all]
```

## Environment Setup

### API Keys
Set environment variables for your model providers:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-key"

# Groq
export GROQ_API_KEY="your-groq-key"

# Ollama (local models, no API key needed)
# Just ensure Ollama is running locally
```

### Verify Installation
```python
import sifaka
from pydantic_ai import Agent
from sifaka.agents import create_pydantic_chain

print(f"Sifaka version: {sifaka.__version__}")

# Test basic functionality
agent = Agent("openai:gpt-4", system_prompt="You are helpful.")
chain = create_pydantic_chain(agent=agent)
print("✅ Installation successful!")
```

## Optional Dependencies

### Storage Backends
```bash
# Redis storage (production-ready with MCP)
pip install sifaka[redis]

# PostgreSQL storage (enterprise-grade with full-text search)
pip install asyncpg  # Required for PostgreSQL backend

# All storage backends
pip install sifaka[retrievers]
```

### Text Processing and Classifiers
```bash
# Advanced text analysis and classifiers
pip install sifaka[classifiers]

# Performance utilities
pip install sifaka[performance]
```

## Troubleshooting

### Common Issues

**Import Error**: Make sure you have the correct extras installed
```bash
pip install sifaka[openai,anthropic]
```

**API Key Error**: Verify your environment variables are set
```bash
echo $OPENAI_API_KEY
```

**Version Conflicts**: Use a virtual environment
```bash
python -m venv sifaka-env
source sifaka-env/bin/activate  # On Windows: sifaka-env\Scripts\activate
pip install sifaka[all]
```
