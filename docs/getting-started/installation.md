# Installation Guide

Get Sifaka up and running in minutes with this step-by-step installation guide.

## Prerequisites

- **Python 3.11 or higher** (required)
- **pip 21.0 or higher** (for modern dependency resolution)
- **Virtual environment** (recommended)

## Quick Installation

### 1. Basic Installation
For most users, start with the core installation:

```bash
pip install sifaka
```

### 2. With Model Providers
Add your preferred AI model providers:

```bash
# OpenAI models (GPT-4, GPT-3.5)
pip install sifaka[openai]

# Anthropic models (Claude)
pip install sifaka[anthropic]

# Google Gemini models
pip install sifaka[gemini]

# All model providers
pip install sifaka[models]
```

### 3. Complete Installation
For full functionality including storage and validation:

```bash
# Everything you need
pip install sifaka[all]
```

## Installation Options

Sifaka uses modular dependencies - install only what you need:

### Core Features
```bash
pip install sifaka                    # Basic functionality
pip install sifaka[models]            # All model providers
pip install sifaka[retrievers]        # Redis + Milvus storage
pip install sifaka[classifiers]       # ML text classifiers
pip install sifaka[guardrails]        # GuardrailsAI integration
```

### Common Combinations
```bash
# Production setup
pip install sifaka[openai,redis]

# Research and development
pip install sifaka[models,classifiers,guardrails]

# Development with all features
pip install sifaka[full]
```

## Environment Setup

### Option 1: Using uv (Recommended)
[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles virtual environments automatically:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project and install Sifaka
uv init sifaka-project
cd sifaka-project
uv add sifaka[all]

# Or install in existing project
uv add sifaka[all]
```

### Option 2: Using pip and venv
Traditional Python environment setup:

```bash
# Create virtual environment
python -m venv sifaka-env

# Activate it
source sifaka-env/bin/activate  # Linux/Mac
# OR
sifaka-env\Scripts\activate     # Windows

# Install Sifaka
pip install sifaka[all]
```

### 3. Set Up API Keys
Create a `.env` file in your project directory:

```bash
# OpenAI (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: HuggingFace (for open models)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

## Verify Installation

Test your installation with this simple script:

```python
import sifaka
from sifaka.core.thought import Thought

# Create a basic thought
thought = Thought(prompt="Hello, Sifaka!")
print(f"✅ Sifaka version: {sifaka.__version__}")
print(f"✅ Thought created: {thought.prompt}")
```

## Optional Services

### Redis (for caching)
```bash
# Using Docker (recommended)
docker run -d -p 6379:6379 redis:alpine

# Verify connection
python -c "import redis; r=redis.Redis(); print('✅ Redis connected')"
```

### Milvus (for vector storage)
```bash
# Download and start Milvus
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

## Troubleshooting

### Common Issues

**Python Version Error**
```bash
# Check your Python version
python --version
# Should show 3.11 or higher
```

**Import Errors**
```bash
# Check installed packages
pip list | grep sifaka

# Reinstall if needed
pip uninstall sifaka
pip install sifaka[all]
```

**API Key Issues**
```bash
# Test API keys
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))"
```

### Getting Help

- Check the [troubleshooting guide](../troubleshooting/common-issues.md)
- Review [configuration errors](../troubleshooting/configuration-errors.md)
- See [import problems](../troubleshooting/import-problems.md)

## Next Steps

Now that Sifaka is installed:

1. **[Create your first chain](first-chain.md)** - Build a simple text generation pipeline
2. **[Learn basic concepts](basic-concepts.md)** - Understand Thoughts, Models, and Validators
3. **[Explore examples](../../examples/)** - See working code for different use cases

## Development Installation

For contributors and advanced users:

### Using uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka

# Install in development mode with uv
uv sync --all-extras

# Set up pre-commit hooks
uv run pre-commit install
```

### Using pip
```bash
# Clone the repository
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka

# Install in development mode
pip install -e .[full]

# Set up pre-commit hooks
make install-dev
```

See the [contributing guidelines](../guidelines/CONTRIBUTING_GUIDELINES.md) for more details.
