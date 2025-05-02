# Setup Guide

This guide will help you set up and install Sifaka for development and production use.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sifaka.git
cd sifaka
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install production dependencies
pip install -r requirements.txt
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
STABILITY_API_KEY=your_stability_api_key

# Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### 2. Chain Configuration

Create a configuration file for your chain (e.g., `config/chains/text_validation.json`):

```json
{
    "name": "text_validation",
    "description": "Text validation chain",
    "model": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "config": {
            "temperature": 0.7,
            "max_tokens": 500
        }
    },
    "rules": [
        {
            "name": "length",
            "min_chars": 10,
            "max_chars": 1000
        },
        {
            "name": "toxicity",
            "threshold": 0.8
        }
    ],
    "max_attempts": 3
}
```

## Development Setup

### 1. Install Development Tools

```bash
# Install pre-commit hooks
pre-commit install

# Install testing dependencies
pip install -r requirements-test.txt
```

### 2. Run Tests

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=sifaka tests/
```

### 3. Code Quality Tools

```bash
# Run linter
flake8 sifaka tests

# Run type checker
mypy sifaka tests

# Run formatter
black sifaka tests
```

## Production Setup

### 1. Install Production Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Logging

Create a logging configuration file (e.g., `config/logging.yaml`):

```yaml
version: 1
formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: standard
    level: INFO
  file:
    class: logging.FileHandler
    formatter: standard
    filename: sifaka.log
    level: INFO
root:
  level: INFO
  handlers: [console, file]
```

### 3. Start the Service

```bash
# Start the service
python -m sifaka.service
```

## Docker Setup

### 1. Build the Image

```bash
docker build -t sifaka .
```

### 2. Run the Container

```bash
docker run -d \
  --name sifaka \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -e OPENAI_API_KEY=your_api_key \
  sifaka
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify API keys are correctly set in environment variables
   - Check API key permissions and quotas

2. **Configuration Errors**
   - Validate configuration files against schemas
   - Check for missing required fields

3. **Performance Issues**
   - Monitor resource usage
   - Check for rate limiting
   - Verify model provider status

### Getting Help

- Check the [documentation](README.md)
- Open an issue on GitHub
- Join the community chat

## Next Steps

- Read the [documentation](README.md)
- Try the [examples](examples/README.md)
- Learn about [contributing](contributing.md)