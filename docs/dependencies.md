# Sifaka Dependencies

This document explains the dependency management in Sifaka and provides guidance on installing the right dependencies for your use case.

## Core Dependencies

These dependencies are always installed with Sifaka and provide the basic functionality:

- `pydantic`: Data validation and settings management
- `typing-extensions`: Additional typing features
- `python-dotenv`: Environment variable management
- `tqdm`: Progress bars
- `requests` and `httpx`: HTTP clients
- `tenacity`: Retry logic

## Optional Dependencies

Sifaka uses optional dependencies to keep the core package lightweight while allowing for extended functionality. These dependencies are organized into logical groups.

### Model Providers

| Group | Dependencies | Features |
|-------|--------------|----------|
| `openai` | `openai`, `tiktoken` | OpenAI API integration and token counting |
| `anthropic` | `anthropic` | Anthropic Claude API integration |
| `gemini` | `google-generativeai` | Google Gemini API integration |
| `all_models` | All of the above | All model provider integrations |

```bash
# Install with OpenAI support
pip install "sifaka[openai]"

# Install with Anthropic support
pip install "sifaka[anthropic]"

# Install with all model providers
pip install "sifaka[all_models]"
```

### Classifiers

| Group | Dependencies | Features |
|-------|--------------|----------|
| `toxicity` | `detoxify`, `torch`, `transformers` | Toxicity detection |
| `sentiment` | `vaderSentiment` | Sentiment analysis |
| `profanity` | `better-profanity` | Profanity detection |
| `language` | `langdetect` | Language detection |
| `readability` | `textstat` | Readability metrics |
| `ner` | `spacy` | Named Entity Recognition |
| `all_classifiers` | All of the above | All classifier functionality |

```bash
# Install with toxicity detection
pip install "sifaka[toxicity]"

# Install with all classifiers
pip install "sifaka[all_classifiers]"
```

### Integrations

| Group | Dependencies | Features |
|-------|--------------|----------|
| `pydantic-ai` | `pydantic-ai` | PydanticAI integration |
| `guardrails` | `guardrails-ai` | Guardrails integration |
| `all_integrations` | All of the above | All integrations |

```bash
# Install with Guardrails integration
pip install "sifaka[guardrails]"

# Install with all integrations
pip install "sifaka[all_integrations]"
```

### Development

| Group | Dependencies | Features |
|-------|--------------|----------|
| `dev` | `pytest`, `black`, `isort`, `mypy`, `ruff`, `pytest-cov`, `flake8` | Development tools |
| `benchmark` | `memory-profiler`, `psutil`, `numpy`, `pandas`, `matplotlib` | Benchmarking tools |

```bash
# Install with development tools
pip install "sifaka[dev]"

# Install with benchmarking tools
pip install "sifaka[benchmark]"
```

### Complete Installation

To install Sifaka with all optional dependencies (except development tools):

```bash
pip install "sifaka[all]"
```

## Feature-Specific Dependencies

This section explains which dependencies are required for specific features in Sifaka.

### Rules

| Feature | Required Dependencies |
|---------|----------------------|
| Length rules | None (core only) |
| Toxicity rules | `toxicity` |
| Sentiment rules | `sentiment` |
| Language rules | `language` |
| Profanity rules | `profanity` |
| NER rules | `ner` |

### Critics

| Feature | Required Dependencies |
|---------|----------------------|
| Prompt critic | Any model provider (`openai`, `anthropic`, or `gemini`) |
| Reflexion critic | Any model provider |
| Constitutional critic | Any model provider |
| Self-refine critic | Any model provider |
| Self-RAG critic | Any model provider |
| LAC critic | Any model provider |

### Adapters

| Feature | Required Dependencies |
|---------|----------------------|
| Guardrails adapter | `guardrails` |
| PydanticAI adapter | `pydantic-ai` |
| Classifier adapter | Depends on classifier (`toxicity`, `sentiment`, etc.) |

## Dependency Conflicts

If you encounter dependency conflicts, try installing dependencies in this order:

1. Install core Sifaka: `pip install sifaka`
2. Install model providers: `pip install "sifaka[all_models]"`
3. Install classifiers: `pip install "sifaka[all_classifiers]"`
4. Install integrations: `pip install "sifaka[all_integrations]"`

For specific version conflicts, you may need to install dependencies manually with specific version constraints.
