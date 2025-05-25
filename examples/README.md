# Sifaka Examples

This directory contains comprehensive examples demonstrating all aspects of the Sifaka framework. Each example is self-contained and showcases specific features and use cases.

## Quick Start

To run any example:

```bash
# Navigate to the examples directory
cd examples

# Run a specific example
python openai/constitutional_critic_guardrails.py
python mock/basic_chain_demo.py
```

## Example Categories

### 🤖 OpenAI Examples (`openai/`)

Examples using OpenAI models with various critics and validators.

- **`constitutional_critic_guardrails.py`** - Constitutional critic with Guardrails validators for ethical AI content
- **`n_critics_redis_milvus.py`** - N-Critics ensemble with Redis model retrieval and Milvus critic retrieval

### 🧠 Anthropic Examples (`anthropic/`)

Examples using Anthropic Claude models with advanced retrieval and validation.

- **`self_rag_redis_length.py`** - Self-RAG critic with Redis retrieval and length validation
- **`self_refine_multi_validators.py`** - Self-Refine critic with comprehensive validator suite

### 🦙 Ollama Examples (`ollama/`)

Examples using local Ollama models for privacy-focused processing.

- **`spanish_translation_prompt_critic.py`** - Spanish to English translation with prompt critic and language validation
- **`constitutional_critic_redis.py`** - Constitutional critic with Redis retrieval for digital privacy content

### 🤗 HuggingFace Examples (`huggingface/`)

Examples using HuggingFace models with remote inference and caching.

- **`self_rag_no_retrievers.py`** - Self-RAG critic using only internal model knowledge
- **`reflexion_three_tier_caching.py`** - Reflexion critic with Memory → Redis → Milvus caching

### 💾 Thought Caching Examples (`thought_caching/`)

Examples demonstrating thought persistence and caching strategies.

- **`file_based_caching.py`** - File-based thought persistence for debugging and analysis
- **`three_tier_caching.py`** - Memory → Redis → Milvus three-tier caching architecture

### 🎭 Mock Examples (`mock/`)

Examples using mock models for testing and learning without external dependencies.

- **`basic_chain_demo.py`** - Basic chain setup and execution (perfect for getting started)
- **`comprehensive_validation_demo.py`** - Multiple validators working together
- **`multi_critic_ensemble.py`** - Multiple critics providing specialized feedback

### 🧪 Tests (`tests/`)

Test utilities for validating all examples.

- **`test_all_examples.py`** - Automated test runner for all examples

## Prerequisites

### Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# HuggingFace
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

### External Services

Some examples require external services:

- **Redis**: `docker run -d -p 6379:6379 redis:latest`
- **Milvus**: Follow [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md)
- **Ollama**: Download from [ollama.ai](https://ollama.ai) and install models

### Dependencies

Install required dependencies:

```bash
# Core dependencies
pip install sifaka

# For specific features
pip install sifaka[openai]      # OpenAI examples
pip install sifaka[anthropic]   # Anthropic examples
pip install sifaka[huggingface] # HuggingFace examples
pip install sifaka[redis]       # Redis caching examples
pip install sifaka[milvus]      # Milvus vector storage examples
pip install sifaka[guardrails]  # Guardrails validation examples

# Or install everything
pip install sifaka[all]
```

## Running Examples

### Individual Examples

```bash
# Basic example (no external dependencies)
python mock/basic_chain_demo.py

# OpenAI example (requires API key)
python openai/constitutional_critic_guardrails.py

# Local processing example (requires Ollama)
python ollama/spanish_translation_prompt_critic.py
```

### Test All Examples

```bash
# Run all examples and generate report
python tests/test_all_examples.py
```

## Example Features Matrix

| Example | Model | Critics | Validators | Retrievers | Caching |
|---------|-------|---------|------------|------------|---------|
| OpenAI Constitutional | OpenAI | Constitutional | Guardrails | None | None |
| OpenAI N-Critics | OpenAI | N-Critics | None | Redis + Milvus | None |
| Anthropic Self-RAG | Anthropic | Self-RAG | Length | Redis | None |
| Anthropic Self-Refine | Anthropic | Self-Refine | Multiple | None | None |
| Ollama Translation | Ollama | Prompt | Language | None | None |
| Ollama Constitutional | Ollama | Constitutional | None | Redis | None |
| HF Self-RAG | HuggingFace | Self-RAG | None | None | None |
| HF Reflexion | HuggingFace | Reflexion | None | Multiple | Three-tier |
| File Caching | Mock | Reflexion | Length | None | File |
| Three-tier Caching | Mock | Self-Refine | Length | None | Memory+Redis+Milvus |
| Basic Demo | Mock | Reflexion | Length | None | None |
| Validation Demo | Mock | Self-Refine | Multiple | None | None |
| Multi-Critic | Mock | Multiple | Length | None | None |

## Learning Path

1. **Start Here**: `mock/basic_chain_demo.py` - Learn basic concepts
2. **Validation**: `mock/comprehensive_validation_demo.py` - Understand quality control
3. **Critics**: `mock/multi_critic_ensemble.py` - See improvement strategies
4. **Real Models**: Choose based on your preferred provider
5. **Advanced Features**: Explore caching and retrieval examples

## Troubleshooting

### Common Issues

- **API Key Errors**: Ensure environment variables are set correctly
- **Service Unavailable**: Check that Redis/Milvus/Ollama services are running
- **Import Errors**: Install required dependencies with `pip install sifaka[feature]`
- **Permission Errors**: Ensure write permissions for file caching examples

### Getting Help

- Check the main [Sifaka documentation](../README.md)
- Review [API reference](../docs/API_REFERENCE.md)
- See [architecture documentation](../docs/ARCHITECTURE.md)

## Contributing

To add new examples:

1. Choose the appropriate category directory
2. Follow the existing naming convention
3. Include comprehensive docstrings and comments
4. Add error handling and logging
5. Test with `python tests/test_all_examples.py`
6. Update this README if needed
