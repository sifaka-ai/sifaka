# Configuration Simplification Guide

This guide covers the new configuration simplification features in Sifaka, making it easier to set up chains for common use cases.

## Overview

Sifaka now provides several ways to simplify configuration:

1. **Enhanced QuickStart Methods** - One-liner setups for common scenarios
2. **Configuration Presets** - Predefined configurations for specific use cases
3. **Setup Wizards** - Guided configuration with environment validation
4. **Improved Validation** - Better error messages and suggestions

## QuickStart Methods

### Basic Usage

```python
from sifaka.quickstart import QuickStart

# Simple chain with just a model and prompt
chain = QuickStart.basic_chain("openai:gpt-4", "Write a story about AI")
result = chain.run()
```

### Use Case-Specific Methods

#### Development Setup
Fast setup for testing and development:

```python
# Uses mock model by default, minimal iterations
chain = QuickStart.for_development()

# Or with specific model
chain = QuickStart.for_development("openai:gpt-4", "Test prompt")
```

#### Production Setup
Production-ready configuration with validation and improvement:

```python
chain = QuickStart.for_production(
    model_spec="openai:gpt-4",
    prompt="Generate a professional report",
    storage="memory+redis",  # Persistent storage
    validators=["length", "toxicity"],  # Content validation
    critics=["reflexion"]  # Quality improvement
)
```

#### Research Setup
Comprehensive setup for research and analysis:

```python
chain = QuickStart.for_research(
    model_spec="anthropic:claude-3-sonnet",
    prompt="Analyze the impact of AI on scientific research",
    storage="memory+redis+milvus",  # Full storage stack
    retrievers=True  # Include context retrievers
)
```

### Component-Specific Methods

#### Adding Validators
```python
chain = QuickStart.with_validation(
    "openai:gpt-4",
    validators=["length", "toxicity"],
    prompt="Generate content"
)
```

#### Adding Critics
```python
chain = QuickStart.with_critics(
    "openai:gpt-4",
    critics=["reflexion", "constitutional", "self_consistency"],
    prompt="Write an essay"
)
```

## Configuration Presets

### Available Presets

```python
from sifaka.quickstart import ConfigPresets

# Get preset configurations
dev_config = ConfigPresets.development()
content_config = ConfigPresets.content_generation()
fact_config = ConfigPresets.fact_checking()
research_config = ConfigPresets.research_analysis()
prod_config = ConfigPresets.production_safe()
```

### Using Presets

```python
# Use preset through QuickStart
chain = QuickStart.from_preset(
    "content_generation",
    model_spec="openai:gpt-4",
    prompt="Write a blog post"
)
```

### Preset Details

| Preset | Max Iterations | Storage | Validators | Critics |
|--------|---------------|---------|------------|---------|
| `development` | 1 | memory | none | none |
| `content_generation` | 2 | memory+redis | length, toxicity | constitutional |
| `fact_checking` | 3 | memory+redis+milvus | length | self_rag, reflexion |
| `research_analysis` | 5 | memory+redis+milvus | length | reflexion |
| `production_safe` | 3 | memory+redis | length, toxicity | constitutional, reflexion |

## Configuration Wizard

### Environment Validation

```python
from sifaka.quickstart import ConfigWizard

wizard = ConfigWizard()

# Check if environment is properly configured
env_status = wizard.validate_environment("openai:gpt-4")
print(env_status)
# {'openai_api_key': True, 'redis_available': False, ...}
```

### Guided Setup

```python
# Get recommendations for a use case
recommendations = wizard.get_recommendations("production", "openai:gpt-4")

# Setup chain with validation and fallbacks
chain = wizard.setup_for_use_case(
    "production",
    "openai:gpt-4",
    "Your production prompt"
)
```

## Storage Configuration

### Simple Storage Options

```python
# Memory only (default)
chain = QuickStart.basic_chain("openai:gpt-4", "prompt")

# File persistence
chain = QuickStart.with_file_storage("openai:gpt-4", "./thoughts.json")

# Redis caching
chain = QuickStart.with_redis("openai:gpt-4", redis_url="redis://localhost:6379")

# Milvus vector storage
chain = QuickStart.with_milvus("openai:gpt-4", collection_name="my_thoughts")
```

### Multi-Tier Storage

```python
# Two-tier: Memory + Redis
chain = QuickStart.full_stack("openai:gpt-4", storage="memory+redis")

# Three-tier: Memory + Redis + Milvus
chain = QuickStart.full_stack("openai:gpt-4", storage="memory+redis+milvus")
```

## Error Handling and Validation

### Improved Error Messages

The new configuration system provides helpful error messages:

```python
try:
    chain = QuickStart.for_production("openai:gpt-4", "")  # Empty prompt
except ConfigurationError as e:
    print(e.message)  # "Prompt is required for production chains"
    print(e.suggestions)  # ["Provide a specific prompt for your use case", ...]
```

### Configuration Validation

```python
from sifaka.core.chain import Chain
from sifaka.models.base import create_model

model = create_model("openai:gpt-4")
chain = Chain(model=model, prompt="Test")

# Validate configuration
try:
    chain.config.validate()
    print("Configuration is valid")
except ConfigurationError as e:
    print(f"Validation failed: {e}")
```

## Best Practices

### Development
- Use `QuickStart.for_development()` for quick testing
- Start with mock models to avoid API costs
- Use memory storage for fast iteration

### Production
- Always use `QuickStart.for_production()` with explicit prompts
- Include validators for content safety
- Use persistent storage (Redis/Milvus)
- Add critics for quality improvement

### Research
- Use `QuickStart.for_research()` for comprehensive analysis
- Enable retrievers for context
- Use full storage stack for data persistence
- Increase max iterations for thorough processing

## Migration from Manual Configuration

### Before (Manual)
```python
from sifaka.core.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.storage.redis import RedisStorage

model = create_model("openai:gpt-4")
validator = LengthValidator(min_length=50, max_length=1000)
critic = ReflexionCritic(model=model)
storage = RedisStorage(redis_config)

chain = Chain(model=model, prompt="Write content", storage=storage)
chain.validate_with(validator)
chain.improve_with(critic)
```

### After (Simplified)
```python
from sifaka.quickstart import QuickStart

chain = QuickStart.for_production(
    "openai:gpt-4",
    "Write content",
    validators=["length"],
    critics=["reflexion"]
)
```

## Environment Variables

Set these environment variables for automatic configuration:

```bash
# Model providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_TOKEN=your_hf_token

# Storage (optional)
REDIS_URL=redis://localhost:6379
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Set environment variables or pass explicitly
2. **Redis Not Available**: Wizard will fallback to memory storage
3. **Import Errors**: Install required dependencies with `pip install sifaka[all]`

### Getting Help

```python
from sifaka.quickstart import ConfigWizard

wizard = ConfigWizard()
env_status = wizard.validate_environment("your-model")

# Check what's missing
for component, available in env_status.items():
    if not available:
        print(f"Missing: {component}")
```
