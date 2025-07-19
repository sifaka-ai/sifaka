# Configuration Guide

Sifaka offers flexible configuration options to customize text improvement behavior.

## Configuration Overview

Configuration can be set at multiple levels:
1. Function parameters (highest priority)
2. Config object
3. Environment variables
4. Defaults

## Basic Configuration

### Using Function Parameters

```python
from sifaka import improve

result = await improve(
    "Your text",
    model="gpt-4",
    temperature=0.8,
    max_iterations=5
)
```

### Using Config Object

```python
from sifaka import improve
from sifaka.core.config import Config, LLMConfig

config = Config(
    llm=LLMConfig(
        model="gpt-4",
        temperature=0.8
    )
)

result = await improve("Your text", config=config)
```

## Configuration Options

### LLM Configuration

Controls language model behavior:

```python
from sifaka.core.config import LLMConfig

llm_config = LLMConfig(
    model="gpt-4o-mini",           # Model to use
    critic_model="gpt-3.5-turbo",  # Different model for critics
    temperature=0.7,               # Creativity (0.0-2.0)
    max_tokens=2000,              # Max response length
    timeout_seconds=60.0          # Request timeout
)
```

**Available models:**
- OpenAI: `gpt-4`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- Anthropic: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- Google: `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-pro`

### Critic Configuration

Controls critic behavior:

```python
from sifaka.core.config import CriticConfig
from sifaka.core.types import CriticType

critic_config = CriticConfig(
    critics=[CriticType.SELF_REFINE, CriticType.REFLEXION],
    critic_model="gpt-3.5-turbo",  # Optional: different model for critics
    confidence_threshold=0.6       # Minimum confidence to continue
)
```

### Engine Configuration

Controls the improvement engine:

```python
from sifaka.core.config import EngineConfig

engine_config = EngineConfig(
    max_iterations=3,        # Maximum improvement rounds
    parallel_critics=True,   # Run critics in parallel
    timeout_seconds=120.0    # Overall timeout
)
```

### Complete Configuration Example

```python
from sifaka import improve
from sifaka.core.config import Config, LLMConfig, CriticConfig, EngineConfig
from sifaka.core.types import CriticType

config = Config(
    llm=LLMConfig(
        model="gpt-4",
        temperature=0.8,
        max_tokens=2000,
        timeout_seconds=60.0
    ),
    critic=CriticConfig(
        critics=[CriticType.SELF_REFINE, CriticType.STYLE],
        critic_model="gpt-3.5-turbo",
        confidence_threshold=0.7
    ),
    engine=EngineConfig(
        max_iterations=4,
        parallel_critics=True,
        timeout_seconds=180.0
    )
)

result = await improve("Your text", config=config)
```

## Environment Variables

Set default API keys and configuration:

```bash
# API Keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

# Optional: Default model
export SIFAKA_DEFAULT_MODEL="gpt-4o-mini"
export SIFAKA_DEFAULT_TEMPERATURE="0.7"
```

## Model Selection

### Choosing the Right Model

**For quality:**
- GPT-4 or Claude 3 Opus
- Best for complex reasoning
- Higher cost

**For speed:**
- GPT-3.5-turbo or Gemini Flash
- Good for simple improvements
- Lower cost

**For balance:**
- GPT-4o-mini or Claude 3 Haiku
- Good quality at reasonable cost
- Recommended default

### Model-Specific Tips

**OpenAI:**
```python
config = Config(
    llm=LLMConfig(
        model="gpt-4o-mini",
        temperature=0.7  # Good default
    )
)
```

**Anthropic:**
```python
config = Config(
    llm=LLMConfig(
        model="claude-3-haiku-20240307",
        temperature=0.6  # Claude prefers lower temps
    )
)
```

**Google:**
```python
config = Config(
    llm=LLMConfig(
        model="gemini-1.5-flash",
        temperature=0.8  # Gemini handles higher temps well
    )
)
```

## Temperature Settings

Temperature controls creativity vs consistency:

- **0.0-0.3**: Very consistent, minimal variation
- **0.4-0.6**: Balanced, some creativity
- **0.7-0.9**: Creative, more variation (recommended)
- **1.0-2.0**: Very creative, high variation

### Temperature by Use Case

```python
# Technical documentation
config = Config(llm=LLMConfig(temperature=0.3))

# Marketing copy
config = Config(llm=LLMConfig(temperature=0.8))

# Creative writing
config = Config(llm=LLMConfig(temperature=1.0))
```

## Performance Optimization

### Faster Processing

```python
# Use faster models and fewer iterations
fast_config = Config(
    llm=LLMConfig(
        model="gpt-3.5-turbo",
        timeout_seconds=30
    ),
    engine=EngineConfig(
        max_iterations=2,
        parallel_critics=True
    )
)
```

### Higher Quality

```python
# Use better models and more iterations
quality_config = Config(
    llm=LLMConfig(
        model="gpt-4",
        temperature=0.7
    ),
    critic=CriticConfig(
        critics=[
            CriticType.SELF_REFINE,
            CriticType.REFLEXION,
            CriticType.META_REWARDING
        ]
    ),
    engine=EngineConfig(
        max_iterations=5,
        parallel_critics=False  # Sequential for quality
    )
)
```

### Cost Optimization

```python
# Use different models for generation vs critique
cost_config = Config(
    llm=LLMConfig(
        model="gpt-4o-mini",        # Good generation model
        critic_model="gpt-3.5-turbo" # Cheaper critic model
    )
)
```

## Advanced Configuration

### Custom Timeouts

```python
config = Config(
    llm=LLMConfig(
        timeout_seconds=30.0  # Per LLM call timeout
    ),
    engine=EngineConfig(
        timeout_seconds=120.0  # Overall operation timeout
    )
)
```

### Parallel Processing

```python
# Enable parallel critic evaluation
config = Config(
    engine=EngineConfig(
        parallel_critics=True  # Run multiple critics simultaneously
    )
)
```

### Confidence Thresholds

```python
# Stop early if critics are confident
config = Config(
    critic=CriticConfig(
        confidence_threshold=0.8  # Stop if 80% confident
    )
)
```

## Configuration Patterns

### Development Configuration

```python
dev_config = Config(
    llm=LLMConfig(
        model="gpt-3.5-turbo",
        temperature=0.5  # Consistent for testing
    ),
    engine=EngineConfig(
        max_iterations=1,  # Fast feedback
        timeout_seconds=30.0
    )
)
```

### Production Configuration

```python
prod_config = Config(
    llm=LLMConfig(
        model="gpt-4o-mini",
        temperature=0.7,
        timeout_seconds=60.0
    ),
    critic=CriticConfig(
        critics=[CriticType.SELF_REFINE, CriticType.CONSTITUTIONAL],
        confidence_threshold=0.7
    ),
    engine=EngineConfig(
        max_iterations=3,
        parallel_critics=True,
        timeout_seconds=180.0
    )
)
```

### High-Stakes Configuration

```python
# For critical content (medical, legal, etc.)
critical_config = Config(
    llm=LLMConfig(
        model="gpt-4",
        temperature=0.3  # Low for consistency
    ),
    critic=CriticConfig(
        critics=[
            CriticType.CONSTITUTIONAL,
            CriticType.SELF_RAG,
            CriticType.META_REWARDING
        ],
        confidence_threshold=0.9  # High confidence required
    ),
    engine=EngineConfig(
        max_iterations=5,
        parallel_critics=False  # Sequential for thoroughness
    )
)
```

## Troubleshooting Configuration

### Common Issues

**Timeouts:**
```python
# Increase timeouts for long texts
config = Config(
    llm=LLMConfig(timeout_seconds=120.0),
    engine=EngineConfig(timeout_seconds=300.0)
)
```

**Inconsistent results:**
```python
# Lower temperature for consistency
config = Config(
    llm=LLMConfig(temperature=0.3)
)
```

**High costs:**
```python
# Use cheaper models and fewer iterations
config = Config(
    llm=LLMConfig(model="gpt-3.5-turbo"),
    engine=EngineConfig(max_iterations=2)
)
```

## Best Practices

1. **Start with defaults**: Only configure what you need
2. **Test configurations**: Find what works for your use case
3. **Monitor costs**: Use appropriate models for your budget
4. **Set timeouts**: Prevent runaway operations
5. **Use environment variables**: For API keys and defaults
6. **Document your config**: Explain why specific settings were chosen
