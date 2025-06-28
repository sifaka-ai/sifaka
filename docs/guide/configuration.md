# Configuration Guide

Sifaka provides flexible configuration options to customize the text improvement process.

## Basic Configuration

```python
from sifaka import improve, Config

# Create a custom configuration
config = Config(
    temperature=0.7,
    max_iterations=5,
    min_quality_score=0.8
)

result = await improve(
    "Your text here",
    config=config
)
```

## Configuration Options

### Core Settings

```python
config = Config(
    # LLM temperature (0.0-1.0)
    temperature=0.7,

    # Maximum improvement iterations
    max_iterations=3,

    # Minimum quality score to continue iterations
    min_quality_score=0.7,

    # Model to use
    model="gpt-4o-mini",

    # Provider (openai, anthropic, google)
    provider="openai",

    # Request timeout in seconds
    timeout=60.0,

    # Use advanced confidence calculation
    use_advanced_confidence=True
)
```

### Performance Optimization

By default, Sifaka uses a hybrid model approach for optimal performance:
- **Generation**: `gpt-4o-mini` (high quality)
- **Critics**: `gpt-3.5-turbo` (2-3x faster)

This reduces processing time by 50-60% with minimal quality impact. You can override these defaults:

```python
# Use the same model for everything
config = Config(
    model="gpt-4o-mini",
    critic_model="gpt-4o-mini",  # Override the default gpt-3.5-turbo
)

# Or use a faster model for everything
config = Config(
    model="gpt-3.5-turbo",
    critic_model="gpt-3.5-turbo",
)
```

### Memory Management

Control memory usage for long-running operations:

```python
config = Config(
    # Maximum text length in characters
    max_text_length=50000,

    # Maximum critique history to keep
    max_history_size=10
)
```

### Retry Configuration

Handle transient failures gracefully:

```python
from sifaka import RetryConfig

config = Config(
    retry_config=RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=True
    )
)
```

## Environment Variables

Configure Sifaka using environment variables:

```bash
# API Keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Default settings
export SIFAKA_DEFAULT_MODEL="gpt-4o-mini"
export SIFAKA_DEFAULT_TEMPERATURE="0.7"
export SIFAKA_DEFAULT_MAX_ITERATIONS="3"
```

## Provider-Specific Configuration

### OpenAI
```python
result = await improve(
    text,
    provider="openai",
    model="gpt-4o",
    api_key="your-openai-key",
    config=Config(temperature=0.8)
)
```

### Anthropic
```python
result = await improve(
    text,
    provider="anthropic",
    model="claude-3-opus-20240229",
    api_key="your-anthropic-key"
)
```

### Google
```python
result = await improve(
    text,
    provider="google",
    model="gemini-1.5-pro",
    api_key="your-google-key"
)
```

## Advanced Configuration

### Custom Base URL

For self-hosted or proxy endpoints:

```python
config = Config(
    base_url="https://your-proxy.com/v1"
)
```

### Middleware

Add custom middleware for logging, caching, etc:

```python
from sifaka.core.middleware import LoggingMiddleware, CachingMiddleware

# Create middleware pipeline
pipeline = MiddlewarePipeline()
pipeline.add(LoggingMiddleware())
pipeline.add(CachingMiddleware(max_size=100))

# Use with improve (if supported in future versions)
```

### Storage Backend

Configure where to store results:

```python
from sifaka.storage import FileStorage

storage = FileStorage(base_path="./sifaka_cache")

# Use with configuration (if supported in future versions)
config = Config(storage_backend=storage)
```

## Configuration Precedence

Configuration is applied in this order (later overrides earlier):

1. Default values
2. Environment variables
3. Config object
4. Function parameters

Example:
```python
# Environment: SIFAKA_DEFAULT_MODEL="gpt-4o-mini"

config = Config(model="gpt-4o")

result = await improve(
    text,
    model="claude-3-haiku-20240307",  # This takes precedence
    config=config
)
```

## Best Practices

1. **Start with defaults**: The default configuration works well for most use cases
2. **Adjust temperature**: Lower (0.3-0.5) for factual content, higher (0.7-0.9) for creative
3. **Set appropriate timeouts**: Longer for complex improvements
4. **Use environment variables**: For API keys and deployment settings
5. **Monitor iterations**: Set `max_iterations` based on your quality/speed tradeoff

## Configuration Examples

### High-Quality Technical Writing
```python
technical_config = Config(
    temperature=0.3,
    max_iterations=5,
    min_quality_score=0.85,
    use_advanced_confidence=True
)
```

### Creative Writing
```python
creative_config = Config(
    temperature=0.8,
    max_iterations=3,
    min_quality_score=0.7
)
```

### Quick Improvements
```python
quick_config = Config(
    temperature=0.5,
    max_iterations=1,
    timeout=30.0
)
```

### Production Settings
```python
production_config = Config(
    temperature=0.5,
    max_iterations=3,
    timeout=60.0,
    retry_config=RetryConfig(
        max_attempts=3,
        initial_delay=1.0
    ),
    use_advanced_confidence=True
)
