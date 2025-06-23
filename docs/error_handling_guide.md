# Error Handling Guide for Sifaka

This guide establishes consistent error handling patterns across the Sifaka codebase.

## Principles

1. **Use Custom Exceptions**: Always use Sifaka's custom exceptions from `core.exceptions`
2. **Provide Context**: Include helpful error messages with suggestions for resolution
3. **Fail Gracefully**: Return valid results with error information when possible
4. **Retry Transient Failures**: Use retry decorators for network and API operations
5. **Log Appropriately**: Log errors with proper severity levels

## Exception Hierarchy

```python
SifakaError (base)
├── ConfigurationError    # Invalid config values
├── ModelProviderError    # LLM API issues
├── CriticError          # Critic evaluation failures
├── ValidationError      # Text validation failures
├── StorageError         # Storage operation failures
├── PluginError          # Plugin loading/execution
├── TimeoutError         # Operation timeouts
└── MemoryError          # Memory limit exceeded
```

## Standard Patterns

### 1. Basic Error Handling

```python
from sifaka.core.exceptions import ModelProviderError

try:
    response = await client.complete(messages)
except openai.APIError as e:
    # Convert to our custom exception with context
    raise ModelProviderError(
        f"OpenAI API error: {str(e)}",
        suggestion="Check your API key and rate limits"
    ) from e
```

### 2. Graceful Degradation

```python
async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
    try:
        # Normal operation
        return await self._perform_critique(text, result)
    except Exception as e:
        # Return valid result with error info
        return CritiqueResult(
            critic=self.name,
            feedback=f"Error during critique: {str(e)}",
            suggestions=["Review the text manually"],
            needs_improvement=True,
            confidence=0.0,
            metadata={"error": str(e), "error_type": type(e).__name__}
        )
```

### 3. Using Retry Decorator

```python
from sifaka.core.retry import with_retry, RETRY_STANDARD

@with_retry(RETRY_STANDARD)
async def call_api(self, messages: List[Dict[str, str]]) -> Response:
    """API call with automatic retry on transient failures."""
    return await self.client.complete(messages)
```

### 4. Validation Errors

```python
from sifaka.core.exceptions import ValidationError

def validate_config(config: Config) -> None:
    if config.max_iterations < 1:
        raise ValidationError(
            "max_iterations must be at least 1",
            suggestion=f"Set max_iterations to a value between 1 and 10"
        )
```

### 5. Storage Errors

```python
from sifaka.core.exceptions import StorageError

async def save(self, result: SifakaResult) -> str:
    try:
        # Storage operation
        return await self._write_to_disk(result)
    except IOError as e:
        raise StorageError(
            f"Failed to save result: {str(e)}",
            suggestion="Check disk space and permissions"
        ) from e
```

## Anti-Patterns to Avoid

### ❌ Don't use bare except
```python
# BAD
try:
    do_something()
except:
    pass
```

### ❌ Don't swallow errors silently
```python
# BAD
try:
    result = process()
except Exception:
    result = None  # Silent failure
```

### ❌ Don't use generic exceptions
```python
# BAD
raise Exception("Something went wrong")

# GOOD
raise CriticError(
    "Failed to parse critic response",
    suggestion="Check the prompt format"
)
```

### ❌ Don't lose error context
```python
# BAD
except Exception as e:
    raise RuntimeError("Operation failed")

# GOOD
except Exception as e:
    raise OperationError(
        f"Operation failed: {str(e)}",
        suggestion="Check input parameters"
    ) from e
```

## Error Recovery Strategies

### 1. Fallback Values
```python
try:
    config = load_config()
except ConfigurationError:
    # Use defaults
    config = Config()
```

### 2. Alternative Approaches
```python
try:
    # Try primary critic
    result = await primary_critic.critique(text)
except CriticError:
    # Fallback to simpler critic
    result = await fallback_critic.critique(text)
```

### 3. Partial Results
```python
results = []
for item in items:
    try:
        results.append(process(item))
    except ProcessingError as e:
        # Log error but continue
        logger.warning(f"Failed to process {item}: {e}")
        results.append(None)
```

## Logging Errors

```python
import logging

logger = logging.getLogger(__name__)

try:
    operation()
except ModelProviderError as e:
    logger.error(f"API call failed: {e}", exc_info=True)
    # Re-raise or handle as appropriate
    raise
```

## Testing Error Handling

```python
import pytest
from sifaka.core.exceptions import ValidationError

def test_invalid_config_raises_error():
    with pytest.raises(ValidationError) as exc_info:
        validate_config(Config(max_iterations=-1))
    
    assert "must be at least 1" in str(exc_info.value)
    assert exc_info.value.suggestion is not None
```

## Summary

1. **Always** use custom exceptions from `core.exceptions`
2. **Always** include helpful error messages and suggestions
3. **Never** use bare except or swallow errors silently
4. **Use** retry decorators for external operations
5. **Log** errors appropriately for debugging
6. **Test** error handling paths explicitly