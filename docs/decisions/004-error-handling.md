# ADR-004: Error Handling and Recovery Strategy

## Status
Accepted

## Context
Sifaka interacts with multiple external systems (LLM APIs, storage backends, web services) and processes user input, making it susceptible to various failure modes:
- Network timeouts and connection errors
- API rate limiting and authentication failures
- Invalid user input and configuration errors
- Out-of-memory conditions and resource exhaustion
- Plugin failures and compatibility issues

We need a comprehensive error handling strategy that:
- Provides clear, actionable error messages
- Enables graceful degradation when possible
- Supports retry logic for transient failures
- Maintains system stability under adverse conditions

## Decision
We will implement a hierarchical exception system with structured error handling, automatic retry mechanisms, and graceful degradation strategies.

```python
# Structured exceptions with suggestions
try:
    result = await improve("text")
except ModelProviderError as e:
    print(f"LLM API error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
    print(f"Provider: {e.provider}")
    print(f"Error code: {e.error_code}")
```

## Exception Hierarchy

### Base Exception
```python
class SifakaError(Exception):
    def __init__(self, message: str, suggestion: str = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)

    def __str__(self):
        if self.suggestion:
            return f"{self.message}\n💡 Suggestion: {self.suggestion}"
        return self.message
```

### Specific Exception Types
- **ConfigurationError**: Invalid configuration parameters
- **ModelProviderError**: LLM API failures
- **CriticError**: Critic evaluation failures
- **ValidationError**: Text validation failures
- **StorageError**: Storage backend issues
- **PluginError**: Plugin loading/execution failures
- **TimeoutError**: Operation time limits exceeded
- **MemoryError**: Memory bounds reached

## Error Classification

### 1. Transient Errors (Retryable)
- Network timeouts
- Rate limiting
- Server errors (5xx)
- Temporary resource unavailability

### 2. Permanent Errors (Non-retryable)
- Authentication failures
- Invalid requests (4xx)
- Configuration errors
- Missing resources

### 3. Partial Errors (Recoverable)
- Single critic failures
- Optional feature unavailability
- Non-critical validation failures

## Retry Strategy

### Configuration
```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    delay: float = 1.0
    backoff: float = 2.0

    def calculate_delay(self, attempt: int) -> float:
        return self.delay * (self.backoff ** attempt)
```

### Implementation
```python
@with_retry(RetryConfig(max_attempts=3, delay=1.0, backoff=2.0))
async def call_llm_api(prompt: str) -> str:
    # API call implementation
    pass
```

### Retry Logic
- Exponential backoff with jitter
- Selective retry based on error type
- Configurable retry limits
- Circuit breaker pattern for persistent failures

## Graceful Degradation

### 1. Critic Failures
When a critic fails:
- Log the error with context
- Continue with remaining critics
- Include failure information in results
- Provide fallback suggestions

### 2. Storage Failures
When storage fails:
- Fall back to memory storage
- Warn about data loss risk
- Continue processing
- Attempt to recover on next operation

### 3. Validation Failures
When validation fails:
- Log validation errors
- Continue with text improvement
- Include validation status in results
- Provide best-effort quality assessment

### 4. Tool Failures
When external tools fail:
- Disable tool-dependent features
- Use cached results if available
- Continue with available tools
- Provide reduced functionality notifications

## Error Recovery Mechanisms

### 1. Automatic Recovery
```python
class ErrorRecovery:
    async def recover_from_api_failure(self, error: ModelProviderError):
        if error.error_code == "rate_limit":
            await asyncio.sleep(error.retry_after or 60)
            return await self.retry_operation()

        if error.error_code == "authentication":
            await self.refresh_api_key()
            return await self.retry_operation()
```

### 2. Fallback Strategies
- Alternative API providers
- Cached responses
- Simplified operations
- Default configurations

### 3. Recovery Workflows
- Health check mechanisms
- Automatic failover
- Connection pooling
- Resource cleanup

## Error Reporting

### 1. Structured Logging
```python
logger.error(
    "Critic failure",
    extra={
        "critic": critic.name,
        "error_type": type(error).__name__,
        "error_code": getattr(error, 'error_code', None),
        "retryable": getattr(error, 'retryable', False),
        "text_length": len(text),
        "iteration": result.iteration,
    }
)
```

### 2. Error Metrics
- Error rate by type
- Recovery success rate
- Performance impact
- User impact assessment

### 3. User Feedback
- Clear error messages
- Actionable suggestions
- Progress indicators
- Status updates

## Implementation Examples

### 1. Configuration Validation
```python
def validate_config(config: Config):
    if config.temperature < 0 or config.temperature > 2:
        raise ConfigurationError(
            f"Temperature {config.temperature} is invalid",
            parameter="temperature",
            valid_range="0.0-2.0"
        )
```

### 2. API Error Handling
```python
async def call_openai_api(prompt: str):
    try:
        response = await openai.ChatCompletion.acreate(...)
        return response
    except openai.RateLimitError as e:
        raise ModelProviderError(
            "Rate limit exceeded",
            provider="OpenAI",
            error_code="rate_limit"
        ) from e
```

### 3. Graceful Critic Failure
```python
async def run_critics(text: str, critics: List[Critic]) -> List[CritiqueResult]:
    results = []
    for critic in critics:
        try:
            result = await critic.critique(text)
            results.append(result)
        except Exception as e:
            logger.warning(f"Critic {critic.name} failed: {e}")
            # Continue with other critics
    return results
```

## Consequences

### Positive
- Robust error handling improves reliability
- Clear error messages reduce user confusion
- Automatic recovery reduces manual intervention
- Graceful degradation maintains functionality
- Structured logging aids debugging

### Negative
- Additional complexity in error handling code
- Potential performance impact from retry logic
- Risk of masking underlying problems
- Complexity in testing error scenarios

### Mitigation
- Comprehensive error handling tests
- Performance monitoring for retry logic
- Clear documentation of error behaviors
- Configurable error handling strategies
- Regular review of error patterns

## Related Decisions
- [ADR-001: Single Function API](001-single-function-api.md)
- [ADR-002: Plugin Architecture](002-plugin-architecture.md)
- [ADR-003: Memory Management](003-memory-management.md)
