# ADR-004: Error Handling Philosophy

## Status
Accepted

## Context
When working with external LLM APIs and complex text processing, many things can go wrong:
- Network failures
- API rate limits
- Invalid responses
- Token limits exceeded
- Validation failures
- Timeout errors

We need a consistent, user-friendly approach to error handling that:
- Provides clear, actionable error messages
- Enables graceful degradation
- Supports retry strategies
- Maintains audit trails even during failures

## Decision
Implement a hierarchical error handling strategy with custom exceptions, automatic retries, and graceful degradation.

### 1. Custom Exception Hierarchy
```python
SifakaError
├── ConfigurationError     # Invalid configuration
├── ValidationError        # Text validation failures
├── LLMError              # LLM-related errors
│   ├── RateLimitError    # Rate limit exceeded
│   ├── TokenLimitError   # Context too long
│   ├── APIError          # API communication errors
│   └── ResponseError     # Invalid/unparseable response
├── StorageError          # Storage backend errors
├── PluginError           # Plugin loading/execution
└── TimeoutError          # Operation timeout
```

### 2. Error Context and Suggestions
Every error includes:
- Clear description of what went wrong
- Contextual information (iteration, critic, text snippet)
- Actionable suggestions for resolution
- Link to relevant documentation

```python
raise TokenLimitError(
    f"Text exceeds token limit for {model}",
    context={
        "text_length": len(text),
        "token_count": token_count,
        "limit": model_limit,
        "iteration": current_iteration
    },
    suggestions=[
        "Use a model with larger context window",
        "Enable text chunking with ChunkingMiddleware",
        "Reduce text length before processing"
    ],
    docs_url="https://docs.sifaka.ai/errors/token-limit"
)
```

### 3. Retry Strategy
Automatic retries with exponential backoff for transient errors:

```python
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=60),
    before_sleep=log_retry_attempt
)
async def call_llm(prompt: str) -> str:
    # LLM call implementation
```

### 4. Graceful Degradation
When errors occur, preserve partial results:

```python
try:
    result = await improve_iteration(text)
except SifakaError as e:
    # Return best result so far
    return SifakaResult(
        final_text=best_text_so_far,
        iterations=completed_iterations,
        error=e,
        partial=True,
        metadata={"error_details": e.context}
    )
```

### 5. Error Recovery Patterns

#### Validation Failures
```python
for validator in validators:
    try:
        validation = await validator.validate(text)
        if not validation.is_valid:
            # Try to fix with a fixer critic
            text = await fix_validation_issues(text, validation)
    except ValidationError:
        # Skip this validator, log warning
        logger.warning(f"Validator {validator} failed, skipping")
```

#### API Failures
```python
providers = ["openai", "anthropic", "google"]
for provider in providers:
    try:
        return await call_provider(provider, prompt)
    except APIError:
        continue  # Try next provider
raise APIError("All providers failed")
```

## Consequences

### Positive
- **User-Friendly**: Clear errors with actionable suggestions
- **Resilient**: Automatic retries and fallbacks
- **Debuggable**: Rich context for troubleshooting
- **Graceful**: Partial results better than complete failure
- **Consistent**: Uniform error handling across codebase

### Negative
- **Complexity**: More code for error handling
- **Performance**: Retries can increase latency
- **Masking Issues**: Automatic recovery might hide problems
- **Cost**: Retries increase API costs

### Mitigation Strategies

1. **Configurable Retries**: Let users disable/customize retry behavior
2. **Error Metrics**: Track all errors for monitoring
3. **Circuit Breakers**: Stop retrying after repeated failures
4. **Cost Limits**: Don't retry if cost budget exceeded
5. **Debug Mode**: Fail fast in debug mode

## Implementation Guidelines

### Do's
- Always include context in errors
- Provide helpful suggestions
- Log errors with appropriate levels
- Preserve partial results
- Test error paths explicitly

### Don'ts
- Don't swallow errors silently
- Don't retry non-retryable errors
- Don't expose sensitive data in errors
- Don't use generic error messages
- Don't ignore error patterns

### Error Message Templates
```python
# Good
"Failed to improve text: OpenAI API returned 429 (rate limit). Wait 60s or switch providers."

# Bad
"Error occurred"
"Something went wrong"
"API failed"
```

## Monitoring Integration
Track error metrics:
- Error rate by type
- Retry success rate
- Provider failure rate
- Recovery time
- Cost of retries

## References
- [Python Exception Best Practices](https://docs.python.org/3/tutorial/errors.html)
- [Resilient API Design](https://aws.amazon.com/builders-library/implementing-health-checks/)
- [Error Handling in Distributed Systems](https://martinfowler.com/articles/microservices.html#DesignForFailure)
