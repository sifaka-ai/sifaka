# Error Handling Guide

This document provides guidance on error handling patterns, common error types, and recovery strategies in Sifaka.

## Error Handling Principles

Sifaka follows these core principles for error handling:

1. **Fail Gracefully**: Components should handle errors internally when possible and return valid results.
2. **Provide Context**: Error messages should include context about what failed and why.
3. **Enable Recovery**: When possible, provide information that helps with recovery.
4. **Preserve State**: Errors should not leave components in an invalid state.
5. **Log Details**: Log detailed error information for debugging while returning appropriate user-facing errors.

## Common Error Types

### ValidationError

ValidationError occurs when input validation fails.

```python
from sifaka.classifiers import ToxicityClassifier

classifier = ToxicityClassifier()

try:
    classifier.validate_input(123)  # Not a string
except ValueError as e:
    print(f"Validation error: {e}")
    # Handle the error or propagate it
```

### ModelError

ModelError occurs when a model fails to load or generate output.

```python
from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelError

model = AnthropicProvider(model="claude-3-sonnet")

try:
    response = model.generate("Generate a short poem")
except ModelError as e:
    print(f"Model error: {e.message}")
    print(f"Error type: {e.error_type}")
    print(f"Recovery possible: {e.can_retry}")

    if e.can_retry:
        # Wait and retry with backoff
        # ...
    else:
        # Use fallback approach
        # ...
```

### ClassificationError

ClassificationError occurs when classification fails.

```python
from sifaka.classifiers import SentimentClassifier

classifier = SentimentClassifier()

try:
    result = classifier.classify("Some text to analyze")
except Exception as e:
    print(f"Classification failed: {e}")
    # Use a fallback classification
    fallback_result = ClassificationResult(
        label="unknown",
        confidence=0.0,
        metadata={"error": str(e)}
    )
```

Note: Most classifiers handle errors internally and return valid results with low confidence rather than raising exceptions.

### AdapterError

AdapterError occurs when an adapter fails to integrate with an external system.

```python
from sifaka.adapters.langchain import LangChainAdapter
from sifaka.adapters.base import AdapterError

try:
    adapter = LangChainAdapter.create_chain(model)
except AdapterError as e:
    print(f"Adapter initialization failed: {e}")
    print(f"External system: {e.system}")
    print(f"Context: {e.context}")
```

## Error Handling Patterns

### Pattern 1: Try/Except with Fallback

This pattern handles errors by providing a fallback mechanism.

```python
from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelError

primary_model = AnthropicProvider(model="claude-3-sonnet")
fallback_model = OpenAIProvider(model="gpt-3.5-turbo")

def generate_with_fallback(prompt):
    try:
        return primary_model.generate(prompt)
    except ModelError as e:
        print(f"Primary model failed: {e}")
        try:
            return fallback_model.generate(prompt)
        except ModelError as fallback_error:
            print(f"Fallback model also failed: {fallback_error}")
            return {"text": "Unable to generate response", "error": str(e)}
```

### Pattern 2: Result-Based Error Handling

This pattern uses result objects rather than exceptions for error handling.

```python
from sifaka.classifiers.toxicity import create_toxicity_classifier

classifier = create_toxicity_classifier()
result = classifier.classify("Some text to analyze")

if result.confidence < classifier.min_confidence:
    # Handle low confidence result
    print(f"Low confidence classification: {result.label}")
    print(f"Confidence: {result.confidence}")
    print(f"Error info: {result.metadata.get('error')}")

    # Take appropriate action based on the result
    if "error" in result.metadata:
        # Handle error case
        pass
    elif result.confidence > 0:
        # Use result but with caution
        pass
    else:
        # Discard result completely
        pass
```

### Pattern 3: Retry with Backoff

This pattern retries operations with exponential backoff.

```python
import time
import random
from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelError, RateLimitError

model = AnthropicProvider(model="claude-3-sonnet")

def generate_with_retry(prompt, max_retries=3, base_delay=1.0):
    retries = 0
    while retries < max_retries:
        try:
            return model.generate(prompt)
        except RateLimitError as e:
            retries += 1
            if retries >= max_retries:
                raise

            # Calculate backoff delay with jitter
            delay = base_delay * (2 ** retries) + random.uniform(0, 0.5)
            print(f"Rate limited, retrying in {delay:.2f} seconds...")
            time.sleep(delay)
        except ModelError as e:
            # Only retry specific error types that might be transient
            if e.can_retry and retries < max_retries:
                retries += 1
                delay = base_delay * (2 ** retries)
                print(f"Model error, retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                raise
```

## Error Recovery Strategies

### Strategy 1: Graceful Degradation

When a component fails, fall back to simpler alternatives.

```python
from sifaka.classifiers.toxicity import create_toxicity_classifier
from sifaka.rules.content.toxicity import create_toxicity_rule

def check_toxicity(text):
    try:
        # Try the ML-based classifier first
        classifier = create_toxicity_classifier()
        classifier.warm_up()  # Might fail if model can't be loaded
        result = classifier.classify(text)
        return result
    except Exception as e:
        print(f"ML classifier failed: {e}")

        # Fall back to simpler rule-based approach
        try:
            rule = create_toxicity_rule(threshold=0.5)
            result = rule.validate(text)
            return ClassificationResult(
                label="toxic" if not result.passed else "non_toxic",
                confidence=0.6,
                metadata={"fallback": True, "rule_result": result.metadata}
            )
        except Exception as fallback_error:
            print(f"Rule-based fallback also failed: {fallback_error}")

            # Ultimate fallback: very basic keyword check
            bad_words = ["terrible", "hate", "stupid"]
            has_bad_words = any(word in text.lower() for word in bad_words)
            return ClassificationResult(
                label="toxic" if has_bad_words else "non_toxic",
                confidence=0.3,
                metadata={"fallback": True, "fallback_level": 2}
            )
```

### Strategy 2: Circuit Breaker

Prevent cascading failures by temporarily disabling components that consistently fail.

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=3, reset_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.open = False

    def execute(self, function, fallback, *args, **kwargs):
        # Check if circuit is open
        current_time = time.time()
        if self.open:
            # Check if we should attempt reset
            if current_time - self.last_failure_time > self.reset_timeout:
                # Half-open state: allow one request through
                try:
                    result = function(*args, **kwargs)
                    # Success! Reset circuit
                    self.open = False
                    self.failure_count = 0
                    return result
                except Exception as e:
                    # Still failing, keep circuit open
                    self.last_failure_time = current_time
                    return fallback(*args, error=e, **kwargs)
            else:
                # Circuit still open, use fallback
                return fallback(*args, error=None, **kwargs)

        # Circuit closed, try normal execution
        try:
            return function(*args, **kwargs)
        except Exception as e:
            # Function failed
            self.failure_count += 1
            self.last_failure_time = current_time

            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                self.open = True

            # Use fallback
            return fallback(*args, error=e, **kwargs)

# Usage example
def fallback_classifier(text, error=None, **kwargs):
    return ClassificationResult(
        label="unknown",
        confidence=0.0,
        metadata={"error": str(error) if error else "circuit open"}
    )

# Create circuit breaker for classifier
classifier_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=60)

# Use with classifier
result = classifier_breaker.execute(
    classifier.classify,
    fallback_classifier,
    "Text to classify"
)
```

### Strategy 3: Bulkhead Pattern

Isolate components to prevent failures in one from affecting others.

```python
from concurrent.futures import ThreadPoolExecutor

class Bulkhead:
    def __init__(self, max_concurrent=5, timeout=10):
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.timeout = timeout

    def execute(self, function, fallback, *args, **kwargs):
        future = self.executor.submit(function, *args, **kwargs)
        try:
            return future.result(timeout=self.timeout)
        except Exception as e:
            # Function failed or timed out
            return fallback(*args, error=e, **kwargs)

# Usage example
bulkhead = Bulkhead(max_concurrent=3, timeout=5)

def fallback_generation(prompt, error=None, **kwargs):
    return {"text": "Service unavailable", "error": str(error) if error else "timeout"}

# Use with model
response = bulkhead.execute(
    model.generate,
    fallback_generation,
    "Generate a response to this prompt"
)
```

## Logging and Debugging Errors

Sifaka includes enhanced logging utilities to help with error diagnosis.

```python
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

try:
    # Risky operation
    result = classifier.classify(text)
except Exception as e:
    # Log detailed error information for debugging
    logger.error(
        "Classification failed",
        exc_info=True,  # Include stack trace
        extra={
            "text": text[:100],  # First 100 chars of input
            "classifier": classifier.name,
            "error_type": type(e).__name__
        }
    )

    # Return simplified error to caller
    return ClassificationResult(
        label="error",
        confidence=0.0,
        metadata={"error": str(e)}
    )
```

## Error Prevention

### Input Validation

Validate inputs early to prevent downstream errors.

```python
def process_content(text):
    # Validate input
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    if not text.strip():
        return {"result": "empty", "message": "Input is empty"}

    # Process valid input
    # ...
```

### Configuration Validation

Validate configuration before component initialization.

```python
from sifaka.classifiers.base import ClassifierConfig

try:
    config = ClassifierConfig(
        labels=["positive", "negative"],
        min_confidence=1.5  # Invalid: must be between 0 and 1
    )
except ValueError as e:
    print(f"Configuration error: {e}")
    # Use default configuration instead
    config = ClassifierConfig(labels=["positive", "negative"])
```

### Resource Management

Ensure proper resource acquisition and release.

```python
from sifaka.models.anthropic import AnthropicProvider
import contextlib

@contextlib.contextmanager
def managed_model():
    model = AnthropicProvider(model="claude-3-sonnet")
    try:
        yield model
    finally:
        # Release any resources if needed
        model.cleanup()

# Use with context manager
with managed_model() as model:
    response = model.generate("Hello")
    print(response["text"])
# Model resources are automatically cleaned up
```

## Error Handling in Adapters

Adapters have special error handling requirements since they bridge between Sifaka and external systems.

```python
from sifaka.adapters.langchain import LangChainAdapter
from sifaka.models.anthropic import AnthropicProvider
from sifaka.adapters.base import AdapterError

try:
    model = AnthropicProvider(model="claude-3-sonnet")
    adapter = LangChainAdapter.create_chain(model)

    # Use adapter
    result = adapter.invoke({"input": "Generate a summary"})
except ImportError as e:
    # LangChain not installed
    print(f"LangChain integration unavailable: {e}")
    print("Install with: pip install langchain")
except AdapterError as e:
    # Adapter-specific error
    print(f"Adapter error: {e}")
    if "translation" in str(e).lower():
        print("Check compatibility between model and adapter")
except Exception as e:
    # Other unexpected errors
    print(f"Unexpected error: {e}")
```

## Summary

Effective error handling in Sifaka involves:

1. **Validating inputs** early to prevent downstream errors
2. **Using appropriate patterns** like fallbacks, retries, and circuit breakers
3. **Logging detailed error information** for debugging
4. **Returning meaningful errors** to callers
5. **Implementing recovery strategies** for different failure modes

By following these guidelines, your applications will be more resilient and provide better user experiences even when things go wrong.