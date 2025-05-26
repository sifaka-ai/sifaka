# Async/Sync Pattern Guidelines

This document establishes the design principles and implementation patterns for async/sync functionality in Sifaka. These guidelines ensure consistency across the codebase and provide clear direction for developers.

## Design Philosophy

### Core Principles

1. **Async-First for I/O Operations**: Components that perform I/O operations (model calls, storage, network requests) should have async implementations for better performance.

2. **Sync API for Simplicity**: Public APIs remain synchronous for ease of use and backward compatibility, with async implementations used internally.

3. **No Dual Interfaces**: We avoid maintaining both sync and async public APIs. Instead, we choose the best pattern for each component and use async internally where beneficial.

4. **Intentional Mixed Patterns**: Mixed async/sync patterns are intentional design choices, not compatibility compromises.

5. **Performance Over Purity**: We prioritize performance gains from concurrency over architectural purity.

## Component-Specific Guidelines

### Chain (Orchestrator)

**Pattern**: Sync public API with async internal implementation

```python
# ✅ Public API (Sync)
chain = Chain(model=model, prompt="Write a story")
result = chain.run()  # Internally uses async for concurrency

# ❌ Not supported (No async public API)
# result = await chain.run_async()
```

**Rationale**:
- Chains orchestrate multiple I/O operations that benefit from concurrency
- Internal async enables concurrent validation and criticism
- Sync public API maintains simplicity for users

**Implementation Pattern**:
- `run()` - Public sync method that calls internal async implementation
- `_run_async()` - Internal async implementation with concurrent operations
- `_execute_*_async()` - Internal async methods for specific operations

### Models (I/O Heavy)

**Pattern**: Sync public API with async internal implementation

```python
# ✅ Public API (Sync)
model = create_model("openai:gpt-4")
text = model.generate("Write a story")
text, prompt = model.generate_with_thought(thought)

# ✅ Internal async methods (used by Chain)
# text = await model._generate_async("Write a story")
# text, prompt = await model._generate_with_thought_async(thought)
```

**Rationale**:
- Model calls are I/O heavy and benefit from async
- Sync public API for ease of use
- Async internal methods enable concurrent operations in chains

**Implementation Pattern**:
- `generate()` - Public sync method
- `generate_with_thought()` - Public sync method
- `_generate_async()` - Internal async method (required by Model protocol)
- `_generate_with_thought_async()` - Internal async method (required by Model protocol)

### Validators (CPU Light)

**Pattern**: Sync public API with optional async internal implementation

```python
# ✅ Public API (Sync)
validator = LengthValidator(min_length=10, max_length=100)
result = validator.validate(thought)

# ✅ Internal async methods (used by Chain for concurrency)
# result = await validator._validate_async(thought)
```

**Rationale**:
- Most validation is CPU-bound and fast
- Sync public API for simplicity
- Async internal methods enable concurrent validation in chains
- For CPU-bound validators, async methods can just call sync versions

**Implementation Pattern**:
- `validate()` - Public sync method (required by Validator protocol)
- `_validate_async()` - Internal async method (optional, for concurrency)

### Critics (I/O Heavy)

**Pattern**: Sync public API with async internal implementation

```python
# ✅ Public API (Sync)
critic = ReflexionCritic(model=model)
feedback = critic.critique(thought)
improved_text = critic.improve(thought)

# ✅ Internal async methods (used by Chain)
# feedback = await critic._critique_async(thought)
```

**Rationale**:
- Critics use models for analysis, which are I/O heavy
- Sync public API for ease of use
- Async internal methods enable concurrent criticism in chains

**Implementation Pattern**:
- `critique()` - Public sync method (required by Critic protocol)
- `improve()` - Public sync method (required by Critic protocol)
- `_critique_async()` - Internal async method (optional, for concurrency)

### Storage (I/O Heavy)

**Pattern**: Sync public API with async internal implementation

```python
# ✅ Public API (Sync)
storage.set("key", value)
value = storage.get("key")
results = storage.search("query")

# ✅ Internal async methods
# await storage._set_async("key", value)
# value = await storage._get_async("key")
```

**Rationale**:
- Storage operations are I/O heavy
- Sync public API for simplicity
- Async internal methods for performance

**Implementation Pattern**:
- `get()`, `set()`, `search()`, `clear()` - Public sync methods
- `_get_async()`, `_set_async()`, `_search_async()`, `_clear_async()` - Internal async methods

### Retrievers (I/O Heavy)

**Pattern**: Sync public API (no async internal methods yet)

```python
# ✅ Public API (Sync)
retriever = MilvusRetriever()
documents = retriever.retrieve("query")
thought = retriever.retrieve_for_thought(thought)
```

**Rationale**:
- Retrieval operations are I/O heavy but currently synchronous
- Future enhancement: Add async internal methods for concurrent retrieval

## Naming Conventions

### Internal Async Methods

**Pattern**: `_method_name_async`

```python
# ✅ Correct naming
async def _generate_async(self, prompt: str) -> str: ...
async def _validate_async(self, thought: Thought) -> ValidationResult: ...
async def _critique_async(self, thought: Thought) -> Dict[str, Any]: ...
async def _run_async(self) -> Thought: ...

# ❌ Incorrect naming
async def generate_async(self, prompt: str) -> str: ...  # Public async (not supported)
async def async_generate(self, prompt: str) -> str: ...  # Wrong prefix
async def generateAsync(self, prompt: str) -> str: ...   # Wrong case
```

**Rules**:
1. Internal async methods MUST use `_method_name_async` pattern
2. The `_async` suffix clearly indicates internal async implementation
3. The leading underscore indicates internal/private method
4. No public async methods in current design

### Public Sync Methods

**Pattern**: Standard method names without async indicators

```python
# ✅ Correct public API
def run(self) -> Thought: ...
def generate(self, prompt: str) -> str: ...
def validate(self, thought: Thought) -> ValidationResult: ...
def critique(self, thought: Thought) -> Dict[str, Any]: ...

# ❌ Incorrect (no sync suffixes)
def run_sync(self) -> Thought: ...
def generate_sync(self, prompt: str) -> str: ...
```

## Protocol Definitions

### Required Async Methods in Protocols

Only components that benefit from async operations require async methods in their protocols:

**Model Protocol** (I/O Heavy):
```python
@runtime_checkable
class Model(Protocol):
    # Public sync methods
    def generate(self, prompt: str, **options: Any) -> str: ...
    def generate_with_thought(self, thought: "Thought", **options: Any) -> tuple[str, str]: ...
    def count_tokens(self, text: str) -> int: ...

    # Internal async methods (required for Chain concurrency)
    async def _generate_async(self, prompt: str, **options: Any) -> str: ...
    async def _generate_with_thought_async(self, thought: "Thought", **options: Any) -> tuple[str, str]: ...
```

**Validator Protocol** (CPU Light):
```python
@runtime_checkable
class Validator(Protocol):
    # Public sync method (required)
    def validate(self, thought: "Thought") -> "ValidationResult": ...

    # Internal async method (optional, for concurrency)
    # Not in protocol to avoid forcing all validators to implement it
```

**Critic Protocol** (I/O Heavy):
```python
@runtime_checkable
class Critic(Protocol):
    # Public sync methods (required)
    def critique(self, thought: "Thought") -> Dict[str, Any]: ...
    def improve(self, thought: "Thought") -> str: ...

    # Internal async method (optional, for concurrency)
    # Not in protocol to avoid forcing all critics to implement it
```

## Implementation Guidelines

### When to Add Async Internal Methods

**Always Add For**:
- Model implementations (I/O heavy)
- Storage backends (I/O heavy)
- Network-based operations

**Consider Adding For**:
- Critics that use models (I/O heavy)
- Validators that use external services
- Complex CPU-bound operations that could benefit from thread pools

**Don't Add For**:
- Simple CPU-bound validators (length, regex)
- In-memory operations
- Configuration and setup methods

### Async Implementation Patterns

**Pattern 1: I/O Heavy Operations**
```python
class OpenAIModel:
    def generate(self, prompt: str) -> str:
        """Public sync method."""
        return asyncio.run(self._generate_async(prompt))

    async def _generate_async(self, prompt: str) -> str:
        """Internal async implementation with real async I/O."""
        async with aiohttp.ClientSession() as session:
            # Real async HTTP call
            response = await session.post(...)
            return response.text
```

**Pattern 2: CPU-Bound Operations**
```python
class LengthValidator:
    def validate(self, thought: Thought) -> ValidationResult:
        """Public sync method."""
        # Direct implementation - no need to call async
        return ValidationResult(...)

    async def _validate_async(self, thought: Thought) -> ValidationResult:
        """Internal async method for concurrency."""
        # For CPU-bound operations, just call sync version
        return self.validate(thought)
```

**Pattern 3: Mixed Operations**
```python
class ReflexionCritic:
    def critique(self, thought: Thought) -> Dict[str, Any]:
        """Public sync method."""
        return asyncio.run(self._critique_async(thought))

    async def _critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Internal async implementation."""
        # Use model's async method for I/O
        critique_text = await self.model._generate_async(prompt)
        # CPU-bound parsing
        return self._parse_critique(critique_text)
```

## Usage Examples

### Basic Sync Usage (Recommended for Most Users)

```python
from sifaka import Chain
from sifaka.models import create_model
from sifaka.validators import LengthValidator
from sifaka.critics import ReflexionCritic

# All sync API - simple and straightforward
model = create_model("openai:gpt-4")
validator = LengthValidator(min_length=50, max_length=500)
critic = ReflexionCritic(model=model)

chain = Chain(
    model=model,
    prompt="Write a story about AI",
    max_improvement_iterations=2
)
chain.validate_with(validator)
chain.improve_with(critic)

# Sync execution with internal async concurrency
result = chain.run()  # Internally uses async for performance
```

### Advanced Async Usage (For Framework Developers)

```python
import asyncio
from sifaka import Chain

async def run_multiple_chains():
    """Example of running multiple chains concurrently."""
    chains = [
        Chain(model=model, prompt=f"Write about topic {i}")
        for i in range(5)
    ]

    # Run multiple chains concurrently using internal async methods
    tasks = [chain._run_async() for chain in chains]
    results = await asyncio.gather(*tasks)
    return results

# This is advanced usage - most users should stick to sync API
results = asyncio.run(run_multiple_chains())
```

## Migration Guidelines

### For Existing Code

**No Changes Required**: All existing sync code continues to work unchanged.

```python
# ✅ This continues to work exactly as before
chain = Chain(model=model, prompt="Write a story")
result = chain.run()
```

### For New Implementations

**Models**: Must implement both sync and async methods
```python
class CustomModel:
    def generate(self, prompt: str) -> str:
        # Sync implementation
        pass

    async def _generate_async(self, prompt: str) -> str:
        # Async implementation (required by Model protocol)
        pass
```

**Validators**: Sync method required, async optional
```python
class CustomValidator:
    def validate(self, thought: Thought) -> ValidationResult:
        # Sync implementation (required)
        pass

    async def _validate_async(self, thought: Thought) -> ValidationResult:
        # Async implementation (optional, for concurrency)
        return self.validate(thought)  # Can just call sync version
```

**Critics**: Sync methods required, async optional
```python
class CustomCritic:
    def critique(self, thought: Thought) -> Dict[str, Any]:
        # Sync implementation (required)
        pass

    def improve(self, thought: Thought) -> str:
        # Sync implementation (required)
        pass

    async def _critique_async(self, thought: Thought) -> Dict[str, Any]:
        # Async implementation (optional, for concurrency)
        return self.critique(thought)  # Can just call sync version
```

## Performance Considerations

### Benefits of Internal Async

1. **Concurrent Validation**: Multiple validators run simultaneously
2. **Concurrent Criticism**: Multiple critics analyze text in parallel
3. **Non-blocking I/O**: Model calls don't block other operations
4. **Better Resource Utilization**: CPU and I/O operations can overlap

### Performance Comparison

```python
# Sync execution (sequential)
# Validator 1: 100ms
# Validator 2: 150ms
# Validator 3: 200ms
# Total: 450ms

# Async execution (concurrent)
# All validators: max(100ms, 150ms, 200ms) = 200ms
# Speedup: 2.25x for validation phase
```

### When Async Doesn't Help

- Single validator/critic scenarios
- CPU-bound operations without I/O
- Very fast operations (< 10ms)
- Memory-only storage backends

## Testing Async Functionality

### Testing Internal Async Methods

```python
import asyncio
import pytest

class TestAsyncMethods:
    @pytest.mark.asyncio
    async def test_model_async_generation(self):
        """Test model's internal async method."""
        model = create_model("openai:gpt-4")
        result = await model._generate_async("Test prompt")
        assert isinstance(result, str)

    def test_chain_concurrent_validation(self):
        """Test that chain uses async for concurrent validation."""
        validators = [
            LengthValidator(min_length=10),
            RegexValidator(pattern=r"\w+"),
            ContentValidator(prohibited=["bad"])
        ]

        chain = Chain(model=model, prompt="Test")
        for validator in validators:
            chain.validate_with(validator)

        # This internally uses async for concurrent validation
        result = chain.run()
        assert len(result.validation_results) == 3
```

### Performance Testing

```python
import time

def test_concurrent_performance():
    """Test that concurrent execution is faster than sequential."""
    # Setup multiple slow validators
    validators = [SlowValidator(delay=0.1) for _ in range(5)]

    chain = Chain(model=model, prompt="Test")
    for validator in validators:
        chain.validate_with(validator)

    start_time = time.time()
    result = chain.run()  # Uses concurrent validation internally
    execution_time = time.time() - start_time

    # Should be much faster than 5 * 0.1 = 0.5 seconds
    assert execution_time < 0.3  # Allow for overhead
```

## Troubleshooting

### Common Issues

**Issue**: `RuntimeError: asyncio.run() cannot be called from a running event loop`
```python
# ❌ Problem: Calling sync method from async context
async def my_function():
    model = create_model("openai:gpt-4")
    result = model.generate("prompt")  # This calls asyncio.run() internally

# ✅ Solution: Use internal async methods
async def my_function():
    model = create_model("openai:gpt-4")
    result = await model._generate_async("prompt")
```

**Issue**: Async method not found
```python
# ❌ Problem: Not all components have async methods
validator = LengthValidator()
result = await validator._validate_async(thought)  # May not exist

# ✅ Solution: Check if async method exists
if hasattr(validator, '_validate_async'):
    result = await validator._validate_async(thought)
else:
    result = validator.validate(thought)
```

### Debugging Async Issues

1. **Enable Debug Logging**: Set log level to DEBUG to see async operation details
2. **Check Event Loop**: Ensure you're not mixing sync/async incorrectly
3. **Use Type Hints**: Proper typing helps catch async/sync mismatches
4. **Test Concurrency**: Verify that concurrent operations actually improve performance

## Future Considerations

### Planned Enhancements

1. **Retriever Async Methods**: Add async support for retrieval operations
2. **Storage Async Optimization**: Optimize storage backends for async operations
3. **Streaming Support**: Add async streaming for long-running model calls
4. **Batch Operations**: Add async batch processing for multiple inputs

### Backward Compatibility

- All sync APIs will remain stable
- Internal async methods may evolve
- New async features will be additive, not breaking
- Migration guides will be provided for major changes

---

This document establishes the foundation for consistent async/sync patterns in Sifaka. All new code should follow these guidelines, and existing code should be gradually updated to match these patterns during regular maintenance.