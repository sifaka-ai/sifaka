# ADR-003: Memory Management and Result Storage

## Status
Accepted

## Context
Sifaka processes text through multiple iterations, with each iteration generating:
- Critiques from various critics
- Improved text versions
- Validation results
- Metadata and metrics

This can lead to significant memory usage, especially for:
- Long documents
- Multiple iterations
- Many critics
- Large metadata objects

We need a strategy to manage memory efficiently while maintaining functionality.

## Decision
We will implement a bounded memory management system with configurable limits and intelligent cleanup.

```python
# Configuration
config = Config(
    max_generations=10,      # Keep last 10 text versions
    max_critiques=50,        # Keep last 50 critiques
    max_validations=20,      # Keep last 20 validation results
    memory_limit_mb=100      # Total memory limit
)

# Automatic cleanup
result = await improve("text", config=config)
```

## Memory Management Strategy

### 1. Bounded Collections
Use fixed-size collections that automatically evict old items:
- `deque` with `maxlen` for generations and critiques
- LRU cache for expensive computations
- Configurable bounds for all collections

### 2. Lazy Loading
- Load large objects only when needed
- Use generators for large datasets
- Implement pagination for long histories

### 3. Garbage Collection
- Automatic cleanup of old iterations
- Reference counting for shared objects
- Periodic memory pressure checks

### 4. Storage Backends
Multiple storage options for different use cases:
- **Memory**: Fast, volatile, limited capacity
- **File**: Persistent, slower, unlimited capacity
- **Database**: Persistent, queryable, scalable

## Implementation Details

### Bounded Result Storage
```python
class SifakaResult:
    def __init__(self, max_generations=10, max_critiques=50):
        self.generations = deque(maxlen=max_generations)
        self.critiques = deque(maxlen=max_critiques)
        self.validations = deque(maxlen=20)
```

### Memory Monitoring
```python
class MemoryMonitor:
    def __init__(self, limit_mb=100):
        self.limit_bytes = limit_mb * 1024 * 1024

    def check_memory_usage(self, result: SifakaResult):
        if self.get_memory_usage() > self.limit_bytes:
            self.cleanup_old_data(result)
```

### Storage Abstraction
```python
class StorageBackend(ABC):
    @abstractmethod
    async def save(self, result: SifakaResult) -> str:
        pass

    @abstractmethod
    async def load(self, result_id: str) -> SifakaResult:
        pass
```

## Configuration Options

### Memory Limits
```python
config = Config(
    # Collection size limits
    max_generations=10,
    max_critiques=50,
    max_validations=20,

    # Memory limits
    memory_limit_mb=100,
    gc_threshold=0.8,  # Trigger cleanup at 80% usage

    # Storage options
    storage_backend="memory",  # "memory", "file", "redis"
    persistent_storage=True,
)
```

### Cleanup Strategies
- **Age-based**: Remove items older than N iterations
- **Size-based**: Remove items when collection exceeds limit
- **Memory-based**: Remove items when memory usage is high
- **Importance-based**: Keep important items longer

## Consequences

### Positive
- Predictable memory usage
- Configurable limits for different use cases
- Prevents out-of-memory errors
- Supports both development and production use
- Maintains performance under memory pressure

### Negative
- Some historical data may be lost
- Additional complexity in result management
- Potential performance impact from monitoring
- Configuration complexity for advanced users

### Mitigation
- Provide sensible defaults for all limits
- Allow unlimited collections for special cases
- Implement efficient storage backends
- Clear documentation of memory behavior
- Monitoring and alerting for memory issues

## Storage Backend Comparison

| Backend | Speed | Persistence | Scalability | Use Case |
|---------|-------|-------------|-------------|----------|
| Memory  | Fast  | No          | Limited     | Development, testing |
| File    | Medium| Yes         | Medium      | Single-user, persistence |
| Redis   | Fast  | Yes         | High        | Multi-user, production |
| Database| Medium| Yes         | High        | Enterprise, analytics |

## Memory Optimization Techniques

### 1. Lazy Evaluation
```python
# Don't compute expensive metrics until needed
@property
def similarity_score(self):
    if not hasattr(self, '_similarity_score'):
        self._similarity_score = self._compute_similarity()
    return self._similarity_score
```

### 2. Weak References
```python
# Use weak references for cached objects
import weakref
self._cache = weakref.WeakValueDictionary()
```

### 3. Compression
```python
# Compress large text objects
import gzip
self.compressed_text = gzip.compress(text.encode('utf-8'))
```

## Related Decisions
- [ADR-001: Single Function API](001-single-function-api.md)
- [ADR-002: Plugin Architecture](002-plugin-architecture.md)
- [ADR-004: Error Handling](004-error-handling.md)
