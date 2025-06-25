# ADR-003: Memory Management Strategy

## Status
Accepted

## Context
Language models can generate large amounts of text, and iterative improvement compounds this issue. Without proper memory management:
- Long-running processes could consume unbounded memory
- Production deployments could crash due to OOM errors
- Cost tracking becomes unreliable
- Performance degrades as memory pressure increases

We need a strategy that balances complete audit trails with practical memory constraints.

## Decision
Implement a multi-layered memory management strategy:

### 1. Bounded History Collections
Use custom `BoundedList` and `BoundedDict` classes that automatically evict old entries when limits are reached:

```python
class BoundedList:
    def __init__(self, max_size: int, size_fn: Callable = len):
        self.max_size = max_size
        self.size_fn = size_fn

    def append(self, item):
        while self._current_size() + self.size_fn(item) > self.max_size:
            self._items.pop(0)  # FIFO eviction
```

### 2. Configurable Limits
Expose memory limits through the API:

```python
improve(
    text,
    memory_limits={
        "max_history_size": 100_000,  # characters
        "max_critique_length": 10_000,
        "max_text_length": 50_000,
        "max_iterations": 10
    }
)
```

### 3. Streaming for Large Operations
Implement streaming APIs that don't hold entire history in memory:

```python
async for event in improve_stream(text):
    # Process events as they arrive
    # Only current state in memory
    pass
```

### 4. Automatic Cleanup
- Clear intermediate results after each iteration
- Use weak references where appropriate
- Implement `__del__` methods for cleanup
- Provide explicit `cleanup()` methods

### 5. Memory Monitoring
Track memory usage and warn when approaching limits:

```python
class MemoryMonitor:
    def check_memory_usage(self):
        if self.current_usage > self.soft_limit:
            logger.warning(f"Memory usage high: {self.current_usage}MB")
        if self.current_usage > self.hard_limit:
            raise MemoryError("Memory limit exceeded")
```

## Consequences

### Positive
- **Predictable Memory Usage**: Bounded collections prevent unbounded growth
- **Production Stability**: No OOM crashes from long-running improvements
- **Performance**: Consistent performance regardless of iteration count
- **Flexibility**: Users can adjust limits based on their needs

### Negative
- **Data Loss**: Old history entries may be evicted
- **Complexity**: Additional code for memory management
- **Configuration**: Users need to understand memory settings
- **Debugging**: Harder to debug with partial history

### Mitigation Strategies

1. **Smart Eviction**: Keep most important entries (errors, final results)
2. **Persistence Option**: Allow saving full history to disk/database
3. **Clear Documentation**: Explain memory settings and trade-offs
4. **Sensible Defaults**: Choose defaults that work for most use cases
5. **Monitoring Tools**: Provide memory usage visibility

## Implementation Details

### Default Limits
- Max history size: 1MB
- Max critique length: 10KB
- Max text length: 100KB
- Max iterations: 10

### Eviction Strategy
1. FIFO for improvement history
2. Keep first and last iteration always
3. Preserve error states
4. Compact large texts (store hash + snippet)

### Monitoring Integration
```python
# Expose memory metrics
result.metadata["memory_stats"] = {
    "peak_usage_mb": 45.2,
    "history_entries_evicted": 3,
    "total_text_processed": 150000
}
```

## Testing Strategy

1. **Memory Leak Tests**: Run 1000+ iterations, verify stable memory
2. **Eviction Tests**: Verify correct entries are removed
3. **Performance Tests**: Ensure no degradation with memory management
4. **Integration Tests**: Test with real LLM responses

## References
- [Python Memory Management](https://docs.python.org/3/c-api/memory.html)
- [Bounded Collections in Production](https://engineering.linkedin.com/blog/2016/06/bounded-cache)
- [Memory-Efficient Python](https://realpython.com/python-memory-management/)
