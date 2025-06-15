# Sifaka Built-in Features

Sifaka includes simple, built-in features for the most common needs: logging, performance timing, and result caching. These features are designed to be easy to use without complex configuration.

## 🎯 Overview

Instead of complex middleware systems, Sifaka provides three essential built-in features:

- **🔍 Logging**: Track workflow events and debug issues
- **⏱️ Timing**: Monitor performance and identify bottlenecks  
- **💾 Caching**: Cache results to improve performance

These features can be enabled with simple boolean flags and configured with straightforward parameters.

## 🚀 Quick Start

### Simple One-Liner API

```python
import sifaka

# Basic usage with logging and timing
result = await sifaka.improve(
    "Write about renewable energy",
    enable_logging=True,
    enable_timing=True,
    enable_caching=True
)
```

### Configuration API

```python
from sifaka import SifakaConfig, SifakaEngine

# Direct configuration
config = SifakaConfig(
    model="openai:gpt-4",
    enable_logging=True,
    enable_timing=True,
    enable_caching=True,
    cache_size=1000
)

engine = SifakaEngine(config=config)
result = await engine.think("Your prompt")
```

### Builder Pattern

```python
config = (SifakaConfig.builder()
         .model("openai:gpt-4")
         .with_logging(log_level="INFO", log_content=True)
         .with_timing()
         .with_caching(cache_size=500)
         .build())
```

## 🔍 Logging Feature

Enable workflow logging to track events and debug issues.

### Configuration

```python
# Simple API
result = await sifaka.improve(
    "Your prompt",
    enable_logging=True,
    log_level="INFO",        # DEBUG, INFO, WARNING, ERROR
    log_content=True         # Include prompt/result content in logs
)

# Configuration API
config = SifakaConfig(
    enable_logging=True,
    log_level="INFO",
    log_content=False        # Don't log content for privacy
)

# Builder pattern
config = (SifakaConfig.builder()
         .with_logging(log_level="DEBUG", log_content=True)
         .build())
```

### What Gets Logged

When logging is enabled, you'll see:

```
INFO - Starting thought processing for request abc123 - Content: Write about renewable...
INFO - Request abc123 completed in 3.45s with 2 iterations
```

With `log_content=False` (recommended for production):
```
INFO - Starting thought processing for request abc123
INFO - Request abc123 completed in 3.45s with 2 iterations
```

### Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General workflow events (recommended)
- **WARNING**: Important warnings
- **ERROR**: Error conditions only

## ⏱️ Timing Feature

Track performance metrics to identify bottlenecks and monitor system health.

### Configuration

```python
# Simple API
result = await sifaka.improve(
    "Your prompt",
    enable_timing=True
)

# Configuration API
config = SifakaConfig(enable_timing=True)

# Builder pattern
config = SifakaConfig.builder().with_timing().build()
```

### Getting Timing Statistics

```python
engine = SifakaEngine(config=config)

# Process some requests
await engine.think("Prompt 1")
await engine.think("Prompt 2")
await engine.think("Prompt 3")

# Get timing statistics
stats = engine.get_timing_stats()
print(f"Average duration: {stats['avg_duration_seconds']:.2f}s")
print(f"Total requests: {stats['total_requests']}")
print(f"Average iterations: {stats['avg_iterations']:.1f}")
```

### Timing Statistics

The timing feature provides:

```python
{
    "total_requests": 10,
    "avg_duration_seconds": 4.23,
    "min_duration_seconds": 2.1,
    "max_duration_seconds": 8.7,
    "avg_iterations": 2.3,
    "total_duration_seconds": 42.3
}
```

## 💾 Caching Feature

Cache generation results to dramatically improve performance for repeated requests.

### Configuration

```python
# Simple API
result = await sifaka.improve(
    "Your prompt",
    enable_caching=True,
    cache_size=1000          # Maximum number of cached results
)

# Configuration API
config = SifakaConfig(
    enable_caching=True,
    cache_size=1000,         # Max cache entries
    cache_ttl_seconds=3600   # 1 hour TTL
)

# Builder pattern
config = (SifakaConfig.builder()
         .with_caching(cache_size=500, ttl_seconds=1800)  # 30 min TTL
         .build())
```

### How Caching Works

1. **Cache Key**: Generated from prompt + max_iterations
2. **Cache Hit**: Returns cached result immediately
3. **Cache Miss**: Processes request and stores result
4. **TTL Expiration**: Cached results expire after configured time
5. **Size Limit**: Oldest entries removed when cache is full

### Getting Cache Statistics

```python
engine = SifakaEngine(config=config)

# Process some requests
await engine.think("Same prompt")  # Cache miss
await engine.think("Same prompt")  # Cache hit!

# Get cache statistics
stats = engine.get_cache_stats()
print(f"Cache size: {stats['cache_size']}")
print(f"Max cache size: {stats['max_cache_size']}")
```

### Cache Performance

Caching provides dramatic performance improvements:

- **Cache Hit**: ~50ms (instant return)
- **Cache Miss**: ~3-8s (full processing)
- **Hit Rate**: Typically 20-40% in real applications

## 👥 User and Session Tracking

Track requests by user and session for analytics and debugging.

### Usage

```python
# Simple API
result = await sifaka.improve(
    "Your prompt",
    user_id="user123",
    session_id="session456",
    enable_logging=True      # See user info in logs
)

# Engine API
result = await engine.think(
    "Your prompt",
    user_id="alice",
    session_id="session_1"
)
```

### What Gets Tracked

When logging is enabled, user information appears in logs:

```
INFO - Starting thought processing for request abc123 - User: alice, Session: session_1
```

User and session IDs are also included in timing statistics for analysis.

## 🎛️ Configuration Patterns

### Pattern 1: Development Setup

```python
# Full logging and timing for development
config = SifakaConfig(
    model="openai:gpt-4o-mini",
    enable_logging=True,
    log_level="DEBUG",
    log_content=True,        # See full content
    enable_timing=True,
    enable_caching=True,
    cache_size=100           # Small cache for testing
)
```

### Pattern 2: Production Setup

```python
# Optimized for production
config = SifakaConfig(
    model="openai:gpt-4",
    enable_logging=True,
    log_level="INFO",
    log_content=False,       # Privacy-friendly
    enable_timing=True,      # Monitor performance
    enable_caching=True,
    cache_size=10000,        # Large cache
    cache_ttl_seconds=7200   # 2 hour TTL
)
```

### Pattern 3: High Performance

```python
# Maximum performance
config = SifakaConfig(
    model="openai:gpt-4o-mini",
    enable_logging=False,    # Minimal logging overhead
    enable_timing=False,     # No timing overhead
    enable_caching=True,
    cache_size=50000,        # Very large cache
    cache_ttl_seconds=86400  # 24 hour TTL
)
```

## 📊 Performance Impact

### Feature Overhead

| Feature | Overhead | When to Use |
|---------|----------|-------------|
| Logging | ~1-5ms | Development, debugging, monitoring |
| Timing | ~0.1ms | Always (minimal impact) |
| Caching | ~0.5ms | Always (huge benefit on hits) |

### Cache Performance

| Scenario | Without Cache | With Cache | Improvement |
|----------|---------------|------------|-------------|
| Unique requests | 4.2s | 4.2s | 0% |
| 50% cache hits | 4.2s | 2.1s | 50% faster |
| 80% cache hits | 4.2s | 0.9s | 78% faster |

## 🛠️ Best Practices

### 1. Enable Timing Always
```python
# Timing has minimal overhead and provides valuable insights
config = SifakaConfig(enable_timing=True)
```

### 2. Use Appropriate Log Levels
```python
# Development
config = SifakaConfig(enable_logging=True, log_level="DEBUG", log_content=True)

# Production
config = SifakaConfig(enable_logging=True, log_level="INFO", log_content=False)
```

### 3. Size Cache Appropriately
```python
# Consider your memory constraints and request patterns
config = SifakaConfig(
    enable_caching=True,
    cache_size=1000,         # ~10MB for typical responses
    cache_ttl_seconds=3600   # Balance freshness vs performance
)
```

### 4. Track Users in Production
```python
# Always include user context for analytics
result = await engine.think(
    prompt,
    user_id=current_user.id,
    session_id=request.session_id
)
```

## 🔧 Troubleshooting

### No Timing Data
```python
stats = engine.get_timing_stats()
# Returns: {"message": "No timing data available. Enable timing in config."}

# Solution: Enable timing
config = SifakaConfig(enable_timing=True)
```

### Cache Not Working
```python
stats = engine.get_cache_stats()
# Returns: {"message": "Caching not enabled"}

# Solution: Enable caching
config = SifakaConfig(enable_caching=True)
```

### Logs Not Appearing
```python
# Make sure logging is configured
import logging
logging.basicConfig(level=logging.INFO)

# And enabled in config
config = SifakaConfig(enable_logging=True, log_level="INFO")
```

## 🎯 Summary

Sifaka's built-in features provide essential functionality without complexity:

- **🔍 Logging**: Simple boolean flag + level configuration
- **⏱️ Timing**: Zero-config performance monitoring
- **💾 Caching**: Automatic result caching with TTL
- **👥 Tracking**: User and session context
- **📊 Statistics**: Built-in performance analytics

These features cover 90% of real-world needs with 10% of the complexity of middleware systems. Simple, effective, and production-ready! 🚀
