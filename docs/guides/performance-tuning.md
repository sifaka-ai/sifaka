# Performance Tuning Guide

Optimize your Sifaka applications for speed, efficiency, and scalability with these performance tuning strategies.

## Overview

Sifaka performance depends on several factors:
- **Model selection and configuration**
- **Storage backend optimization**
- **Async vs sync execution**
- **Validation and criticism efficiency**
- **Memory and resource management**

## Model Performance

### Choose the Right Model

```python
from sifaka.models import create_model

# Fast models for development/testing
fast_model = create_model("openai:gpt-3.5-turbo")  # Faster than GPT-4
dev_model = create_model("mock:test-model")         # Instant responses

# Balanced models for production
balanced_model = create_model("openai:gpt-4")       # Good quality/speed balance
claude_model = create_model("anthropic:claude-3-haiku")  # Fast Claude variant

# High-quality models for critical tasks
quality_model = create_model("openai:gpt-4-turbo")  # Latest GPT-4
claude_opus = create_model("anthropic:claude-3-opus")  # Highest quality Claude
```

### Optimize Model Parameters

```python
# Reduce max_tokens for faster responses
model = create_model("openai:gpt-4", max_tokens=500)  # Instead of 2000+

# Use appropriate temperature
model = create_model("openai:gpt-4", temperature=0.3)  # Lower = faster, more deterministic

# Configure timeouts
model = create_model("openai:gpt-4", timeout=30)  # Prevent hanging requests
```

### Batch Processing

```python
from sifaka import Chain
from concurrent.futures import ThreadPoolExecutor
import asyncio

async def process_batch(prompts: list, model):
    """Process multiple prompts concurrently."""
    chains = [Chain(model=model, prompt=prompt) for prompt in prompts]
    
    # Run chains concurrently
    tasks = [chain.run_async() for chain in chains]
    results = await asyncio.gather(*tasks)
    
    return results

# Usage
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
results = asyncio.run(process_batch(prompts, model))
```

## Storage Optimization

### Choose the Right Storage Backend

```python
from sifaka.storage import MemoryStorage, RedisStorage, MilvusStorage, CachedStorage

# Development: Memory storage (fastest)
dev_storage = MemoryStorage()

# Production: Redis for caching
prod_storage = RedisStorage(redis_config, ttl=3600)

# Semantic search: Milvus with optimized index
vector_storage = MilvusStorage(
    milvus_config,
    collection_name="thoughts",
    index_type="HNSW",  # Faster than IVF_FLAT for queries
    metric_type="COSINE"
)

# Best of both: 3-tier storage
optimal_storage = CachedStorage(
    cache=MemoryStorage(max_size=1000),  # L1: Fast access
    persistence=CachedStorage(
        cache=RedisStorage(redis_config, ttl=7200),  # L2: Session cache
        persistence=vector_storage  # L3: Long-term storage
    )
)
```

### Storage Configuration

```python
# Memory storage with limits
memory_storage = MemoryStorage(
    max_size=5000,  # Limit memory usage
    eviction_policy="lru"  # Remove least recently used items
)

# Redis with connection pooling
redis_storage = RedisStorage(
    redis_config,
    max_connections=20,  # Connection pool size
    retry_on_timeout=True,
    key_prefix="sifaka:fast:",
    ttl=1800  # 30 minutes
)

# Milvus with optimized parameters
milvus_storage = MilvusStorage(
    milvus_config,
    collection_name="fast_thoughts",
    dimension=768,
    index_type="HNSW",
    M=16,  # Lower M = faster build, higher M = better recall
    efConstruction=200,  # Higher = better quality, slower build
    ef=100  # Query parameter: higher = better recall, slower query
)
```

## Async vs Sync Performance

### Use Async for Better Concurrency

```python
import asyncio
from sifaka import Chain

# Sync version (slower for multiple operations)
def sync_processing():
    results = []
    for prompt in prompts:
        chain = Chain(model=model, prompt=prompt)
        result = chain.run()
        results.append(result)
    return results

# Async version (faster for multiple operations)
async def async_processing():
    chains = [Chain(model=model, prompt=prompt) for prompt in prompts]
    tasks = [chain.run_async() for chain in chains]
    results = await asyncio.gather(*tasks)
    return results

# Async is significantly faster for multiple operations
results = asyncio.run(async_processing())
```

### Mixed Async/Sync Patterns

```python
# Use async for I/O-bound operations
async def optimized_chain():
    chain = Chain(model=model, prompt="Your prompt")
    
    # Async model generation
    result = await chain.run_async()
    
    # Sync validation (CPU-bound, fast)
    validation_results = chain.validate_sync(result)
    
    return result

# Batch async operations
async def batch_optimize():
    tasks = [optimized_chain() for _ in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
```

## Validation and Criticism Optimization

### Efficient Validator Selection

```python
from sifaka.validators import LengthValidator, RegexValidator
from sifaka.validators.classifier import ClassifierValidator

# Fast validators (CPU-bound, milliseconds)
fast_validators = [
    LengthValidator(min_length=50, max_length=1000),  # Very fast
    RegexValidator(patterns=[r'\b\w+@\w+\.\w+\b'])    # Fast regex
]

# Slower validators (ML-based, seconds)
slow_validators = [
    ClassifierValidator(classifier=toxicity_classifier),  # ML inference
    ClassifierValidator(classifier=sentiment_classifier)  # ML inference
]

# Optimize by running fast validators first
chain = Chain(model=model, prompt="Your prompt")
for validator in fast_validators:
    chain.validate_with(validator)

# Only run slow validators if fast ones pass
if all(result.passed for result in chain.validation_results.values()):
    for validator in slow_validators:
        chain.validate_with(validator)
```

### Concurrent Validation

```python
import asyncio
from sifaka.validators.base import LengthValidator

class AsyncLengthValidator(LengthValidator):
    """Async version of length validator."""
    
    async def validate_async(self, thought):
        """Async validation (useful for I/O-bound validators)."""
        # For CPU-bound validators, just call sync version
        return self.validate(thought)

# Run validators concurrently
async def concurrent_validation(thought, validators):
    tasks = [validator.validate_async(thought) for validator in validators]
    results = await asyncio.gather(*tasks)
    return dict(zip([v.name for v in validators], results))
```

### Efficient Critics

```python
from sifaka.critics import ReflexionCritic, SelfRefineCritic

# Choose efficient critics
fast_critic = SelfRefineCritic(model=fast_model)  # Use faster model for criticism
quality_critic = ReflexionCritic(model=quality_model)  # Use quality model sparingly

# Conditional criticism
chain = Chain(model=model, prompt="Your prompt")

# Only apply expensive critics if validation fails
if not all(result.passed for result in chain.validation_results.values()):
    chain.improve_with(fast_critic)  # Try fast improvement first
    
    # Only use expensive critic if fast one doesn't work
    if not chain.validation_passed:
        chain.improve_with(quality_critic)
```

## Memory Management

### Monitor Memory Usage

```python
import psutil
from sifaka.utils.performance import memory_usage

def monitor_memory():
    """Monitor memory usage during chain execution."""
    initial_memory = memory_usage()
    
    # Your chain operations
    chain = Chain(model=model, prompt="Your prompt")
    result = chain.run()
    
    final_memory = memory_usage()
    memory_delta = final_memory.current_mb - initial_memory.current_mb
    
    print(f"Memory used: {memory_delta:.1f} MB")
    print(f"Peak memory: {final_memory.peak_mb:.1f} MB")
    
    return result

# Usage
result = monitor_memory()
```

### Memory-Efficient Patterns

```python
# Clear thought history for long-running processes
def memory_efficient_processing(prompts):
    results = []
    
    for prompt in prompts:
        chain = Chain(model=model, prompt=prompt)
        result = chain.run()
        
        # Clear history to save memory
        result.history = None
        results.append(result)
        
        # Explicit garbage collection for large batches
        if len(results) % 100 == 0:
            import gc
            gc.collect()
    
    return results

# Use generators for large datasets
def process_large_dataset(prompts):
    """Generator that yields results without storing all in memory."""
    for prompt in prompts:
        chain = Chain(model=model, prompt=prompt)
        yield chain.run()

# Process without loading all results into memory
for result in process_large_dataset(large_prompt_list):
    # Process each result individually
    save_result(result)
```

## Caching Strategies

### Model Response Caching

```python
from functools import lru_cache
import hashlib

class CachedModel:
    """Model wrapper with response caching."""
    
    def __init__(self, model):
        self.model = model
        self.cache = {}
    
    def generate(self, prompt, **options):
        # Create cache key from prompt and options
        cache_key = hashlib.md5(
            f"{prompt}{sorted(options.items())}".encode()
        ).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate and cache result
        result = self.model.generate(prompt, **options)
        self.cache[cache_key] = result
        return result

# Usage
cached_model = CachedModel(model)
chain = Chain(model=cached_model, prompt="Your prompt")
```

### Validation Result Caching

```python
class CachedValidator:
    """Validator with result caching."""
    
    def __init__(self, validator):
        self.validator = validator
        self.cache = {}
    
    def validate(self, thought):
        text_hash = hashlib.md5(thought.text.encode()).hexdigest()
        
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        result = self.validator.validate(thought)
        self.cache[text_hash] = result
        return result

# Usage
cached_validator = CachedValidator(LengthValidator(min_length=50))
```

## Profiling and Monitoring

### Performance Profiling

```python
import time
from sifaka.utils.performance import time_operation

# Profile individual operations
with time_operation("model_generation") as timer:
    result = model.generate("Your prompt")
print(f"Generation took: {timer.elapsed:.2f}s")

# Profile entire chain
with time_operation("full_chain") as timer:
    chain = Chain(model=model, prompt="Your prompt")
    result = chain.run()
print(f"Full chain took: {timer.elapsed:.2f}s")

# Profile with detailed breakdown
def profile_chain():
    timings = {}
    
    with time_operation("initialization") as timer:
        chain = Chain(model=model, prompt="Your prompt")
    timings["init"] = timer.elapsed
    
    with time_operation("generation") as timer:
        result = chain.run()
    timings["generation"] = timer.elapsed
    
    return result, timings

result, timings = profile_chain()
print(f"Breakdown: {timings}")
```

### Continuous Monitoring

```python
import logging
from sifaka.utils.logging import get_logger

# Set up performance logging
perf_logger = get_logger("sifaka.performance")
perf_logger.setLevel(logging.INFO)

class PerformanceMonitor:
    """Monitor chain performance over time."""
    
    def __init__(self):
        self.metrics = []
    
    def monitor_chain(self, chain):
        start_time = time.time()
        result = chain.run()
        end_time = time.time()
        
        metrics = {
            "duration": end_time - start_time,
            "iterations": result.iteration,
            "text_length": len(result.text or ""),
            "validation_count": len(result.validation_results or {}),
            "memory_mb": memory_usage().current_mb
        }
        
        self.metrics.append(metrics)
        perf_logger.info(f"Chain performance: {metrics}")
        
        return result
    
    def get_average_metrics(self):
        if not self.metrics:
            return {}
        
        return {
            "avg_duration": sum(m["duration"] for m in self.metrics) / len(self.metrics),
            "avg_iterations": sum(m["iterations"] for m in self.metrics) / len(self.metrics),
            "avg_text_length": sum(m["text_length"] for m in self.metrics) / len(self.metrics)
        }

# Usage
monitor = PerformanceMonitor()
result = monitor.monitor_chain(chain)
print(f"Average metrics: {monitor.get_average_metrics()}")
```

## Production Optimization Checklist

### ✅ Model Configuration
- [ ] Use appropriate model size for your use case
- [ ] Set reasonable max_tokens limits
- [ ] Configure timeouts to prevent hanging
- [ ] Use faster models for non-critical operations

### ✅ Storage Configuration  
- [ ] Choose appropriate storage backend
- [ ] Configure connection pooling
- [ ] Set up 3-tier storage for optimal performance
- [ ] Monitor storage metrics

### ✅ Async Usage
- [ ] Use async for I/O-bound operations
- [ ] Batch concurrent operations
- [ ] Avoid blocking sync calls in async contexts

### ✅ Memory Management
- [ ] Monitor memory usage
- [ ] Clear unnecessary data
- [ ] Use generators for large datasets
- [ ] Configure garbage collection

### ✅ Caching
- [ ] Cache model responses for repeated prompts
- [ ] Cache validation results
- [ ] Use appropriate cache TTLs
- [ ] Monitor cache hit rates

### ✅ Monitoring
- [ ] Set up performance logging
- [ ] Profile critical paths
- [ ] Monitor resource usage
- [ ] Set up alerts for performance degradation

Following these optimization strategies will help you build fast, efficient, and scalable Sifaka applications!
