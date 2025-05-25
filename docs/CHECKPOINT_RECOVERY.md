# Chain Checkpoint Recovery System

The checkpoint recovery system saves execution state at key points during chain execution and provides mechanisms to resume from those points when failures occur.

## Overview

The system consists of three components:

1. **Checkpointing**: Saves execution state at major steps
2. **Recovery Manager**: Analyzes failures and suggests recovery options
3. **Recovery Actions**: Implements different strategies to resume execution

## How It Works

The system automatically creates checkpoints during chain execution:
- Before and after each major operation (retrieval, generation, validation, criticism)
- Stores the current `Thought` state and execution context
- Provides recovery options when failures occur
- Allows resuming from the most recent successful checkpoint

## Basic Usage

### Setting Up Checkpoint Storage

```python
from sifaka.storage.checkpoints import CachedCheckpointStorage
from sifaka.storage.cached import CachedStorage
from sifaka.storage.memory import InMemoryStorage

# Create storage backend
memory_storage = InMemoryStorage()
cached_storage = CachedStorage(memory_storage=memory_storage)
checkpoint_storage = CachedCheckpointStorage(cached_storage)
```

### Creating a Chain with Recovery

```python
from sifaka.chain import Chain
from sifaka.models.base import create_model

# Create chain with checkpoint storage
chain = Chain(
    model=create_model("anthropic:claude-3-sonnet"),
    checkpoint_storage=checkpoint_storage,
    max_improvement_iterations=3
)

# Configure chain as usual
chain.with_prompt("Write about AI ethics")
chain.validate_with(your_validator)
chain.improve_with(your_critic)

# Run with automatic recovery
result = chain.run_with_recovery()
```

### Manual Recovery Analysis

```python
from sifaka.recovery import RecoveryManager

# Create recovery manager
recovery_manager = RecoveryManager(checkpoint_storage)

# Analyze a failure and get suggestions
try:
    result = chain.run_with_recovery()
except Exception as e:
    suggestions = chain.get_recovery_suggestions(e)

    print("Recovery suggestions:")
    for i, action in enumerate(suggestions[:3], 1):
        print(f"{i}. {action.description}")
        print(f"   Strategy: {action.strategy.value}")
        print(f"   Confidence: {action.confidence:.1%}")
```

## Recovery Strategies

The system provides several recovery strategies:

### 1. Retry Current Step
- **When**: Temporary failures, network issues
- **Action**: Retry the failed operation with the same parameters

### 2. Skip to Next Step
- **When**: Non-critical step failures
- **Action**: Skip the failed step and continue execution

### 3. Restart Iteration
- **When**: Failures in critic loops
- **Action**: Restart from the beginning of the current iteration

### 4. Restart from Generation
- **When**: Validation or criticism failures
- **Action**: Restart from the text generation step

### 5. Modify Parameters
- **When**: Configuration-related failures
- **Action**: Adjust chain parameters based on error analysis

### 6. Full Restart
- **When**: All other strategies fail
- **Action**: Restart the entire chain execution from the beginning

## Checkpoint Structure

Each checkpoint contains:

```python
class ChainCheckpoint:
    checkpoint_id: str          # Unique checkpoint identifier
    chain_id: str              # Chain this checkpoint belongs to
    timestamp: datetime        # When checkpoint was created
    current_step: str          # Current execution step
    iteration: int             # Current iteration number
    thought: Thought           # Current thought state
    performance_data: dict     # Performance metrics
    recovery_point: str        # Where to resume from
    completed_validators: list # Completed validators
    completed_critics: list    # Completed critics
    metadata: dict            # Additional checkpoint data
```

## Execution Steps

The system creates checkpoints at these execution steps:

1. **initialization** - Chain setup complete
2. **pre_retrieval** - Before context retrieval
3. **generation** - After text generation
4. **validation** - After validation phase
5. **pre_criticism** - Before criticism phase
6. **criticism** - After criticism phase
7. **complete** - Execution finished

## Error Analysis

The recovery manager analyzes errors based on:

### Error Type Patterns
- **Timeout errors** → Reduce complexity, add delays
- **Rate limit errors** → Add retry delays, reduce frequency
- **Memory errors** → Reduce batch sizes, limit context
- **Validation errors** → Relax constraints, allow partial results

### Historical Patterns
- Similar checkpoint contexts
- Previous recovery success rates
- Error frequency analysis
- Performance correlation

### Recovery Confidence
Each recovery action includes:
- **Confidence score** based on pattern analysis
- **Parameter suggestions** for configuration adjustments
- **Context information** about when this strategy typically applies

## Advanced Usage

### Custom Recovery Strategies

```python
from sifaka.recovery import RecoveryAction, RecoveryStrategy

# Create custom recovery action
custom_action = RecoveryAction(
    strategy=RecoveryStrategy.MODIFY_PARAMETERS,
    description="Reduce model temperature for stability",
    confidence=0.8,
    parameters={"temperature": 0.3, "max_tokens": 1000}
)

# Apply manually
success = chain._apply_recovery_action(custom_action)
```

### Checkpoint History Analysis

```python
# Get checkpoint history
checkpoints = chain.get_checkpoint_history()

print(f"Chain executed {len(checkpoints)} steps:")
for cp in checkpoints:
    duration = cp.performance_data.get("total_time", 0)
    print(f"- {cp.current_step}: {duration:.2f}s")

# Analyze performance bottlenecks
bottlenecks = chain.get_performance_bottlenecks()
print(f"Performance bottlenecks: {bottlenecks}")
```

### Recovery Pattern Analysis

```python
# Analyze recovery patterns for a chain
recovery_history = recovery_manager.get_recovery_history(chain_id)

print(f"Total recovery attempts: {len(recovery_history)}")

# Most common recovery actions
actions = [event["action"] for event in recovery_history]
from collections import Counter
common_actions = Counter(actions).most_common(3)
print(f"Most common recovery actions: {common_actions}")

# Analyze error patterns
error_types = [event["error_type"] for event in recovery_history]
common_errors = Counter(error_types).most_common(3)
print(f"Most common error types: {common_errors}")
```

### Storage Management

```python
# Cleanup old checkpoints
cleaned_count = recovery_manager.cleanup_old_checkpoints(max_age_days=30)
print(f"Cleaned up {cleaned_count} old checkpoints")

# Get storage statistics
storage_stats = checkpoint_storage.get_storage_stats()
print(f"Storage usage: {storage_stats}")
```

## Best Practices

### 1. Configure Appropriate Storage
- Use **Redis** for production persistence
- Use **Milvus** for similarity-based recovery analysis
- Use **InMemory** for development and testing

### 2. Set Reasonable Checkpoint Frequency
- Balance between recovery granularity and storage overhead
- More checkpoints = better recovery, more storage

### 3. Monitor Recovery Patterns
- Track recovery success rates
- Identify common failure points
- Adjust chain configuration based on patterns

### 4. Cleanup Old Checkpoints
- Regularly clean up old checkpoints to manage storage
- Keep recent checkpoints for pattern analysis
- Archive important execution histories

### 5. Use Recovery Insights
- Analyze checkpoint performance data
- Identify bottlenecks in chain execution
- Optimize based on recovery patterns

## Integration with Existing Code

The checkpoint recovery system is designed to be minimally invasive:

```python
# Existing code
chain = Chain(model=model)
result = chain.run()

# With recovery (minimal changes)
chain = Chain(model=model, checkpoint_storage=checkpoint_storage)
result = chain.run_with_recovery()  # Drop-in replacement
```

## Performance Considerations

The checkpoint system adds some overhead to chain execution:

- **Checkpoint creation**: Additional time to save state at each step
- **Storage overhead**: Disk/memory usage for storing checkpoint data
- **Recovery analysis**: Time to analyze failures and suggest recovery options
- **Memory usage**: Additional memory for checkpoint data structures

The system is designed to minimize impact on normal execution while providing recovery capabilities when failures occur.

## Troubleshooting

### Common Issues

1. **Storage connection failures**
   - Check Redis/storage connectivity
   - Verify storage configuration
   - Use fallback storage options

2. **High checkpoint storage usage**
   - Implement regular cleanup
   - Reduce checkpoint frequency
   - Use compression for large thoughts

3. **Recovery loops**
   - Set maximum retry limits
   - Implement exponential backoff
   - Add circuit breaker patterns

4. **Performance degradation**
   - Monitor checkpoint creation time
   - Optimize storage backend
   - Reduce checkpoint data size

The checkpoint recovery system provides mechanisms for handling failures during chain execution by saving state at key points and offering recovery options when errors occur.
