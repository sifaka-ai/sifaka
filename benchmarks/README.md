# Sifaka Performance Benchmarks

This directory contains comprehensive performance benchmarks for the Sifaka library.

## Overview

The benchmarks cover:

- **Core Operations**: Basic `improve()` function performance
- **Scalability**: How performance scales with text size, critic count, iterations
- **Memory Usage**: Memory consumption patterns and bounds
- **Storage**: Performance of different storage backends
- **Concurrency**: Concurrent operation performance
- **Engine**: Engine initialization and reuse patterns

## Running Benchmarks

### Run All Benchmarks
```bash
python -m pytest benchmarks/ -v
```

### Run Specific Benchmark Categories
```bash
# Core operations only
python -m pytest benchmarks/performance_benchmarks.py::TestCoreOperationBenchmarks -v

# Scalability tests only
python -m pytest benchmarks/performance_benchmarks.py::TestScalabilityBenchmarks -v

# Memory benchmarks only
python -m pytest benchmarks/performance_benchmarks.py::TestMemoryBenchmarks -v
```

### Run Basic Benchmark Demo
```bash
python -m benchmarks.performance_benchmarks run
```

## Benchmark Structure

### PerformanceBenchmark Class

The `PerformanceBenchmark` class provides:

- **Timing**: Accurate execution time measurement
- **Memory Tracking**: Memory usage monitoring with psutil
- **CPU Monitoring**: CPU usage tracking
- **Result Collection**: Structured result storage
- **Summary Generation**: Aggregate statistics

### Key Metrics

Each benchmark measures:

- **Execution Time**: Total time for operation completion
- **Memory Usage**: Peak memory consumption during operation
- **CPU Usage**: CPU utilization during operation
- **Throughput**: Operations per second where applicable

## Benchmark Categories

### 1. Core Operations (`TestCoreOperationBenchmarks`)

- `test_basic_improve_benchmark`: Basic improve() performance
- `test_multiple_critics_benchmark`: Performance with multiple critics
- `test_iteration_scaling_benchmark`: Scaling with iteration count

### 2. Scalability (`TestScalabilityBenchmarks`)

- `test_text_size_scaling`: Performance vs input text size
- `test_concurrent_operations_benchmark`: Concurrent operation performance
- `test_validator_scaling_benchmark`: Performance with multiple validators

### 3. Storage (`TestStorageBenchmarks`)

- `test_memory_storage_benchmark`: MemoryStorage performance
- `test_file_storage_benchmark`: FileStorage performance

### 4. Memory (`TestMemoryBenchmarks`)

- `test_memory_growth_benchmark`: Memory growth over many operations
- `test_memory_bounds_benchmark`: Memory bounds enforcement

### 5. Engine (`TestEnginePerformanceBenchmarks`)

- `test_engine_initialization_benchmark`: Engine creation performance
- `test_engine_reuse_benchmark`: Engine reuse vs recreation

## Performance Expectations

### Core Operations
- Basic improve operation: < 2 seconds
- Memory usage per operation: < 20MB
- Multiple critics: Linear scaling with critic count

### Scalability
- Text size scaling: Sub-linear growth
- Concurrent operations: 5-20x faster than sequential
- Validator scaling: Linear with validator count

### Memory
- Total memory growth: < 100MB for 50 operations
- Memory bounds: Enforced at 10 generations, 20 critiques

### Storage
- Memory storage: 1000+ operations/second
- File storage: 10-50 operations/second

## Interpreting Results

### Good Performance Indicators
- Execution times under expected thresholds
- Memory usage within bounds
- Linear or sub-linear scaling
- Successful concurrent operations

### Performance Issues
- Exponential time growth with input size
- Memory leaks (unbounded growth)
- Failed concurrent operations
- Extremely slow storage operations

## Adding New Benchmarks

1. Create a new test class inheriting from appropriate base
2. Use the `benchmark` fixture for measurement
3. Add meaningful metadata to results
4. Include assertions for performance expectations
5. Document expected performance characteristics

Example:
```python
@pytest.mark.asyncio
async def test_my_benchmark(self, benchmark, mock_llm_client):
    \"\"\"Benchmark my operation.\"\"\"
    start_metrics = benchmark.start_measurement()

    # Perform operation
    result = await my_operation()

    metrics = benchmark.end_measurement(start_metrics)
    benchmark.add_result("my_test", metrics, {
        'metadata': 'value'
    })

    # Assert performance expectations
    assert metrics['execution_time'] < 1.0
```

## CI/CD Integration

Benchmarks are integrated into the CI/CD pipeline:

- Run on every pull request
- Performance regression detection
- Results stored for historical analysis
- Alerts for significant performance changes

## Troubleshooting

### Slow Benchmarks
- Check system resources (CPU, memory)
- Verify mock configurations
- Reduce test data size for development

### Memory Issues
- Ensure proper garbage collection
- Check for reference cycles
- Monitor system memory availability

### Inconsistent Results
- Run multiple times for average
- Check for background processes
- Use consistent test environments
