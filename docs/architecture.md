# Sifaka Architecture

## Overview

Sifaka is a framework for building and managing AI-powered text generation chains with built-in validation, improvement, and monitoring capabilities. The system follows a component-based architecture that emphasizes modularity, extensibility, and maintainability.

## Core Components

### 1. Chain Core
The central orchestrator that coordinates all other components.

**Key Responsibilities:**
- Managing the execution flow
- Coordinating between components
- Handling retries and error recovery
- Formatting and returning results

**Lifecycle:**
1. **Initialization**
   - Configure with required components
   - Set up validation rules
   - Initialize monitoring
   - Prepare retry strategies

2. **Execution**
   - Process input prompts
   - Generate text
   - Validate outputs
   - Apply improvements if needed
   - Format and return results

3. **Monitoring**
   - Track performance metrics
   - Log validation results
   - Monitor resource usage
   - Report errors and issues

### 2. Model Providers
Interface with various AI models for text generation.

**Supported Providers:**
- OpenAI
- Anthropic
- Local models
- Custom providers

**Features:**
- Standardized interface
- Token counting
- Error handling
- Rate limiting
- Caching
- Batch processing
- Timeout handling

### 3. Validation System
Ensures generated content meets specified criteria.

**Components:**
- **Rules**: Define validation criteria
  - Length rules
  - Toxicity detection
  - Content classification
  - Topic matching
  - Custom rules

- **Validators**: Apply rules to content
  - Parallel validation
  - Early termination
  - Error collection
  - Result aggregation
  - Confidence scoring

### 4. Improvement System
Enhances generated content that fails validation.

**Components:**
- **Critics**: Analyze and suggest improvements
  - Content analysis
  - Error identification
  - Improvement suggestions
  - Quality assessment
  - Confidence scoring

- **Improvers**: Apply suggested changes
  - Content modification
  - Style adjustment
  - Error correction
  - Quality enhancement
  - Batch processing

### 5. Monitoring System
Tracks system performance and behavior.

**Features:**
- Performance metrics
- Error tracking
- Resource monitoring
- Usage statistics
- Cache statistics
- Alerting
- Detailed logging

## Data Flow

1. **Input Processing**
   ```
   User Input → Prompt Manager → Chain Core
   ```

2. **Generation**
   ```
   Chain Core → Model Provider → Generated Text
   ```

3. **Validation**
   ```
   Generated Text → Validation Manager → Validation Results
   ```

4. **Improvement (if needed)**
   ```
   Validation Results → Critic → Improvement Suggestions → Improver → Enhanced Text
   ```

5. **Output**
   ```
   Final Text → Result Formatter → Formatted Output
   ```

## Error Handling

### 1. Generation Errors
- Model provider failures
- Rate limiting
- Token limits
- Network issues
- Invalid prompts
- Model timeouts
- Batch processing errors

### 2. Validation Errors
- Rule violations
- Content issues
- Format problems
- Quality concerns
- Topic mismatches
- Toxicity detection
- Confidence threshold failures

### 3. Improvement Errors
- Critic failures
- Improver issues
- Content conflicts
- Resource constraints
- Confidence threshold failures
- Maximum attempts exceeded
- Batch processing errors

### 4. Recovery Strategies
- Automatic retries
- Fallback providers
- Simplified validation
- Error reporting
- State preservation
- Graceful degradation

## Configuration

### 1. Chain Configuration
```python
chain_config = {
    "model": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "validation": {
        "rules": [
            {"type": "length", "min": 10, "max": 1000},
            {"type": "toxicity", "threshold": 0.7},
            {"type": "topic", "allowed_topics": ["technology", "science"]}
        ]
    },
    "improvement": {
        "enabled": True,
        "max_attempts": 3,
        "critic": {
            "type": "content",
            "confidence_threshold": 0.8
        }
    },
    "monitoring": {
        "enabled": True,
        "metrics": ["performance", "errors", "usage", "cache_stats"]
    }
}
```

### 2. Component Configuration
Each component can be configured independently:
- Model provider settings
- Validation rule parameters
- Improvement strategies
- Monitoring options
- Cache settings
- Timeout values
- Batch sizes

## Best Practices

### 1. Chain Design
- Start with clear requirements
- Define validation criteria
- Plan improvement strategies
- Consider monitoring needs
- Design for scalability
- Plan for error handling

### 2. Component Selection
- Choose appropriate model providers
- Define relevant validation rules
- Select suitable critics
- Configure monitoring
- Consider performance requirements
- Plan for resource usage

### 3. Error Handling
- Implement comprehensive validation
- Plan for common failures
- Set up monitoring
- Define recovery strategies
- Handle edge cases
- Implement graceful degradation

### 4. Performance Optimization
- Use appropriate model sizes
- Implement caching
- Optimize validation
- Monitor resource usage
- Use batch processing when possible
- Implement early termination for validation
- Configure appropriate timeouts
- Use appropriate confidence thresholds

## Examples

### 1. Basic Chain
```python
from sifaka.chain import create_simple_chain
from sifaka.models import create_openai_provider
from sifaka.rules import create_length_rule

# Create components
model = create_openai_provider("gpt-3.5-turbo")
rules = [create_length_rule(min_chars=10, max_chars=1000)]

# Create chain
chain = create_simple_chain(model=model, rules=rules)

# Run chain
result = chain.run("Write a short story")
```

### 2. Advanced Chain
```python
from sifaka.chain import create_backoff_chain
from sifaka.models import create_openai_provider
from sifaka.rules import create_length_rule, create_toxicity_rule
from sifaka.critics import create_prompt_critic

# Create components
model = create_openai_provider("gpt-3.5-turbo")
rules = [
    create_length_rule(min_chars=10, max_chars=1000),
    create_toxicity_rule(threshold=0.7)
]
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain with backoff retry strategy
chain = create_backoff_chain(
    model=model,
    rules=rules,
    critic=critic,
    max_attempts=3,
    initial_backoff=1.0,
    backoff_factor=2.0,
    max_backoff=60.0
)

# Run chain
result = chain.run("Write a short story")
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```

### 3. Using ChainOrchestrator
```python
from sifaka.chain import ChainOrchestrator
from sifaka.models import create_openai_provider
from sifaka.rules import create_length_rule, create_toxicity_rule
from sifaka.critics import create_prompt_critic

# Create components
model = create_openai_provider("gpt-3.5-turbo")
rules = [
    create_length_rule(min_chars=10, max_chars=1000),
    create_toxicity_rule(threshold=0.7)
]
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain orchestrator
chain = ChainOrchestrator(
    model=model,
    rules=rules,
    critic=critic,
    max_attempts=3
)

# Run chain
result = chain.run("Write a short story")
```