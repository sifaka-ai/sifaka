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

### 3. Validation System
Ensures generated content meets specified criteria.

**Components:**
- **Rules**: Define validation criteria
  - Length rules
  - Toxicity detection
  - Content classification
  - Custom rules

- **Validators**: Apply rules to content
  - Parallel validation
  - Early termination
  - Error collection
  - Result aggregation

### 4. Improvement System
Enhances generated content that fails validation.

**Components:**
- **Critics**: Analyze and suggest improvements
  - Content analysis
  - Error identification
  - Improvement suggestions
  - Quality assessment

- **Improvers**: Apply suggested changes
  - Content modification
  - Style adjustment
  - Error correction
  - Quality enhancement

### 5. Monitoring System
Tracks system performance and behavior.

**Features:**
- Performance metrics
- Error tracking
- Resource monitoring
- Usage statistics
- Alerting

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

### 2. Validation Errors
- Rule violations
- Content issues
- Format problems
- Quality concerns

### 3. Improvement Errors
- Critic failures
- Improver issues
- Content conflicts
- Resource constraints

### 4. Recovery Strategies
- Automatic retries
- Fallback providers
- Simplified validation
- Error reporting

## Configuration

### 1. Chain Configuration
```python
chain_config = {
    "model": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7
    },
    "validation": {
        "rules": [
            {"type": "length", "min": 10, "max": 1000},
            {"type": "toxicity", "threshold": 0.7}
        ]
    },
    "improvement": {
        "enabled": True,
        "max_attempts": 3
    },
    "monitoring": {
        "enabled": True,
        "metrics": ["performance", "errors", "usage"]
    }
}
```

### 2. Component Configuration
Each component can be configured independently:
- Model provider settings
- Validation rule parameters
- Improvement strategies
- Monitoring options

## Best Practices

### 1. Chain Design
- Start with clear requirements
- Define validation criteria
- Plan improvement strategies
- Consider monitoring needs

### 2. Component Selection
- Choose appropriate model providers
- Define relevant validation rules
- Select suitable critics
- Configure monitoring

### 3. Error Handling
- Implement comprehensive validation
- Plan for common failures
- Set up monitoring
- Define recovery strategies

### 4. Performance Optimization
- Use appropriate model sizes
- Implement caching
- Optimize validation
- Monitor resource usage

## Examples

### 1. Basic Chain
```python
from sifaka.chain import ChainCore
from sifaka.models import create_openai_provider
from sifaka.rules import create_length_rule

# Create components
model = create_openai_provider("gpt-3.5-turbo")
rules = [create_length_rule(min=10, max=1000)]

# Create chain
chain = ChainCore(model=model, rules=rules)

# Run chain
result = chain.run("Write a short story")
```

### 2. Advanced Chain
```python
from sifaka.chain import ChainCore
from sifaka.models import create_openai_provider
from sifaka.rules import create_length_rule, create_toxicity_rule
from sifaka.critics import create_content_critic
from sifaka.monitoring import create_metrics_monitor

# Create components
model = create_openai_provider("gpt-4")
rules = [
    create_length_rule(min=100, max=1000),
    create_toxicity_rule(threshold=0.7)
]
critic = create_content_critic()
monitor = create_metrics_monitor()

# Create chain
chain = ChainCore(
    model=model,
    rules=rules,
    critic=critic,
    monitor=monitor
)

# Run chain
result = chain.run("Write a technical article")
```