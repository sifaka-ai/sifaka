# Sifaka Architecture

## Overview

Sifaka is a framework for building reliable and reflective AI systems, with a focus on validation, improvement, and monitoring capabilities. The system follows a component-based architecture that emphasizes modularity, extensibility, and maintainability.

## Core Components

### 1. Chain System (`/chain`)
The central orchestrator that coordinates all other components.

**Key Components:**
- `ChainCore` - Main interface that delegates to specialized components
- `ChainOrchestrator` - Main user-facing class for standardized implementation
- `PromptManager` - Manages prompt creation and management
- `ValidationManager` - Manages validation logic and rule management
- `RetryStrategy` - Handles retry logic with different strategies
- `ResultFormatter` - Handles formatting and processing of results

**Directory Structure:**
- `/chain/core.py` - Core chain implementation
- `/chain/orchestrator.py` - Main orchestrator implementation
- `/chain/strategies/` - Retry strategy implementations
- `/chain/managers/` - Prompt and validation managers
- `/chain/formatters/` - Result formatting implementations
- `/chain/interfaces/` - Component interfaces
- `/chain/factories.py` - Factory functions for creating chains
- `/chain/config.py` - Chain configuration
- `/chain/utils.py` - Utility functions

### 2. Critics System (`/critics`)
Handles content analysis and improvement suggestions.

**Key Components:**
- `CriticCore` - Base critic implementation
- `PromptCritic` - Prompt-based improvement
- `ReflexionCritic` - Self-reflection based improvement
- `ConstitutionalCritic` - Constitutional AI-based improvement
- `SelfRagCritic` - Self-RAG based improvement
- `SelfRefineCritic` - Self-refinement based improvement
- `LacCritic` - Language-Aware Critic for improvement

**Directory Structure:**
- `/critics/core.py` - Core critic implementation
- `/critics/base.py` - Base critic classes and interfaces
- `/critics/strategies/` - Improvement strategies
- `/critics/implementations/` - Specific critic implementations
  - `prompt.py` - Prompt-based critic
  - `reflexion.py` - Reflexion-based critic
  - `constitutional.py` - Constitutional AI critic
  - `self_rag.py` - Self-RAG critic
  - `self_refine.py` - Self-refinement critic
  - `lac.py` - Language-Aware Critic
- `/critics/services/` - Supporting services
- `/critics/managers/` - Critic management
- `/critics/interfaces/` - Critic interfaces
- `/critics/config.py` - Critic configuration
- `/critics/utils.py` - Utility functions

### 3. Rules System (`/rules`)
Defines and manages validation rules.

**Key Components:**
- `Rule` - Base rule interface
- Formatting Rules:
  - `LengthRule` - Length validation
  - `StructureRule` - Structure validation
  - `StyleRule` - Style validation
  - `WhitespaceRule` - Whitespace validation
  - `FormatRule` - General format validation
- Content Rules:
  - `SentimentRule` - Sentiment analysis
  - `ToneRule` - Tone analysis
  - `LanguageRule` - Language validation
  - `ProhibitedRule` - Prohibited content detection
  - `SafetyRule` - Safety checks

**Directory Structure:**
- `/rules/base.py` - Base rule classes and interfaces
- `/rules/formatting/` - Format-specific rules
  - `length.py` - Length validation rules
  - `structure.py` - Structure validation rules
  - `style.py` - Style validation rules
  - `whitespace.py` - Whitespace validation rules
  - `format.py` - General format validation rules
- `/rules/content/` - Content-specific rules
  - `sentiment.py` - Sentiment analysis rules
  - `tone.py` - Tone analysis rules
  - `language.py` - Language validation rules
  - `prohibited.py` - Prohibited content rules
  - `safety.py` - Safety validation rules
- `/rules/managers/` - Rule management
- `/rules/interfaces/` - Rule interfaces
- `/rules/factories.py` - Rule factory functions
- `/rules/config.py` - Rule configuration
- `/rules/utils.py` - Utility functions

### 4. Classifiers (`/classifiers`)
Specialized components for content classification.

**Key Components:**
- Content classifiers
- Topic classifiers
- Quality classifiers

### 5. Adapters (`/adapters`)
Interface adapters for external systems.

**Key Components:**
- Model adapters
- Service adapters
- Integration adapters

## Data Flow

1. **Input Processing**
   ```
   User Input → PromptManager → ChainCore
   ```

2. **Generation**
   ```
   ChainCore → Model Provider → Generated Text
   ```

3. **Validation**
   ```
   Generated Text → Validator → ValidationResult
   ```

4. **Improvement (if needed)**
   ```
   ValidationResult → Critic → Improvement Suggestions → Improver → Enhanced Text
   ```

5. **Output**
   ```
   Final Text → ResultFormatter → Formatted Output
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
- Confidence threshold failures

### 3. Improvement Errors
- Critic failures
- Improver issues
- Content conflicts
- Resource constraints
- Maximum attempts exceeded

### 4. Recovery Strategies
- Simple retry strategy
- Exponential backoff retry strategy
- Error reporting
- State preservation
- Graceful degradation

## Configuration

### 1. Chain Configuration
```python
from sifaka.chain import ChainOrchestrator
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
rules = [create_length_rule(min_chars=10, max_chars=1000)]
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain
chain = ChainOrchestrator(
    model=model,
    rules=rules,
    critic=critic,
    max_attempts=3
)
```

### 2. Validator Configuration
```python
from sifaka.validation import ValidatorConfig
from sifaka.rules import create_length_rule

config = ValidatorConfig(
    rules=[create_length_rule(min_chars=10, max_chars=1000)],
    fail_fast=True,
    params={"custom_param": "value"}
)
```

## Best Practices

### 1. Chain Design
- Use appropriate interfaces for components
- Implement proper error handling
- Configure retry strategies
- Set up monitoring
- Design for scalability
- Plan for resource usage

### 2. Component Selection
- Choose appropriate model providers
- Define relevant validation rules
- Select suitable critics
- Configure monitoring
- Consider performance requirements

### 3. Error Handling
- Implement comprehensive validation
- Use appropriate retry strategies
- Set up monitoring
- Handle edge cases
- Implement graceful degradation

### 4. Performance Optimization
- Use appropriate model sizes
- Implement caching
- Optimize validation
- Monitor resource usage
- Use batch processing when possible
- Configure appropriate timeouts

## Examples

### 1. Basic Chain
```python
from sifaka.chain import ChainOrchestrator
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
rules = [create_length_rule(min_chars=10, max_chars=1000)]

# Create chain
chain = ChainOrchestrator(model=model, rules=rules)

# Run chain
result = chain.run("Write a short story")
```

### 2. Advanced Chain with Critic
```python
from sifaka.chain import ChainOrchestrator
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
rules = [create_length_rule(min_chars=10, max_chars=1000)]
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain
chain = ChainOrchestrator(
    model=model,
    rules=rules,
    critic=critic,
    max_attempts=3
)

# Run chain
result = chain.run("Write a short story")
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```