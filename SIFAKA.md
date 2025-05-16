# Sifaka: A Fresh Start

## Executive Summary

This document outlines a plan for creating a completely new Sifaka framework at `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka`. Rather than refactoring the existing `sifaka_legacy` codebase or maintaining backward compatibility, we will build a new, streamlined framework from scratch, incorporating only the best concepts and patterns from the legacy codebase. This approach allows us to prioritize excellent design principles from the beginning, creating a more maintainable, intuitive, and powerful framework for LLM applications.

## Lessons from the Legacy Codebase

### What to Keep
- The core conceptual model (chains, models, validators, critics)
- The separation of concerns between generation, validation, and improvement
- The extensibility for different model providers
- The focus on reliability and robustness

### What to Leave Behind
- Deep inheritance hierarchies
- Complex dependency injection system
- Registry pattern for circular dependencies
- Verbose component creation and configuration
- Excessive abstraction and indirection
- Complex state management across multiple managers

## Design Principles for the New Framework

1. **Simplicity First**: Optimize for developer experience and readability
2. **Composition Over Inheritance**: Favor composition patterns over deep inheritance
3. **Explicit Over Implicit**: Make dependencies and behavior explicit
4. **Minimal API Surface**: Create a small, focused API that covers 90% of use cases
5. **Progressive Disclosure**: Simple for simple cases, powerful for complex cases
6. **Type Safety**: Leverage Python's type system for better tooling and documentation
7. **Testability**: Design for easy testing from the beginning
8. **Documentation-Driven**: Write documentation alongside code

## Core Architecture

### 1. Minimalist Core API

The new Sifaka will provide an elegant, fluent API that focuses on the most common use cases:

```python
import sifaka

# Create a simple chain with fluent interface
result = (sifaka.Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .run())

# Add validation and improvement
result = (sifaka.Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .validate_with(sifaka.rules.length(max_words=500))
    .improve_with(sifaka.critics.clarity())
    .run())
```

### 2. Clean Component Architecture

The new architecture will use a clean, modular design with minimal dependencies:

```
sifaka/
├── chain.py         # Chain orchestration
├── models/          # Model implementations
│   ├── base.py      # Base model interface
│   ├── openai.py    # OpenAI implementation
│   ├── anthropic.py # Anthropic implementation
│   └── gemini.py    # Google Gemini implementation
├── validators/      # Validation components
│   ├── rules.py     # Simple validation rules
│   └── critics.py   # LLM-based critics
├── results.py       # Result types and handling
├── errors.py        # Error types and handling
└── utils/           # Minimal utilities
    ├── logging.py   # Logging utilities
    └── config.py    # Configuration utilities
```

### 3. Functional Core, Imperative Shell

The new architecture will follow the "functional core, imperative shell" pattern:

- **Functional Core**: Pure functions for core logic (validation, transformation)
- **Imperative Shell**: Stateful components for I/O and orchestration

This approach makes the code more testable, maintainable, and easier to reason about.

## Implementation Strategy

### Phase 1: Core Foundation (2 weeks)
- Design the minimal API surface
- Implement the core Chain orchestrator
- Create the base Model interface
- Develop the Result types
- Establish testing framework

### Phase 2: Model Providers (2 weeks)
- Implement OpenAI provider
- Implement Anthropic provider
- Implement Google Gemini provider
- Create extensible provider interface

### Phase 3: Validation Components (2 weeks)
- Implement core validation rules
- Develop critic framework
- Create common critics
- Build validation pipeline

### Phase 4: Advanced Features (2 weeks)
- Implement caching
- Add observability and logging
- Create retry and fallback mechanisms
- Develop streaming support

### Phase 5: Documentation and Examples (2 weeks)
- Write comprehensive API documentation
- Create tutorials and guides
- Develop example applications
- Build interactive demos

## Technical Implementation Details

### 1. Model Interface

Simple, protocol-based interface for model providers:

```python
from typing import Protocol, Optional, Dict, Any

class Model(Protocol):
    """Protocol defining the interface for model providers."""
    
    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt."""
        ...
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...

# Implementation example
class OpenAIModel:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, **options: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **options
        )
        return response.choices[0].message.content
    
    def count_tokens(self, text: str) -> int:
        # Implementation details
        ...
```

### 2. Chain Implementation

Clean, builder-pattern implementation for chains:

```python
class Chain:
    """Main orchestrator for the generation, validation, and improvement flow."""
    
    def __init__(self):
        self._model = None
        self._prompt = None
        self._validators = []
        self._improvers = []
        self._options = {}
    
    def with_model(self, model: Union[str, Model]) -> "Chain":
        """Set the model to use for generation."""
        if isinstance(model, str):
            # Parse model string (e.g., "openai:gpt-4")
            provider, model_name = model.split(":", 1)
            self._model = create_model(provider, model_name)
        else:
            self._model = model
        return self
    
    def with_prompt(self, prompt: str) -> "Chain":
        """Set the prompt to use for generation."""
        self._prompt = prompt
        return self
    
    def validate_with(self, validator: Validator) -> "Chain":
        """Add a validator to the chain."""
        self._validators.append(validator)
        return self
    
    def improve_with(self, improver: Improver) -> "Chain":
        """Add an improver to the chain."""
        self._improvers.append(improver)
        return self
    
    def run(self) -> Result:
        """Execute the chain and return the result."""
        if not self._model:
            raise ValueError("Model not specified")
        if not self._prompt:
            raise ValueError("Prompt not specified")
        
        # Generate initial text
        text = self._model.generate(self._prompt, **self._options)
        
        # Validate text
        validation_results = []
        for validator in self._validators:
            result = validator.validate(text)
            validation_results.append(result)
            if not result.passed:
                return Result(
                    text=text,
                    passed=False,
                    validation_results=validation_results,
                    improvement_results=[]
                )
        
        # Improve text
        improvement_results = []
        for improver in self._improvers:
            improved_text, result = improver.improve(text)
            improvement_results.append(result)
            text = improved_text
        
        return Result(
            text=text,
            passed=True,
            validation_results=validation_results,
            improvement_results=improvement_results
        )
```

### 3. Validator and Improver Interfaces

Simple, functional interfaces for validation and improvement:

```python
class Validator(Protocol):
    """Protocol defining the interface for validators."""
    
    def validate(self, text: str) -> ValidationResult:
        """Validate text and return a result."""
        ...

class Improver(Protocol):
    """Protocol defining the interface for improvers."""
    
    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        """Improve text and return the improved text and a result."""
        ...

# Implementation examples
def length_validator(min_words: Optional[int] = None, max_words: Optional[int] = None) -> Validator:
    """Create a validator that checks text length."""
    
    def validate(text: str) -> ValidationResult:
        word_count = len(text.split())
        if min_words and word_count < min_words:
            return ValidationResult(
                passed=False,
                message=f"Text is too short ({word_count} words, minimum {min_words})"
            )
        if max_words and word_count > max_words:
            return ValidationResult(
                passed=False,
                message=f"Text is too long ({word_count} words, maximum {max_words})"
            )
        return ValidationResult(passed=True)
    
    return type("LengthValidator", (), {"validate": validate})()
```

## Advantages of the New Approach

1. **Simplicity**: The new design is dramatically simpler, with fewer layers of abstraction and a more intuitive API.

2. **Maintainability**: With a clean, modular architecture and minimal dependencies, the codebase will be easier to maintain and extend.

3. **Performance**: By eliminating unnecessary abstractions and indirection, the new framework will be more efficient.

4. **Developer Experience**: The fluent API and clear documentation will make the framework more accessible to new developers.

5. **Testability**: The functional core design makes the framework easier to test, with clear boundaries between components.

## Conclusion

By starting fresh with a new implementation of Sifaka, we can create a more elegant, maintainable, and powerful framework for LLM applications. This approach allows us to incorporate the best ideas from the legacy codebase while avoiding its complexity and design issues. The result will be a framework that is both simpler to use and more powerful, enabling developers to build reliable, robust LLM applications with less code and fewer headaches.
