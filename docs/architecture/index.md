# Architecture Overview

This page provides an overview of the Sifaka architecture.

## Design Philosophy

Sifaka was designed with the following principles in mind:

1. **Simplicity First**: Optimize for developer experience and readability
2. **Composition Over Inheritance**: Favor composition patterns over deep inheritance
3. **Explicit Over Implicit**: Make dependencies and behavior explicit
4. **Minimal API Surface**: Create a small, focused API that covers 90% of use cases
5. **Progressive Disclosure**: Simple for simple cases, powerful for complex cases
6. **Type Safety**: Leverage Python's type system for better tooling and documentation
7. **Testability**: Design for easy testing from the beginning
8. **Documentation-Driven**: Write documentation alongside code

## Core Architecture

Sifaka follows a "functional core, imperative shell" architecture pattern:

- **Functional Core**: Pure functions for core logic (validation, transformation)
- **Imperative Shell**: Stateful components for I/O and orchestration

This approach makes the code more testable, maintainable, and easier to reason about.

### Component Overview

![Sifaka Architecture](../assets/architecture.png)

The main components of Sifaka are:

1. **Chain**: The main orchestrator that coordinates the generation, validation, and improvement of text
2. **Models**: Interfaces to various LLM providers (OpenAI, Anthropic, etc.)
3. **Validators**: Components that check if generated text meets specific criteria
4. **Critics**: LLM-based components that validate and improve text
5. **Results**: Types that represent the results of various operations

### Flow of Execution

When you run a Chain, the following steps occur:

1. The Chain uses the configured Model to generate text from the prompt
2. The generated text is passed through each Validator in sequence
   - If any validator fails, the Chain returns a failed Result
3. If all validators pass, the text is passed through each Improver in sequence
   - Each improver returns improved text and an ImprovementResult
   - The improved text from one improver is passed to the next
4. The Chain returns a Result containing the final text, validation results, and improvement results

## Component Details

### Chain

The Chain class is the main entry point for Sifaka. It uses a builder pattern to provide a fluent API:

```python
result = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=50, max_words=200))
    .improve_with(clarity())
    .run())
```

Internally, the Chain maintains:
- A reference to the model to use
- The prompt to generate text from
- A list of validators to apply
- A list of improvers to apply
- Options to pass to the model

### Models

Models provide a unified interface to different LLM providers. The core interface is defined by the Model protocol:

```python
class Model(Protocol):
    def generate(self, prompt: str, **options: Any) -> str: ...
    def count_tokens(self, text: str) -> int: ...
```

Sifaka includes implementations for popular providers:
- OpenAIModel for OpenAI models
- AnthropicModel for Anthropic models
- GeminiModel for Google Gemini models

### Validators

Validators check if text meets specific criteria. They follow a simple interface:

```python
class Validator(Protocol):
    def validate(self, text: str) -> ValidationResult: ...
```

Sifaka includes several built-in validators:
- Length validators for checking text length
- Content validators for checking text content
- Format validators for checking text format

### Critics

Critics are LLM-based components that can both validate and improve text. They implement both the Validator and Improver interfaces:

```python
class Critic:
    def validate(self, text: str) -> ValidationResult: ...
    def improve(self, text: str) -> Tuple[str, ImprovementResult]: ...
```

Sifaka includes several built-in critics:
- ClarityAndCoherenceCritic for improving text clarity
- FactualAccuracyCritic for validating factual accuracy

### Results

Results represent the outcome of various operations:

- ValidationResult for validation operations
- ImprovementResult for improvement operations
- Result for chain execution

## Design Patterns

Sifaka uses several design patterns:

### Builder Pattern

The Chain class uses the builder pattern to provide a fluent API:

```python
chain = (Chain()
    .with_model(...)
    .with_prompt(...)
    .validate_with(...)
    .improve_with(...)
    .with_options(...))
```

This makes the API more readable and intuitive.

### Protocol Classes

Sifaka uses Protocol classes from Python's typing module to define interfaces:

```python
class Model(Protocol):
    def generate(self, prompt: str, **options: Any) -> str: ...
    def count_tokens(self, text: str) -> int: ...
```

This allows for structural typing, making the framework more extensible.

### Factory Functions

Sifaka uses factory functions to create instances of various components:

```python
# Create a model
model = create_model("openai", "gpt-4")

# Create a validator
validator = length(min_words=50, max_words=200)

# Create a critic
critic = clarity()
```

This makes the API more concise and easier to use.

### Composition

Sifaka favors composition over inheritance. For example, a Chain is composed of a Model, Validators, and Improvers, rather than inheriting from them.

## Extension Points

Sifaka is designed to be extensible. Here are the main extension points:

### Custom Models

You can create custom model implementations by implementing the Model protocol:

```python
class CustomModel:
    def generate(self, prompt: str, **options: Any) -> str:
        # Custom implementation
        ...
    
    def count_tokens(self, text: str) -> int:
        # Custom implementation
        ...
```

### Custom Validators

You can create custom validators by implementing the Validator protocol:

```python
class CustomValidator:
    def validate(self, text: str) -> ValidationResult:
        # Custom implementation
        ...
```

### Custom Critics

You can create custom critics by extending the Critic base class:

```python
class CustomCritic(Critic):
    def validate(self, text: str) -> ValidationResult:
        # Custom implementation
        ...
    
    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        # Custom implementation
        ...
```

## Conclusion

Sifaka's architecture is designed to be simple, flexible, and extensible. By following the principles of functional core, imperative shell, and composition over inheritance, Sifaka provides a powerful framework for building reliable LLM applications.
