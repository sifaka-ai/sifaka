# Sifaka Architecture

This document provides a detailed overview of the Sifaka architecture, including component relationships, data flow, and design patterns.

## Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
- [Component Interactions](#component-interactions)
- [Data Flow](#data-flow)
- [Design Patterns](#design-patterns)
- [Error Handling](#error-handling)
- [Registry System](#registry-system)
- [Configuration Management](#configuration-management)

## System Overview

Sifaka is designed as a modular framework for text generation, validation, and improvement. The architecture follows these key principles:

1. **Modularity**: Components are designed to be independent and interchangeable
2. **Extensibility**: The system can be easily extended with new components
3. **Composability**: Components can be combined in various ways to create complex workflows
4. **Separation of Concerns**: Each component has a specific responsibility

The high-level architecture can be visualized as follows:

```
┌─────────────────────────────────────────────────────────────────────┐
│                             Chain                                    │
│                                                                      │
│  ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌────────────┐  │
│  │          │    │           │    │            │    │            │  │
│  │  Models  │───▶│ Generation│───▶│ Validation │───▶│ Improvement│  │
│  │          │    │           │    │            │    │            │  │
│  └──────────┘    └───────────┘    └────────────┘    └────────────┘  │
│                                          │                  │        │
│                                          ▼                  ▼        │
│                                   ┌────────────┐    ┌────────────┐  │
│                                   │            │    │            │  │
│                                   │ Validators │    │  Critics   │  │
│                                   │            │    │            │  │
│                                   └────────────┘    └────────────┘  │
│                                          │                  │        │
│                                          │                  │        │
│                                          ▼                  ▼        │
│                                   ┌────────────┐    ┌────────────┐  │
│                                   │            │    │            │  │
│                                   │Classifiers │    │ Retrievers │  │
│                                   │            │    │            │  │
│                                   └────────────┘    └────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                             ┌────────────┐
                             │            │
                             │  Results   │
                             │            │
                             └────────────┘
```

## Core Components

### Chain

The Chain is the central orchestrator that coordinates the entire process. It follows the builder pattern, allowing users to configure the chain through method chaining.

**Responsibilities**:
- Coordinating the text generation, validation, and improvement process
- Managing the flow of data between components
- Handling errors and providing meaningful feedback
- Returning structured results

### Models

Models are integrations with various language model providers that handle text generation.

**Responsibilities**:
- Generating text from prompts
- Counting tokens in text
- Handling API communication with providers

### Validators

Validators check if text meets specific criteria, such as length, content, or format requirements.

**Responsibilities**:
- Validating text against specific criteria
- Providing detailed validation results
- Suggesting improvements when validation fails

### Critics

Critics enhance the quality of text by applying various improvement strategies.

**Responsibilities**:
- Analyzing text for areas of improvement
- Generating improved versions of text
- Providing detailed improvement results

### Retrievers

Retrievers fetch relevant information for queries, primarily used by retrieval-augmented critics.

**Responsibilities**:
- Retrieving relevant documents for queries
- Ranking documents by relevance
- Providing context for text generation and improvement

### Classifiers

Classifiers categorize text into specific classes or labels.

**Responsibilities**:
- Classifying text into categories
- Providing confidence scores for classifications
- Supporting validation through classifier adapters

## Component Interactions

The components interact in a well-defined sequence:

1. **Chain** orchestrates the entire process
2. **Model** generates initial text based on the prompt
3. **Validators** check if the generated text meets the specified criteria
4. **Critics** improve the text if it passes validation
5. **Results** are returned to the user

The detailed interaction flow is as follows:

```
┌─────────┐     ┌─────────┐     ┌─────────────┐     ┌─────────┐     ┌─────────┐
│         │     │         │     │             │     │         │     │         │
│  User   │────▶│  Chain  │────▶│    Model    │────▶│  Chain  │────▶│Validator│
│         │     │         │     │             │     │         │     │         │
└─────────┘     └─────────┘     └─────────────┘     └─────────┘     └─────────┘
                                       │                                  │
                                       │                                  │
                                       ▼                                  ▼
┌─────────┐     ┌─────────┐     ┌─────────────┐     ┌─────────┐     ┌─────────┐
│         │     │         │     │             │     │         │     │         │
│  User   │◀────│  Chain  │◀────│   Result    │◀────│  Chain  │◀────│  Critic │
│         │     │         │     │             │     │         │     │         │
└─────────┘     └─────────┘     └─────────────┘     └─────────┘     └─────────┘
```

## Data Flow

The data flows through the system as follows:

1. **Input**: User provides a prompt and configuration
2. **Generation**: Model generates text from the prompt
3. **Validation**: Validators check if the text meets criteria
   - If validation fails, the process stops and returns a failed result
4. **Improvement**: Critics improve the text if validation passes
5. **Output**: The final text and results are returned to the user

The detailed data flow is as follows:

```
┌─────────────────┐
│                 │
│  User Input     │
│  - Prompt       │
│  - Configuration│
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│                 │
│  Chain          │
│  - Configures   │
│    components   │
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│                 │
│  Model          │
│  - Generates    │
│    text         │
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Validators     │ No  │  Failed Result  │
│  - Text meets   ├────▶│  - Error message│
│    criteria?    │     │  - Suggestions  │
│                 │     │                 │
└────────┬────────┘     └────────┬────────┘
         │ Yes                   │
         ▼                       │
┌─────────────────┐              │
│                 │              │
│  Critics        │              │
│  - Improve text │              │
│                 │              │
└────────┬────────┘              │
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Success Result │     │  User           │
│  - Final text   │────▶│  - Processes    │
│  - Details      │     │    result       │
│                 │     │                 │
└─────────────────┘     └─────────────────┘
```

## Design Patterns

Sifaka uses several design patterns to achieve its goals:

### Builder Pattern

The Chain class uses the builder pattern to provide a fluent interface for configuration:

```python
chain = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story.")
    .validate_with(length(min_words=100))
    .improve_with(create_reflexion_critic())
)
```

### Factory Pattern

The registry system uses the factory pattern to create components:

```python
# Register a factory
registry.register("model", "openai", create_openai_model)

# Create a component using the factory
model = registry.create("model", "openai", model_name="gpt-4")
```

### Strategy Pattern

Validators and Critics use the strategy pattern to provide different validation and improvement strategies:

```python
# Different validation strategies
validator1 = length(min_words=100)
validator2 = prohibited_content(prohibited=["violent"])

# Different improvement strategies
critic1 = create_reflexion_critic()
critic2 = create_n_critics_critic(num_critics=3)
```

### Adapter Pattern

The ClassifierValidator uses the adapter pattern to adapt classifiers to the validator interface:

```python
# Create a classifier
classifier = ProfanityClassifier()

# Adapt it to a validator
validator = classifier_validator(classifier, pass_on="negative")
```

## Error Handling

Sifaka uses a comprehensive error handling system:

1. **Specific Error Types**: Each component has specific error types (ValidationError, ModelError, etc.)
2. **Contextual Information**: Errors include context about what operation failed and why
3. **Suggestions**: Errors include suggestions for fixing the issue
4. **Graceful Degradation**: The system attempts to continue operation when possible

## Registry System

The registry system provides dependency injection and component management:

1. **Component Registration**: Components are registered with the registry
2. **Factory Functions**: Factory functions create component instances
3. **Dependency Resolution**: Dependencies are resolved at runtime
4. **Configuration**: Components are configured based on the provided options

## Configuration Management

Sifaka uses a centralized configuration system:

1. **Hierarchical Configuration**: Configuration is organized hierarchically
2. **Default Values**: Sensible defaults are provided for all configuration options
3. **Override Mechanism**: Configuration can be overridden at different levels
4. **Validation**: Configuration is validated to ensure it's valid

## Next Steps

To learn more about specific components, refer to the following documentation:

- [Chain Documentation](CHAIN.md)
- [Models Documentation](MODELS.md)
- [Validators Documentation](VALIDATORS.md)
- [Critics Documentation](CRITICS.md)
- [Retrievers Documentation](RETRIEVERS.md)
- [Classifiers Documentation](CLASSIFIERS.md)
- [API Reference](API_REFERENCE.md)
