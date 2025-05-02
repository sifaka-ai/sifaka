# System Design

This document provides a high-level overview of Sifaka's system design, including its architecture, components, and their interactions.

## System Overview

Sifaka is a flexible and extensible framework for text validation and improvement. It follows a modular design that allows for easy extension and customization.

### Core Architecture

The system is built around several core components that work together to provide text validation and improvement capabilities:

```mermaid
graph TD
    A[Sifaka] --> B[Domains]
    A --> C[Rules]
    A --> D[Validators]
    A --> E[Model Providers]
    A --> F[Critics]

    B --> G[Domain Config]
    B --> H[Rule Set]

    C --> I[Rule Config]
    C --> J[Validator]

    D --> K[Validation Logic]

    E --> L[Model Config]
    E --> M[Generation Logic]

    F --> N[Critic Config]
    F --> O[Improvement Logic]
```

## Component Design

### 1. Domains

Domains are the top-level containers that manage validation and improvement workflows. They coordinate the interaction between rules, models, and critics.

```mermaid
classDiagram
    class Domain {
        +name: str
        +description: str
        +rules: List[Rule]
        +validate(text: T) ValidationResult
    }

    class Rule {
        +name: str
        +description: str
        +validate(text: T) RuleResult
    }

    class ValidationResult {
        +all_passed: bool
        +rule_results: List[RuleResult]
    }

    Domain --> Rule: contains
    Domain --> ValidationResult: returns
    Rule --> RuleResult: returns
```

### 2. Rules and Validators

Rules and validators follow a delegation pattern where rules manage the validation lifecycle and validators implement the actual validation logic.

```mermaid
classDiagram
    class Rule {
        +name: str
        +description: str
        +config: RuleConfig
        +validator: Validator
        +validate(text: T) R
    }

    class Validator {
        +config: ValidatorConfig
        +validate(text: T) RuleResult
        +handle_empty_text(text: T) RuleResult?
    }

    class RuleConfig {
        +params: Dict[str, Any]
    }

    Rule --> Validator: uses
    Rule --> RuleConfig: has
    Validator --> RuleConfig: has
```

### 3. Model Providers

Model providers interface with language models to generate and improve text. They handle model-specific configuration and token management.

```mermaid
sequenceDiagram
    participant User
    participant Model
    participant Critic
    participant Domain

    User->>Model: Generate Text
    Model-->>User: Return Text

    User->>Domain: Validate Text
    Domain->>Critic: Analyze Text
    Critic-->>Domain: Return Analysis

    alt Needs Improvement
        Domain->>Critic: Improve Text
        Critic->>Model: Generate Improved Version
        Model-->>Critic: Return Improved Text
        Critic-->>Domain: Return Improved Text
        Domain-->>User: Return Final Text
    else No Improvement Needed
        Domain-->>User: Return Original Text
    end
```

## Data Flow

The system follows a clear data flow pattern:

```mermaid
graph LR
    A[Input Text] --> B[Domain]
    B --> C[Rules]
    C --> D[Validators]
    D --> E[Validation Results]
    E --> F[Domain Results]

    G[Model] --> H[Generated Text]
    H --> B

    I[Critic] --> J[Improved Text]
    J --> B

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
    style H fill:#fbb,stroke:#333,stroke-width:2px
    style J fill:#fbf,stroke:#333,stroke-width:2px
```

## Error Handling

The system implements a comprehensive error handling strategy:

```mermaid
graph TD
    A[Validation Error] --> B{Error Type}
    B -->|Rule Violation| C[Rule Result]
    B -->|Model Error| D[Model Error]
    B -->|Critic Error| E[Critic Error]

    C --> F[Error Message]
    D --> F
    E --> F

    F --> G[Error Handling]
    G -->|Retry| H[Retry Logic]
    G -->|Fallback| I[Fallback Logic]
    G -->|Abort| J[Abort Logic]
```

## Configuration Management

Configuration is managed through a hierarchical structure:

```mermaid
graph TD
    A[User] -->|creates| B[DomainConfig]
    B -->|contains| C[RuleConfigs]
    C -->|configures| D[Rules]
    D -->|uses| E[Validators]

    B -->|contains| F[ModelConfig]
    F -->|configures| G[ModelProvider]

    B -->|contains| H[CriticConfig]
    H -->|configures| I[Critic]
```

## Design Principles

1. **Modularity**: Each component has a single responsibility and clear interfaces
2. **Extensibility**: New rules, validators, and critics can be added without modifying core code
3. **Configuration-Driven**: Behavior is controlled through configuration rather than code changes
4. **Type Safety**: Strong typing throughout the system to catch errors early
5. **Error Handling**: Comprehensive error handling with clear error messages and recovery strategies
6. **Performance**: Efficient validation and improvement processes
7. **Testability**: Components are designed to be easily testable in isolation

## Implementation Patterns

1. **Factory Pattern**: Used for creating rules, validators, and other components
2. **Strategy Pattern**: Used for different validation and improvement strategies
3. **Observer Pattern**: Used for monitoring validation and improvement progress
4. **Decorator Pattern**: Used for adding functionality to existing components
5. **Composite Pattern**: Used for managing collections of rules and validators

## Best Practices

1. **Rule Implementation**
   - Use factory functions for creation
   - Delegate validation to validators
   - Process validation results
   - Provide clear error messages

2. **Validator Implementation**
   - Focus on single validation task
   - Handle edge cases
   - Return detailed results
   - Support configuration

3. **Configuration**
   - Use type-safe config classes
   - Validate configuration values
   - Provide sensible defaults
   - Document configuration options

4. **Integration**
   - Follow established patterns
   - Use provided interfaces
   - Handle errors gracefully
   - Support customization