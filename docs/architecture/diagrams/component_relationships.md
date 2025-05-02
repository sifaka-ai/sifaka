# Component Relationship Diagrams

This document provides visual representations of Sifaka's component relationships and interactions.

## Core Component Hierarchy

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

## Validation Flow

```mermaid
sequenceDiagram
    participant User
    participant Domain
    participant Rule
    participant Validator
    participant Model
    participant Critic

    User->>Domain: Validate Text
    Domain->>Rule: Delegate Validation
    Rule->>Validator: Perform Validation
    Validator-->>Rule: Return Result
    Rule-->>Domain: Return Processed Result
    Domain-->>User: Return Final Result

    alt Needs Improvement
        Domain->>Critic: Analyze Text
        Critic-->>Domain: Suggest Improvements
        Domain->>Model: Generate Improved Text
        Model-->>Domain: Return New Text
        Domain->>Rule: Validate New Text
    end
```

## Configuration Flow

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

## Rule-Validator Pattern

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

## Model-Critic Interaction

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

## Domain Composition

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

## Error Handling Flow

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

## Component Dependencies

```mermaid
graph TD
    A[Domain] -->|depends on| B[Rules]
    A -->|depends on| C[Model Provider]
    A -->|depends on| D[Critic]

    B -->|depends on| E[Validators]
    B -->|depends on| F[Rule Config]

    C -->|depends on| G[Model Config]

    D -->|depends on| H[Critic Config]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbb,stroke:#333,stroke-width:2px
```

## Data Flow

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

These diagrams illustrate:
1. The hierarchical structure of Sifaka's components
2. The flow of validation and improvement
3. Configuration relationships
4. The rule-validator pattern
5. Model-critic interactions
6. Domain composition
7. Error handling flows
8. Component dependencies
9. Data flow through the system

Each diagram provides a different perspective on how the components interact and work together to provide Sifaka's functionality.