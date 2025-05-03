# Rules and Validators Relationship

This document provides visual representations of the relationship between Rules and Validators in Sifaka.

## Basic Rule-Validator Pattern

```mermaid
classDiagram
    class Rule {
        +name: str
        +description: str
        +config: RuleConfig
        +validator: Validator
        +validate(text: str) RuleResult
    }

    class Validator {
        +config: ValidatorConfig
        +validate(text: str) RuleResult
        +handle_empty_text(text: str) RuleResult?
    }

    class RuleConfig {
        +params: Dict[str, Any]
        +priority: int
        +cache_size: int
        +cost: float
    }

    class RuleResult {
        +passed: bool
        +message: str
        +metadata: Dict[str, Any]
        +with_metadata() RuleResult
    }

    Rule --> Validator : delegates validation to
    Rule --> RuleConfig : configured by
    Validator --> RuleConfig : accesses
    Rule ..> RuleResult : returns
    Validator ..> RuleResult : returns
```

## Factory Pattern for Rules and Validators

```mermaid
sequenceDiagram
    participant Client
    participant RuleFactory as Rule Factory
    participant ValidatorFactory as Validator Factory
    participant RuleConfig as Rule Config
    participant Rule
    participant Validator

    Client->>RuleFactory: create_example_rule(param1, param2)
    RuleFactory->>ValidatorFactory: create_example_validator(param1, param2)
    ValidatorFactory->>RuleConfig: create config
    RuleConfig-->>ValidatorFactory: return config
    ValidatorFactory->>Validator: create validator with config
    Validator-->>ValidatorFactory: return validator
    ValidatorFactory-->>RuleFactory: return validator
    RuleFactory->>Rule: create rule with validator
    Rule-->>RuleFactory: return rule
    RuleFactory-->>Client: return rule
```

## Rule Hierarchy and Inheritance

```mermaid
classDiagram
    class Rule {
        <<abstract>>
        +name: str
        +description: str
        +validate(text: T) RuleResult
    }

    class BaseRule {
        +config: RuleConfig
        +validator: BaseValidator
        +validate(text: T) RuleResult
    }

    class FormattingRule {
        +validator: FormattingValidator
    }

    class ContentRule {
        +validator: ContentValidator
    }

    class FactualRule {
        +validator: FactualValidator
    }

    class DomainRule {
        +validator: DomainValidator
    }

    Rule <|-- BaseRule
    BaseRule <|-- FormattingRule
    BaseRule <|-- ContentRule
    BaseRule <|-- FactualRule
    BaseRule <|-- DomainRule

    class LengthRule {
        +min_chars: int
        +max_chars: int
    }

    class StyleRule {
        +capitalization: str
        +require_punctuation: bool
    }

    class ToxicityRule {
        +threshold: float
    }

    FormattingRule <|-- LengthRule
    FormattingRule <|-- StyleRule
    ContentRule <|-- ToxicityRule
```

## Rule-Validator Delegation Flow

```mermaid
sequenceDiagram
    participant Client
    participant Rule
    participant Validator
    participant RuleResult

    Client->>Rule: validate("text to validate")
    Rule->>Validator: validate("text to validate")
    Validator->>Validator: handle_empty_text("text to validate")

    alt Text is empty
        Validator-->>Rule: RuleResult(passed=true, message="Empty text validation skipped")
    else Text is not empty
        Validator->>Validator: perform validation logic
        Validator-->>Rule: RuleResult(passed=true/false, message="result message")
    end

    Rule->>Rule: process result
    Rule->>RuleResult: add rule metadata
    RuleResult-->>Rule: return enriched result
    Rule-->>Client: return final RuleResult
```

## Rule Adapters

```mermaid
classDiagram
    class Rule {
        <<interface>>
        +validate(text: str) RuleResult
    }

    class Adapter {
        <<interface>>
        +adapt()
    }

    class ClassifierAdapter {
        +classifier: Classifier
        +validate(text: str) RuleResult
    }

    class GuardrailsAdapter {
        +rules: List[Rule]
        +run(model, prompt) str
    }

    Rule <|.. ClassifierAdapter
    Adapter <|.. ClassifierAdapter
    Adapter <|.. GuardrailsAdapter
    GuardrailsAdapter --> Rule : contains
```

## Validator Types

```mermaid
classDiagram
    class BaseValidator {
        <<abstract>>
        +handle_empty_text(text: T) RuleResult?
        +validate(text: T) RuleResult
    }

    class TextValidator {
        +validate(text: str) RuleResult
    }

    class JSONValidator {
        +validate(data: Dict) RuleResult
    }

    class LengthValidator {
        +min_chars: int
        +max_chars: int
        +validate(text: str) RuleResult
    }

    class StyleValidator {
        +capitalization: str
        +require_punctuation: bool
        +validate(text: str) RuleResult
    }

    class ToxicityValidator {
        +threshold: float
        +validate(text: str) RuleResult
    }

    BaseValidator <|-- TextValidator
    BaseValidator <|-- JSONValidator
    TextValidator <|-- LengthValidator
    TextValidator <|-- StyleValidator
    TextValidator <|-- ToxicityValidator
```

## Rule Configuration

```mermaid
classDiagram
    class RuleConfig {
        +params: Dict[str, Any]
        +priority: int
        +cache_size: int
        +cost: float
    }

    class LengthConfig {
        +min_chars: int
        +max_chars: int
    }

    class StyleConfig {
        +capitalization: str
        +require_punctuation: bool
    }

    class ToxicityConfig {
        +threshold: float
        +categories: List[str]
    }

    RuleConfig <|-- LengthConfig
    RuleConfig <|-- StyleConfig
    RuleConfig <|-- ToxicityConfig
```

## Rule Use Cases

```mermaid
graph TD
    A[Client] --> B{Rule Selection}
    B -->|Formatting| C[Formatting Rules]
    B -->|Content| D[Content Rules]
    B -->|Factual| E[Factual Rules]
    B -->|Domain-specific| F[Domain Rules]

    C --> C1[Length Rule]
    C --> C2[Style Rule]
    C --> C3[Grammar Rule]

    D --> D1[Toxicity Rule]
    D --> D2[Sentiment Rule]
    D --> D3[Topic Rule]

    E --> E1[Facts Rule]
    E --> E2[Citations Rule]
    E --> E3[Consistency Rule]

    F --> F1[Legal Rule]
    F --> F2[Medical Rule]
    F --> F3[Financial Rule]

    C1 --> V[Validation Result]
    C2 --> V
    C3 --> V
    D1 --> V
    D2 --> V
    D3 --> V
    E1 --> V
    E2 --> V
    E3 --> V
    F1 --> V
    F2 --> V
    F3 --> V
```

## Rule Composition

```mermaid
graph TD
    A[Client] --> B[Composite Rule]
    B --> C[Rule 1]
    B --> D[Rule 2]
    B --> E[Rule 3]

    C --> C1[Validator 1]
    D --> D1[Validator 2]
    E --> E1[Validator 3]

    C1 --> R1[Result 1]
    D1 --> R2[Result 2]
    E1 --> R3[Result 3]

    R1 --> M[Merge Results]
    R2 --> M
    R3 --> M

    M --> F[Final Result]

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C1 fill:#bbf,stroke:#333,stroke-width:2px
    style D1 fill:#bbf,stroke:#333,stroke-width:2px
    style E1 fill:#bbf,stroke:#333,stroke-width:2px
    style M fill:#bfb,stroke:#333,stroke-width:2px
```

## Adapter-Rule Integration

```mermaid
sequenceDiagram
    participant Client
    participant ExternalFramework
    participant Adapter
    participant Rule
    participant Validator

    Client->>ExternalFramework: Use framework
    ExternalFramework->>Adapter: Call adapter
    Adapter->>Rule: validate(text)
    Rule->>Validator: validate(text)
    Validator-->>Rule: return result
    Rule-->>Adapter: return result
    Adapter->>Adapter: translate result
    Adapter-->>ExternalFramework: return framework-compatible result
    ExternalFramework-->>Client: return final result
```

These diagrams illustrate:

1. The class relationships between Rules and Validators
2. The factory pattern used to create Rules and Validators
3. The inheritance hierarchy of different Rule and Validator types
4. The delegation flow during validation
5. Rule adaptation to external frameworks
6. Different types of Validators
7. Configuration structure for Rules
8. Rule use cases by category
9. Rule composition for complex validation
10. Integration of Adapters with Rules and external frameworks

The Rule-Validator pattern is central to Sifaka's architecture, providing separation of concerns, reusability, and extensibility.