# Component Relationship Diagrams

This document provides Mermaid diagrams showing the relationships between key Sifaka components.

## High-Level Component Relationships

```mermaid
graph TD
    Client[Client Application] --> Adapters
    Client --> Chain
    Adapters --> Chain
    Chain --> Models
    Chain --> Classifiers
    Chain --> Critics
    Chain --> Rules
    Classifiers --> Models
    Critics --> Models
    Rules --> Classifiers
    subgraph "Core Components"
        Models
        Classifiers
        Critics
        Rules
    end
    subgraph "Integration Layer"
        Chain
        Adapters
    end
    subgraph "Client Layer"
        Client
    end
```

## Classifier Inheritance Hierarchy

```mermaid
classDiagram
    BaseClassifier <|-- ToxicityClassifier
    BaseClassifier <|-- SentimentClassifier
    BaseClassifier <|-- LanguageClassifier
    BaseClassifier <|-- GenreClassifier
    BaseClassifier <|-- BiasClassifier
    BaseClassifier <|-- SpamClassifier
    BaseClassifier <|-- ProfanityClassifier
    BaseClassifier <|-- ReadabilityClassifier
    BaseClassifier <|-- TopicClassifier
    BaseClassifier <|-- NERClassifier

    class BaseClassifier {
        +name: str
        +description: str
        +config: ClassifierConfig
        +classify(text): ClassificationResult
        +batch_classify(texts): List[ClassificationResult]
        #_classify_impl_uncached(text): ClassificationResult
    }

    class ToxicityClassifier {
        -_model: ToxicityModel
        +warm_up()
        #_classify_impl_uncached(text): ClassificationResult
    }

    class SentimentClassifier {
        -_analyzer: SentimentAnalyzer
        +warm_up()
        #_classify_impl_uncached(text): ClassificationResult
    }
```

## Rule Component Relationships

```mermaid
classDiagram
    Rule o-- Validator
    Rule o-- RuleConfig
    BaseValidator <|-- ContentValidator
    BaseValidator <|-- FormattingValidator
    BaseValidator <|-- FactualValidator

    class Rule {
        +name: str
        +description: str
        +config: RuleConfig
        +validate(text): RuleResult
    }

    class Validator {
        +validate(text): RuleResult
        +can_validate(text): bool
    }

    class BaseValidator {
        +validate(text): RuleResult
        +handle_empty_text(text): RuleResult
        +can_validate(text): bool
    }
```

## Request Processing Sequence

```mermaid
sequenceDiagram
    participant Client
    participant Rule
    participant Validator
    participant Classifier
    participant Model

    Client->>Rule: validate(text)
    Rule->>Validator: validate(text)
    alt Uses Classifier
        Validator->>Classifier: classify(text)
        Classifier->>Model: generate(prompt)
        Model-->>Classifier: response
        Classifier-->>Validator: classification
    end
    Validator-->>Rule: validation result
    Rule-->>Client: rule result
```

## Critic Workflow

```mermaid
sequenceDiagram
    participant Client
    participant Critic
    participant Rule
    participant Model

    Client->>Critic: critique(text)
    Critic->>Model: analyze(text)
    Model-->>Critic: analysis
    Critic-->>Client: critique result

    alt Needs Improvement
        Client->>Critic: improve(text, feedback)
        Critic->>Rule: validate(text)
        Rule-->>Critic: validation result
        Critic->>Model: improve(text, feedback)
        Model-->>Critic: improved text
        Critic-->>Client: improved text
    end
```

## Adapter Pattern

```mermaid
classDiagram
    LangChainAdapter o-- SifakaChain
    LangGraphAdapter o-- SifakaChain
    SifakaChain o-- Rule
    SifakaChain o-- Classifier
    SifakaChain o-- Critic

    class LangChainAdapter {
        +run(input): output
    }

    class LangGraphAdapter {
        +build_graph(): Graph
    }

    class SifakaChain {
        +process(input): output
    }
```

## Chain Architecture

```mermaid
classDiagram
    ChainCore o-- Model
    ChainCore o-- ValidationManager
    ChainCore o-- PromptManager
    ChainCore o-- RetryStrategy
    ChainCore o-- ResultFormatter
    ChainCore o-- Critic

    ValidationManager o-- Rule
    RetryStrategy <|-- SimpleRetryStrategy
    RetryStrategy <|-- BackoffRetryStrategy

    class ChainCore {
        +model: Model
        +validation_manager: ValidationManager
        +prompt_manager: PromptManager
        +retry_strategy: RetryStrategy
        +result_formatter: ResultFormatter
        +critic: Critic
        +run(prompt: str): ChainResult
    }

    class ValidationManager {
        +rules: List[Rule]
        +validate(text): ValidationResult
    }

    class PromptManager {
        +create_prompt_with_feedback(prompt, feedback): str
    }

    class RetryStrategy {
        <<abstract>>
        +run(prompt, generator, validation_manager, prompt_manager): ChainResult
    }

    class SimpleRetryStrategy {
        +max_attempts: int
    }

    class BackoffRetryStrategy {
        +max_attempts: int
        +initial_backoff: float
        +backoff_factor: float
        +max_backoff: float
    }

    class ResultFormatter {
        +format(output, rule_results, critique_details): ChainResult
    }
```

## Chain Processing Flow

```mermaid
sequenceDiagram
    participant Client
    participant Chain
    participant Model
    participant ValidationManager
    participant Rule
    participant Critic

    Client->>Chain: run(prompt)
    Chain->>Model: generate(prompt)
    Model-->>Chain: response

    Chain->>ValidationManager: validate(response)
    ValidationManager->>Rule: validate(response)
    Rule-->>ValidationManager: rule_result
    ValidationManager-->>Chain: validation_result

    alt Validation Failed
        Chain->>Critic: improve(response, feedback)
        Critic-->>Chain: improved_text

        loop Until success or max attempts
            Chain->>Model: generate(updated_prompt)
            Model-->>Chain: response
            Chain->>ValidationManager: validate(response)
            ValidationManager-->>Chain: validation_result
        end
    end

    Chain-->>Client: chain_result
```

## Chain Factory Pattern

```mermaid
sequenceDiagram
    participant Client
    participant Factory
    participant ChainCore
    participant Components

    Client->>Factory: create_simple_chain(model, rules, critic)
    Factory->>Components: create specialized components
    Components-->>Factory: return components
    Factory->>ChainCore: create(model, validation_manager, prompt_manager, retry_strategy, result_formatter, critic)
    ChainCore-->>Factory: return chain
    Factory-->>Client: return configured chain
```

These diagrams provide a visual representation of the key relationships and interactions between Sifaka components. They can be rendered directly in GitHub or any Markdown viewer that supports Mermaid.

```mermaid
classDiagram
    %% Model Components
    class ModelProviderCore {
        <<interface>>
        +get_provider_name()
        +generate(prompt, params)
    }

    class ModelProvider {
        <<abstract>>
        +model_name: str
        +config: ModelConfig
        +get_default_config()
        +generate(prompt, params)
    }

    class AnthropicProvider {
        +get_provider_name()
        +generate(prompt, params)
    }

    class OpenAIProvider {
        +get_provider_name()
        +generate(prompt, params)
    }

    %% Rule Components
    class Rule {
        <<abstract>>
        +name: str
        +description: str
        +id: str
        +validator: RuleValidator
        +validate(text)
    }

    class RuleValidator {
        <<interface>>
        +validate(text)
    }

    class LengthRule {
        +validate(text)
    }

    class ToxicityRule {
        +validate(text)
    }

    %% Classifier Components
    class Classifier {
        <<abstract>>
        +name: str
        +description: str
        +config: ClassifierConfig
        +classify(text)
    }

    class ToxicityClassifier {
        +classify(text)
    }

    class SentimentClassifier {
        +classify(text)
    }

    %% Critic Components
    class TextCritic {
        <<interface>>
        +critique(text)
        +improve(text, feedback)
    }

    class PromptCritic {
        +system_prompt: str
        +critique(text)
        +improve(text, feedback)
    }

    class ReflexionCritic {
        +memory: Memory
        +critique(text)
        +improve(text, feedback)
        +reflect(text, result)
    }

    %% Chain Components
    class ChainCore {
        +model: ModelProvider
        +run(prompt)
    }

    class ChainOrchestrator {
        +model: ModelProvider
        +rules: List[Rule]
        +critic: TextCritic
        +run(prompt)
    }

    %% Adapter Components
    class RuleAdapter {
        <<interface>>
        +adapt(component)
        +validate(text)
    }

    class ClassifierAdapter {
        +classifier: Classifier
        +validate(text)
    }

    class GuardrailsAdapter {
        +rules: List[Rule]
        +validate(text)
        +run(model, prompt)
    }

    %% Relationships
    ModelProviderCore <|.. ModelProvider
    ModelProvider <|-- AnthropicProvider
    ModelProvider <|-- OpenAIProvider

    RuleValidator --o Rule
    Rule <|-- LengthRule
    Rule <|-- ToxicityRule

    Classifier <|-- ToxicityClassifier
    Classifier <|-- SentimentClassifier

    TextCritic <|.. PromptCritic
    TextCritic <|.. ReflexionCritic

    ChainCore <|-- ChainOrchestrator

    RuleAdapter <|.. ClassifierAdapter
    RuleAdapter <|.. GuardrailsAdapter

    ModelProvider --o ChainCore
    Rule --o ChainOrchestrator
    TextCritic --o ChainOrchestrator
    ChainCore <|-- ChainOrchestrator

    Classifier --o ClassifierAdapter
    Rule --o GuardrailsAdapter
```

```mermaid
graph TD
    SifakaChain[SifakaChain] --> Model
    SifakaChain --> Rule
    SifakaChain --> Classifier
    SifakaChain --> Critic

    ClassifierAdapter[ClassifierAdapter] --> Classifier
    ClassifierAdapter --> Rule

    GuardrailsAdapter[GuardrailsAdapter] --> Rule
    GuardrailsAdapter --> SifakaChain

    class SifakaChain {
        +process(input): output
    }

    class Rule {
        +validate(text): result
    }

    class Classifier {
        +classify(text): result
    }

    class Critic {
        +improve(text, feedback): improved_text
    }

    class ClassifierAdapter {
        +validate(text): result
    }

    class GuardrailsAdapter {
        +run(prompt): result
    }

    class Model {
        +generate(prompt): text
    }
```