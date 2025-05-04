# Sifaka Components

This directory contains documentation for specific components in the Sifaka framework.

## Core Components

- [Rules](rules.md) - Binary pass/fail validation of text ([API Reference](../api_reference/rules/README.md))
- [Classifiers](classifiers.md) - Analysis and categorization of text ([API Reference](../api_reference/classifiers/README.md))
- [Critics](critics.md) - Feedback and suggestions for improving text ([API Reference](../api_reference/critics/README.md))
- [Model Providers](model_providers.md) - Connection to language models for text generation ([API Reference](../api_reference/models/README.md))
- [Chains](chains.md) - Orchestration of models, rules, and critics ([API Reference](../api_reference/chain/README.md))
- [Adapters](../api_reference/adapters/README.md) - Integration with external frameworks and systems

## Specific Implementations

### Classifiers

- [NER Classifier](ner_classifier.md) - Named Entity Recognition for identifying entities like people, organizations, and locations in text

### Adapters

- [Guardrails Integration](guardrails_integration.md) - Integration with Guardrails AI for advanced validation capabilities

## Component Relationships

The components in Sifaka work together to create a feedback loop for text generation and improvement:

```mermaid
graph TD
    subgraph Chain
        C[Chain] --> MP[Model Provider]
        C --> R[Rules]
        C --> CR[Critics]
        MP --> |Generate| Text[Text]
        Text --> R
        R --> |Validate| Result[Result]
        Result --> |If Failed| CR
        CR --> |Improve| MP
    end

    subgraph Adapters
        CL[Classifiers] --> |Adapt| R
        GR[Guardrails] --> |Adapt| R
    end
```

## Getting Started

To get started with Sifaka components, see the [README](../../README.md) for installation instructions and basic usage examples.

For detailed documentation on specific components, see the links above.
