# Sifaka Framework Components

The Sifaka framework is designed around a modular architecture with several key components that work together to provide a powerful and flexible system for text generation, validation, and improvement. This document provides an overview of these components and how they interact.

## Core Components

Sifaka consists of the following primary components:

1. **Chain** - The central orchestrator that coordinates the entire process
2. **Models** - Language model integrations for text generation
3. **Validators** - Components that check if text meets specific criteria
4. **Critics** - Components that improve text quality through various techniques
5. **Retrievers** - Components that retrieve relevant information for queries
6. **Classifiers** - Components that categorize text into specific classes
7. **Results** - Structured objects that represent operation outcomes

## Component Relationships

The components interact in the following way:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │
│    Model    │────▶│    Chain    │────▶│  Validators │────▶│   Critics   │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │                    ▲                   │
                          │                    │                   │
                          ▼                    │                   ▼
                    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
                    │             │     │             │     │             │
                    │   Results   │     │ Classifiers │     │ Retrievers  │
                    │             │     │             │     │             │
                    └─────────────┘     └─────────────┘     └─────────────┘
                          ▲                                        │
                          │                                        │
                          └────────────────────────────────────────┘
```

## Chain

The Chain is the central component of the Sifaka framework. It orchestrates the process of:
1. Generating text using a language model
2. Validating the text against specified criteria
3. Improving the text using specialized critics

The Chain follows a fluent interface pattern (builder pattern) for easy configuration, allowing you to chain method calls to set up the desired behavior.

[Learn more about Chain](CHAIN.md)

## Models

Models are integrations with various language model providers (like OpenAI, Anthropic, Google Gemini) that handle text generation. Each model implementation follows the Model protocol, which defines a consistent interface for all models.

The key responsibilities of models include:
- Generating text from prompts
- Counting tokens in text
- Handling API communication with the provider

[Learn more about Models](MODELS.md)

## Validators

Validators check if text meets specific criteria, such as length, content, or format requirements. They return validation results that indicate whether the text passed or failed validation, along with details about any issues found.

Validators are used by the Chain to ensure that generated text meets the required criteria before being returned to the user or passed to critics for improvement.

[Learn more about Validators](VALIDATORS.md)

## Critics

Critics (also called Improvers) enhance the quality of text by applying various improvement strategies. They analyze text, identify areas for improvement, and generate improved versions.

Critics are used by the Chain after validation to improve the quality of the generated text. They can be chained together to apply multiple improvement strategies in sequence.

[Learn more about Critics](CRITICS.md)

## Results

Results are structured objects that represent the outcome of operations in the Sifaka framework. There are three main types of results:

1. **ValidationResult** - Represents the outcome of validating text against specific criteria
2. **ImprovementResult** - Represents the outcome of improving text using a critic
3. **Result** - Represents the complete outcome of executing a Chain

Results provide detailed information about the operation, including whether it succeeded, any issues found, and suggestions for improvement.

## Configuration

Sifaka provides a configuration system that allows you to customize the behavior of the framework components. The configuration can be provided when creating a Chain instance and will be applied to all components used by the chain.

## Registry System

Sifaka uses a registry system for dependency injection, allowing components to be created and configured dynamically. This makes it easy to extend the framework with new components and to use different implementations of existing components.

## Retrievers

Retrievers are components that retrieve relevant information for queries. They are primarily used by retrieval-augmented critics like Self-RAG to enhance text generation with external knowledge.

Retrievers implement a common interface defined by the `Retriever` abstract base class, which requires implementing a `retrieve` method that takes a query string and returns a list of relevant document texts.

The Sifaka framework provides several retriever implementations, including:
- SimpleRetriever - A basic retriever that searches through a predefined collection of documents
- ElasticsearchRetriever - A retriever that uses Elasticsearch for document retrieval
- MilvusRetriever - A retriever that uses Milvus vector database for semantic search

[Learn more about Retrievers](RETRIEVERS.md)

## Classifiers

Classifiers are components that categorize text into specific classes or labels. They can be used directly or adapted into validators using the classifier validator adapter.

Classifiers implement a common interface defined by the `Classifier` protocol, which requires implementing methods for classifying text and providing metadata about the classifier.

The Sifaka framework provides several classifier implementations, including:
- SentimentClassifier - Classifies text sentiment as positive, negative, or neutral
- ToxicityClassifier - Detects toxic content in text
- SpamClassifier - Identifies spam or unwanted content
- ProfanityClassifier - Detects profane language in text
- LanguageClassifier - Identifies the language of text

[Learn more about Classifiers](CLASSIFIERS.md)

## Next Steps

To learn more about each component, follow the links to the detailed documentation:

- [Chain Documentation](CHAIN.md)
- [Models Documentation](MODELS.md)
- [Validators Documentation](VALIDATORS.md)
- [Critics Documentation](CRITICS.md)
- [Retrievers Documentation](RETRIEVERS.md)
- [Classifiers Documentation](CLASSIFIERS.md)
