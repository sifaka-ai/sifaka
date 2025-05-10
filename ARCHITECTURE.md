# Sifaka Architecture: Components and Relationships

## Overview

Sifaka is a modular framework for text generation, validation, and improvement. It follows a component-based architecture where different modules work together to provide a complete pipeline for text processing. This document explains how the various components relate to each other and how they integrate with the chain system.

## Core Components

### 1. Core

The `core` directory contains foundational components used across the framework:

- **BaseComponent**: Abstract base class for all components with state management
- **BaseConfig**: Configuration for components
- **BaseResult**: Result container for component operations
- **Generator**: Text generation using model providers
- **Improver**: Text improvement using model providers
- **Validator**: Text validation against rules

### 2. Models

The `models` directory provides interfaces to language models:

- **ModelProviderCore**: Base class for model providers
- **AnthropicProvider/OpenAIProvider**: Specific implementations for different LLM providers
- **TokenCounter**: Counts tokens for optimization
- **APIClient**: Handles API communication

### 3. Rules

The `rules` directory contains components for validating text against specific criteria:

- **Rule**: Base class for all rules
- **BaseValidator**: Implements validation logic
- **RuleResult**: Contains validation results
- **ValidationManager**: Manages multiple rules

Rules are organized by type:
- **Content Rules**: Validate content semantics (prohibited content, language, safety, sentiment, tone)
- **Formatting Rules**: Validate text structure (format, length, structure, style, whitespace)

### 4. Critics

The `critics` directory provides components for critiquing and improving text:

- **CriticCore**: Central implementation for text validation and improvement
- **CriticConfig**: Configuration for critics
- **CriticMetadata**: Contains critique results
- **PromptManager**: Manages prompts for critics
- **ResponseParser**: Parses model responses
- **MemoryManager**: Manages interaction history

### 5. Chain

The `chain` directory implements the orchestration of components:

- **ChainCore**: Coordinates model providers, validators, and critics
- **ValidationManager**: Manages rule validation in chains
- **PromptManager**: Manages prompts for chains
- **RetryStrategy**: Implements retry logic for failed generations
- **ResultFormatter**: Formats chain results

### 6. Adapters

The `adapters` directory provides components that adapt external systems to Sifaka:

- **BaseAdapter**: Adapts external components to Sifaka validators
- **Adaptable**: Protocol for components that can be adapted

### 7. Classifiers

The `classifiers` directory contains components for text classification:

- **BaseClassifier**: Base class for all classifiers
- **ClassificationResult**: Contains classification results
- **ClassifierConfig**: Configuration for classifiers

### 8. Retrieval

The `retrieval` directory provides components for retrieving information:

- **RetrieverCore**: Base class for retrievers
- **RetrievalResult**: Contains retrieval results
- **QueryManager**: Processes queries for retrieval

### 9. Interfaces

The `interfaces` directory defines protocols and interfaces used across components:

- **Rule Interface**: Defines the interface for rules
- **Critic Interface**: Defines the interface for critics
- **Chain Interface**: Defines the interface for chains
- **Classifier Interface**: Defines the interface for classifiers
- **Retriever Interface**: Defines the interface for retrievers

### 10. Utils

The `utils` directory provides utility functions and classes:

- **StateManager**: Manages component state
- **Logging**: Provides logging functionality
- **Tracing**: Provides tracing functionality
- **Errors**: Standardized error handling

## Component Relationships

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                 Chain                                    │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ ChainCore   │  │ Validation  │  │ Prompt      │  │ Result      │     │
│  │             │  │ Manager     │  │ Manager     │  │ Formatter   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
└───────────┬─────────────────┬─────────────────┬─────────────────────────┘
            │                 │                 │
            ▼                 ▼                 ▼
┌───────────────────┐ ┌─────────────────┐ ┌────────────────┐
│      Models       │ │     Rules       │ │    Critics     │
│                   │ │                 │ │                │
│ ┌───────────────┐ │ │ ┌─────────────┐ │ │ ┌────────────┐ │
│ │ ModelProvider │ │ │ │ Rule        │ │ │ │ CriticCore │ │
│ └───────────────┘ │ │ └─────────────┘ │ │ └────────────┘ │
│                   │ │                 │ │                │
│ ┌───────────────┐ │ │ ┌─────────────┐ │ │ ┌────────────┐ │
│ │ TokenCounter  │ │ │ │ BaseValidator│ │ │ │ Critique   │ │
│ └───────────────┘ │ │ └─────────────┘ │ │ │ Service    │ │
│                   │ │                 │ │ └────────────┘ │
└───────────────────┘ └─────────────────┘ └────────────────┘
            ▲                 ▲                 ▲
            │                 │                 │
┌───────────┴─────────────────┴─────────────────┴────────────────┐
│                           Core                                  │
│                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ BaseComponent│  │ BaseConfig  │  │ BaseResult  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└──────────────────────────────────────────────────────────────────┘
            │                 │                 │
            ▼                 ▼                 ▼
┌───────────────────┐ ┌─────────────────┐ ┌────────────────┐
│    Classifiers    │ │    Adapters     │ │   Retrieval    │
│                   │ │                 │ │                │
│ ┌───────────────┐ │ │ ┌─────────────┐ │ │ ┌────────────┐ │
│ │ BaseClassifier│ │ │ │ BaseAdapter │ │ │ │ Retriever  │ │
│ └───────────────┘ │ │ └─────────────┘ │ │ │ Core       │ │
│                   │ │                 │ │ └────────────┘ │
└───────────────────┘ └─────────────────┘ └────────────────┘
```

## Flow of Operations

### Text Generation and Validation Flow

1. **Chain Initialization**:
   - ChainCore is initialized with a model provider, validation manager, prompt manager, and critic
   - Each component is configured with appropriate parameters

2. **Chain Execution**:
   - User provides a prompt to the chain
   - PromptManager formats the prompt
   - ModelProvider generates text from the prompt
   - ValidationManager validates the generated text using rules
   - If validation fails and a critic is provided, the text is improved
   - ResultFormatter formats the final result

3. **Rule Validation**:
   - ValidationManager applies each rule to the text
   - Each rule delegates to its validator for the actual validation
   - Results are collected and returned to the chain

4. **Critic Improvement**:
   - Critic analyzes the text and validation results
   - CritiqueService generates improvement suggestions
   - Improved text is generated and returned to the chain

## Integration Points

1. **Chain and Models**:
   - Chain uses model providers for text generation
   - Models provide a unified interface for different LLM providers

2. **Chain and Rules**:
   - Chain uses ValidationManager to apply rules
   - Rules provide binary validation (pass/fail)

3. **Chain and Critics**:
   - Chain uses critics for text improvement
   - Critics provide nuanced feedback and improvement suggestions

4. **Rules and Classifiers**:
   - Rules can use classifiers for content validation
   - Classifiers provide semantic understanding and categorization

5. **Critics and Retrieval**:
   - Critics can use retrievers for additional context
   - Retrievers provide relevant information for text improvement

6. **Adapters and External Systems**:
   - Adapters connect external systems to Sifaka
   - External systems can be used as validators, classifiers, or retrievers
