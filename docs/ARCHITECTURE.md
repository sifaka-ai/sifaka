# Sifaka Architecture Documentation

This document describes the architecture and design principles of the Sifaka framework.

## Overview

Sifaka is a framework for building AI chains with validation and critique capabilities. The core architecture follows a clean separation of concerns with a central state container (Thought) that flows through different components.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Sifaka Chain                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Thought   │───▶│    Model    │───▶│ Validators  │         │
│  │ (Container) │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                                       │               │
│         │            ┌─────────────┐           │               │
│         └───────────▶│   Critics   │◀──────────┘               │
│                      │             │                           │
│                      └─────────────┘                           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Supporting Components                        │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Retrievers  │    │Classifiers  │    │Persistence  │         │
│  │             │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Flow

### 1. Chain Execution Flow

```
Start
  │
  ▼
┌─────────────────┐
│ Initialize      │
│ Thought         │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Pre-Generation  │
│ Context         │
│ Retrieval       │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Model           │
│ Generation      │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Validation      │
│ Check           │
└─────────────────┘
  │
  ▼
┌─────────────────┐    Yes    ┌─────────────────┐
│ Validation      │──────────▶│ Apply Critics?  │
│ Passed?         │           │ (Optional)      │
└─────────────────┘           └─────────────────┘
  │ No                          │
  ▼                             ▼
┌─────────────────┐           ┌─────────────────┐
│ Apply Critics   │           │ Complete        │
│ for Feedback    │           │ (Success)       │
└─────────────────┘           └─────────────────┘
  │
  ▼
┌─────────────────┐
│ Post-Generation │
│ Context         │
│ Retrieval       │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Max Iterations  │    Yes    ┌─────────────────┐
│ Reached?        │──────────▶│ Complete        │
└─────────────────┘           │ (Max Reached)   │
  │ No                        └─────────────────┘
  ▼
┌─────────────────┐
│ Next Iteration  │
│ (Loop Back)     │
└─────────────────┘
```

### 2. Thought State Management

The Thought object serves as the central state container:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Thought Object                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Core Properties:                                               │
│  • prompt: str                    - Original user prompt        │
│  • text: str                      - Generated text             │
│  • iteration: int                 - Current iteration          │
│  • timestamp: datetime            - Creation time              │
│                                                                 │
│  Context Management:                                            │
│  • pre_generation_context: List[Document]                      │
│  • post_generation_context: List[Document]                     │
│                                                                 │
│  Validation & Critique:                                         │
│  • validation_results: List[ValidationResult]                  │
│  • critic_feedback: List[CriticFeedback]                       │
│                                                                 │
│  History & Metadata:                                            │
│  • metadata: Dict[str, Any]       - Custom metadata           │
│  • model_name: str                - Model used for generation  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### Models

Models are responsible for text generation:

```
┌─────────────────┐
│ Model Protocol  │
├─────────────────┤
│ • name: str     │
│ • generate_     │
│   with_thought()│
│   → (str, str)  │
└─────────────────┘
         △
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──┐
│OpenAI │ │Mock │
│Model  │ │Model│
└───────┘ └─────┘
```

### Validators

Validators check generated text against criteria:

```
┌─────────────────┐
│Validator Protocol│
├─────────────────┤
│ • name: str     │
│ • validate()    │
│   → ValidationResult│
└─────────────────┘
         △
         │
    ┌────┼────┐
    │    │    │
┌───▼─┐ ┌▼──┐ ┌▼────┐
│Length│ │Regex│ │Content│
│Valid.│ │Valid│ │Valid. │
└─────┘ └───┘ └─────┘
```

### Critics

Critics provide feedback for improvement:

```
┌─────────────────┐
│ Critic Protocol │
├─────────────────┤
│ • name: str     │
│ • critique()    │
│   → CriticFeedback│
└─────────────────┘
         △
         │
    ┌────┼────┐
    │    │    │
┌───▼───┐ ┌──▼──┐ ┌▼────┐
│Reflexion│ │Self │ │N-   │
│Critic  │ │RAG  │ │Critics│
└───────┘ └─────┘ └─────┘
```

### Retrievers

Retrievers provide context from external sources:

```
┌─────────────────┐
│Retriever Protocol│
├─────────────────┤
│ • retrieve()    │
│ • retrieve_for_ │
│   thought()     │
└─────────────────┘
         △
         │
    ┌────┼────┐
    │    │    │
┌───▼─┐ ┌▼───┐ ┌▼────┐
│Mock │ │Redis│ │Milvus│
│Retr.│ │Retr.│ │Retr. │
└─────┘ └────┘ └─────┘
```

## Design Principles

### 1. Separation of Concerns

Each component has a single responsibility:
- **Thought**: State management
- **Chain**: Orchestration
- **Models**: Text generation
- **Validators**: Quality checking
- **Critics**: Improvement feedback
- **Retrievers**: Context provision

### 2. Protocol-Based Design

All components implement protocols (interfaces) for:
- Type safety
- Extensibility
- Testability
- Interchangeability

### 3. Immutable State Flow

Thoughts are modified through controlled methods:
- Context is added, not replaced
- History is preserved
- State changes are tracked

### 4. Configurable Behavior

Key behaviors are configurable:
- Maximum iterations
- Critic application strategy
- Validation requirements
- Context retrieval timing

## Context Management

### Pre-Generation Context

Retrieved before model generation:
- Used to inform the model
- Mapped to generated text
- Helps with factual accuracy

### Post-Generation Context

Retrieved after model generation:
- Used by critics for evaluation
- Becomes pre-generation context for next iteration
- Enables fact-checking and improvement

## Error Handling

Sifaka uses structured error handling:

```
┌─────────────────┐
│  SifakaError    │
│   (Base)        │
└─────────────────┘
         △
         │
    ┌────┼────┐
    │    │    │
┌───▼─┐ ┌▼───┐ ┌▼────┐
│Model│ │Valid│ │Critic│
│Error│ │Error│ │Error │
└─────┘ └────┘ └─────┘
```

Each component provides:
- Specific error types
- Detailed error messages
- Context information
- Recovery suggestions

## Performance Considerations

### Caching

- **Classifier Caching**: LRU cache for ML model predictions
- **Retrieval Caching**: Cache for expensive retrieval operations
- **Model Caching**: Response caching for identical inputs

### Async Support

Future versions will support:
- Async model calls
- Parallel validation
- Concurrent critic evaluation

## Extension Points

### Custom Components

Implement protocols to create:
- Custom models
- Domain-specific validators
- Specialized critics
- Novel retrievers

### Plugin System

Future plugin architecture will support:
- Dynamic component loading
- Configuration-driven assembly
- Third-party integrations

## Testing Strategy

### Unit Testing

Each component is tested in isolation:
- Mock dependencies
- Test protocols
- Verify behavior

### Integration Testing

End-to-end testing:
- Full chain execution
- Component interaction
- Real model integration

### Example Testing

All examples are tested:
- Automated execution
- Output validation
- Documentation accuracy
