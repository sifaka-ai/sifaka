# Sifaka Migration TODO List

This document tracks the progress of migrating the Sifaka library to the new architecture based on the Thought container design.

## Core Architecture Understanding

A Chain in Sifaka consists of:
- **Thought**: Central state container that flows through the system
- **Model**: Generates text based on the prompt and context in the Thought
- **Validators**: Check if the generated text meets specific criteria
- **Critics**: Improve text that fails validation by providing feedback

```
Chain = [Thought, Model, [Validators], [Critics]]
```

The flow of execution is:
1. A Thought is created with a Prompt
2. Retrievers may be called to add Pre-generation Context to the Thought
3. The Model receives the Thought and generates Text
4. Validators check the Text and add Validation Results to the Thought
5. If validation fails, Critics receive the Thought, may call Retrievers, and add feedback
6. The cycle repeats until validation passes or max iterations are reached

## Phase 1: Code Relocation

- [x] Move the current `sifaka` directory to `sifaka_legacy`
- [x] Create a new empty `sifaka` directory
- [x] Set up the basic package structure for the new implementation

## Phase 2: Core Implementation

- [x] Create the `Thought` container as a Pydantic 2 model
- [x] Implement the core interfaces for models, critics, validators, and retrievers
- [x] Build the new Chain class that uses the Thought container

## Phase 3: Component Migration

- [x] Implement basic model interface (MockModel)
- [x] Complete model implementations:
  - [x] OpenAI model implementation
  - [x] Anthropic model implementation
- [x] Implement basic validators
  - [x] LengthValidator
  - [x] RegexValidator
  - [ ] ContentValidator
  - [ ] FormatValidator
  - [ ] ClassifierValidator
  - [ ] GuardrailsValidator
- [x] Implement basic critics
  - [x] ReflexionCritic
  - [ ] SelfRAGCritic
  - [ ] SelfRefineCritic
  - [ ] ConstitutionalCritic
  - [ ] PromptCritic
  - [ ] NCriticsCritics
- [x] Implement basic retrieval mechanisms
  - [x] MockRetriever
  - [x] InMemoryRetriever
  - [ ] VectorDBRetriever (Milvus, Redis)
- [ ] Implement persistence layer
  - [ ] JSON persistence
  - [ ] Redis persistence
  - [ ] Milvus persistence


WE ARE NOT MIGRATING THE ELASTICSEARCH PARTS OF THE CODEBASE

## Key Architectural Improvements

- [x] **Unified State Container**: Implement the Thought container as the central state mechanism
- [ ] **Universal Retrieval**: Make retrievers available to both models and critics
  - [x] Implement retriever interface
  - [ ] Update models to use retrievers directly
  - [ ] Update critics to use retrievers directly
  - [ ] Ensure retrievers can be called at any point in the chain
- [ ] **Clean Dependency Injection**:
  - [ ] Central service registry
  - [ ] Standardized component registration
  - [ ] Elimination of circular dependencies
  - [ ] Clear component lifecycle management
- [x] **Consistent Error Handling**:
  - [x] Standardize error types
  - [x] Detailed context for failures
  - [x] Proper propagation of errors
  - [x] Helpful error messages and recovery suggestions
- [ ] **Improved Configuration**:
  - [ ] Implement centralized configuration management
  - [ ] Environment variable integration
  - [ ] Configuration validation
- [x] **Better Validation Feedback**: Ensure proper use of validation results
- [x] **Simplified API**: Create a more intuitive and consistent API

## Pydantic 2 Integration

- [x] Use Pydantic 2 for the Thought container
- [ ] Ensure all models use Pydantic 2 features:
  - [ ] Use `model_config` instead of `Config` class
  - [ ] Use `model_dump` instead of `dict()`
  - [ ] Use `model_validate` instead of `parse_obj`
  - [ ] Use `Field` for field definitions
  - [ ] Use `model_dump_json` and `model_validate_json` for serialization
- [ ] Add JSON schema generation for all models
- [ ] Implement validation with custom validators using Pydantic 2 syntax

## Testing

- [ ] Unit tests for all components:
  - [ ] Thought container tests
  - [ ] Chain tests
  - [ ] Model tests
  - [ ] Validator tests
  - [ ] Critic tests
  - [ ] Retriever tests
- [ ] Integration tests for component interactions
- [ ] End-to-end tests for complete chains
- [ ] Performance benchmarks

## Documentation

- [x] Update README.md with new architecture
- [ ] API reference for all components
- [ ] Architecture documentation with diagrams
- [ ] Migration guide for users of the legacy library
- [ ] Examples and tutorials:
  - [x] Basic chain example
  - [ ] Retrieval-enhanced generation example
  - [ ] Critic-based improvement example
  - [ ] Custom validator example
  - [ ] Persistence example

## Additional Tasks

- [ ] Set up proper package structure with setup.py
- [ ] Configure CI/CD pipeline
- [ ] Add type hints throughout the codebase
- [ ] Run mypy for type checking
- [ ] Implement logging throughout the codebase
- [ ] Add docstrings to all classes and methods
- [ ] Create CONTRIBUTING.md with development guidelines
- [ ] Add license file

## Next Steps

1. ✅ Complete the OpenAI and Anthropic model implementations
2. Implement the remaining critics and validators
3. Add the persistence layer
4. Write comprehensive tests
5. Complete the documentation
6. Set up the package for distribution

## Notes

- Ensure all new code uses Pydantic 2 syntax and features
- Maintain consistent error handling across all components
- Keep the Thought container independent from utility modules
- Follow consistent naming conventions throughout the codebase
- Write comprehensive docstrings for all public APIs
