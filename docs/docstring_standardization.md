# Docstring Standardization Tracking

This document tracks the progress of standardizing docstrings across all Sifaka components. The goal is to ensure all components have comprehensive, consistent docstrings that follow the standards outlined in [contributing.md](contributing.md).

## Standards Summary

All docstrings should include:

1. **Brief description**: A concise summary of the component's purpose
2. **Detailed description**: A comprehensive explanation of the component's functionality
3. **Architecture**: Description of the component's design and integration with other components
4. **Lifecycle**: Explanation of initialization, operation, and cleanup
5. **Error handling**: Description of how errors are handled
6. **Examples**: Code examples showing common usage patterns
7. **Attributes/Parameters**: Description of all attributes and parameters
8. **Return values**: Description of return values for methods
9. **Raises**: Description of exceptions that may be raised

## Progress Tracking

### Core Components

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| BaseRule | ✅ Complete | High | |
| BaseValidator | ✅ Complete | High | |
| BaseClassifier | ✅ Complete | High | |
| BaseCritic | ✅ Complete | High | |
| BaseModelProvider | ✅ Complete | High | |
| BaseChain | ✅ Complete | High | |
| BaseAdapter | ✅ Complete | High | |

### Rules

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| LengthRule | ✅ Complete | High | |
| ToxicityRule | ✅ Complete | High | |
| ProfanityRule | ✅ Complete | High | |
| BiasRule | ✅ Complete | High | |
| FormatRule | ✅ Complete | Medium | |
| ConsistencyRule | ✅ Complete | Medium | |
| PythonRule | ✅ Complete | Low | |

### Classifiers

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| ToxicityClassifier | ✅ Complete | High | |
| ProfanityClassifier | ✅ Complete | High | |
| SpamClassifier | ✅ Complete | High | |
| BiasDetector | ✅ Complete | High | |
| NERClassifier | ✅ Complete | Medium | |

### Critics

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| PromptCritic | ✅ Complete | High | |
| ReflexionCritic | ✅ Complete | High | |
| SelfRefineCritic | ✅ Complete | High | |
| SelfRAGCritic | ✅ Complete | High | |
| LACCritic | ✅ Complete | Medium | |
| ConstitutionalCritic | ✅ Complete | High | |

### Model Providers

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| OpenAIChatProvider | ✅ Complete | High | |
| AnthropicProvider | ✅ Complete | High | |
| HuggingFaceProvider | ⚠️ Needs Examples | Medium | Missing usage examples |
| AzureOpenAIProvider | ⚠️ Needs Examples | Medium | Missing usage examples |


### Chains

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| SimpleChain | ✅ Complete | High | |

### Adapters

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| ClassifierAdapter | ✅ Complete | High | |
| GuardrailsAdapter | ✅ Complete | Medium | |
| PydanticAdapter | ✅ Complete | Low | |

## Next Steps

1. **High Priority Components**: All high priority components have been completed.
2. **Medium Priority Components**: All medium priority components have been completed.
3. **Low Priority Components**: Add examples for HuggingFaceProvider and AzureOpenAIProvider.

## Completion Criteria

A component is considered to have complete docstrings when:

1. All classes, methods, and functions have docstrings
2. Docstrings follow the format specified in contributing.md
3. Docstrings include all required sections
4. Examples are provided for all common use cases
5. Examples have been verified to work correctly

## Verification Process

To verify that docstrings are complete and correct:

1. Run the doctest examples to ensure they work
2. Review docstrings for completeness and accuracy
3. Update this tracking document with the current status
