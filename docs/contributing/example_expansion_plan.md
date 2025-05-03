# Example Expansion Plan

This document outlines the plan for expanding the examples in the Sifaka codebase. It identifies areas where new examples are needed and prioritizes them based on user needs.

## Current Status

| Example Type | Status | Notes |
|--------------|--------|-------|
| Basic Usage | ✅ Complete | Simple examples for core functionality |
| Classifier Integration | ✅ Complete | Examples for using classifiers |
| Rules | 🟡 Partial | Missing examples for some rule types |
| Critics | 🟡 Partial | Missing examples for some critic types |
| Chains | 🟡 Partial | Missing examples for some chain types |
| Guardrails Integration | ✅ Complete | Examples for Guardrails |
| Error Handling | 🔴 Missing | No examples for error handling |

## Example Expansion Priorities

The following are prioritized based on user needs and complexity:

### Highest Priority (P0)

- [x] Basic Rule Usage
- [x] Toxicity Classification
- [x] Chain Architecture
- [x] Claude Integration
- [x] OpenAI GPT Integration
- [ ] Error Handling and Recovery

### High Priority (P1)

- [ ] Domain-Specific Rules
- [ ] Advanced Rules Composition
- [ ] ReflexionCritic Usage
- [ ] Multi-step Chain Processing
- [ ] Customizing Validators
- [ ] Benchmarking Example

### Medium Priority (P2)

- [ ] Custom Classifier Implementation
- [ ] Response Formatting
- [ ] Advanced Chain Configuration
- [ ] Logging and Tracing
- [ ] Performance Optimization
- [ ] Guardrails Advanced Usage

### Lower Priority (P3)

- [ ] Testing Examples
- [ ] Factory Function Patterns
- [ ] Advanced Configuration
- [ ] CLI Tool Example
- [ ] Custom Model Integration
- [ ] Real-world Applications

## Example Structure

Each example should follow this structure:

1. **File Header** - Description, purpose, and requirements
2. **Imports** - All necessary imports
3. **Configuration** - Any required setup or API keys
4. **Implementation** - The actual example code with comments
5. **Output Handling** - How to process and display results
6. **Error Handling** - How to handle common errors
7. **Cleanup** - Any necessary cleanup

## Implementation Plan

| Example | Assigned To | Target Date | Status |
|---------|------------|-------------|--------|
| Error Handling | @quality-team | June 15 | 🟡 In Progress |
| Domain-Specific Rules | @rule-expert | June 22 | 🔴 Not Started |
| Advanced Rules Composition | @rule-expert | June 30 | 🔴 Not Started |
| ReflexionCritic Usage | @ml-team | July 7 | 🔴 Not Started |

## Example Categories

### Basic Examples

- [x] Simple Rule Example
- [x] Simple Chain Example
- [x] Simple Critic Example
- [x] Claude Integration Example
- [x] OpenAI Integration Example
- [ ] Gemini Integration Example

### Intermediate Examples

- [x] Toxicity Classification
- [x] Sentiment Analysis
- [x] Rule Composition
- [ ] Custom Rule Creation
- [ ] Advanced Chain Configuration
- [ ] Error Handling and Recovery

### Advanced Examples

- [ ] Multi-stage Processing Pipeline
- [ ] Custom Critic Implementation
- [ ] Response Formatting for Different Outputs
- [ ] Performance Optimization
- [ ] Benchmarking and Evaluation
- [ ] Domain-Specific Applications

## Success Metrics

- Each example category has at least 3 examples
- All examples have consistent structure and documentation
- Examples cover at least 90% of Sifaka's core functionality
- Feedback from users indicates examples are helpful
- Reduction in support questions about basic functionality

## Review Process

Each example should be reviewed for:
1. Technical accuracy and best practices
2. Documentation quality and clarity
3. Completeness (covers the feature adequately)
4. Error handling and edge cases
5. Consistency with other examples
