# Example Expansion Plan

This document outlines the plan for expanding Sifaka's examples to demonstrate its capabilities and help users understand how to use the framework effectively.

## Current State

Sifaka currently has a few examples, but needs a more comprehensive set covering all major components and common use cases.

## Goals

1. Create a complete set of examples covering all major components
2. Develop examples for common use cases
3. Provide advanced examples demonstrating best practices
4. Ensure all examples follow consistent standards
5. Make examples accessible to users with different levels of experience

## Example Categories

### Basic Examples

- [x] Length Validation Example (`claude_length_critic.py`)
- [x] Length Expansion Example (`claude_expand_length_critic.py`)
- [x] Toxicity Filtering Example (`toxicity_filtering.py`)
- [x] Reflexion Critic Example (`reflexion_critic_example.py`)
- [ ] Simple Rule Example
- [ ] Multiple Rules Example
- [ ] Basic Chain Example

### Component Examples

- [ ] Rules Examples
  - [ ] Formatting Rules Example
  - [ ] Content Rules Example
  - [ ] Factual Rules Example
  - [ ] Domain-Specific Rules Example
  
- [ ] Validators Examples
  - [ ] Custom Validator Example
  - [ ] Validator Composition Example
  
- [ ] Classifiers Examples
  - [ ] Sentiment Classifier Example
  - [ ] Toxicity Classifier Example
  - [ ] NER Classifier Example
  - [ ] Classifier as Rule Example
  
- [ ] Critics Examples
  - [ ] Prompt Critic Example
  - [ ] Reflexion Critic Example
  - [ ] Style Critic Example
  - [ ] Custom Critic Example
  
- [ ] Chains Examples
  - [ ] Simple Chain Example
  - [ ] Validation Chain Example
  - [ ] Improvement Chain Example
  - [ ] Custom Chain Example
  
- [ ] Model Providers Examples
  - [ ] OpenAI Provider Example
  - [ ] Anthropic Provider Example
  - [ ] Gemini Provider Example
  - [ ] Custom Provider Example

### Use Case Examples

- [ ] Content Moderation Example
- [ ] Text Summarization Example
- [ ] Style Transfer Example
- [ ] Question Answering Example
- [ ] Code Generation Example
- [ ] Multi-step Processing Example

### Integration Examples

- [ ] LangChain Integration Example
- [ ] LangGraph Integration Example
- [ ] Guardrails Integration Example
- [ ] Custom Integration Example

### Advanced Examples

- [ ] Advanced Chain Configuration Example
- [ ] Custom Component Example
- [ ] Performance Optimization Example
- [ ] Error Handling Example
- [ ] Testing Example

## Implementation Plan

### Phase 1: Basic Examples (Weeks 1-2)

Focus on simple examples that demonstrate core functionality:
- [ ] Simple Rule Example
- [ ] Multiple Rules Example
- [ ] Basic Chain Example
- [ ] Basic Classifier Example
- [ ] Basic Critic Example

### Phase 2: Component Examples (Weeks 3-4)

Expand to cover all major components in detail:
- [ ] Rules Examples (all types)
- [ ] Validators Examples
- [ ] Classifiers Examples
- [ ] Critics Examples
- [ ] Chains Examples
- [ ] Model Providers Examples

### Phase 3: Use Case Examples (Weeks 5-6)

Add examples focused on specific use cases:
- [ ] Content Moderation Example
- [ ] Text Summarization Example
- [ ] Style Transfer Example
- [ ] Question Answering Example
- [ ] Code Generation Example

### Phase 4: Advanced and Integration Examples (Weeks 7-8)

Complete the examples with advanced topics and integration examples:
- [ ] Integration Examples
- [ ] Advanced Examples
- [ ] Performance Optimization Example
- [ ] Error Handling Example

## Example Template

All examples should follow the [Example Template](../templates/example_template.py) and adhere to the [Example Standards](./example_standards.md).

## Review Process

Each example should be reviewed for:
1. Technical accuracy
2. Completeness
3. Clarity
4. Consistency with standards
5. Quality of code and comments

## Tracking Progress

Progress will be tracked in this document by marking completed examples with [x].

## Prioritization Criteria

Examples are prioritized based on:
1. User impact (how many users will benefit)
2. Complexity (how difficult the topic is to understand)
3. Dependency (whether other examples depend on it)
4. Current example gaps

## Next Steps

1. Create the Simple Rule Example
2. Create the Multiple Rules Example
3. Create the Basic Chain Example
4. Review and refine the example template based on initial examples
5. Continue with Phase 1 examples
