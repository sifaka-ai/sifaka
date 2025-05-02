# Adapters API Reference

This document provides detailed API documentation for Sifaka's adapter modules.

## Overview

The adapters module provides integration interfaces between Sifaka components and external frameworks. It enables using Sifaka's functionality within other ecosystems and adapts external components to work within Sifaka.

## Module Structure

```
sifaka.adapters
├── __init__.py         # Package initialization
├── langchain.py        # LangChain integration
├── langgraph.py        # LangGraph integration
└── rules/              # Rule adapters
    ├── __init__.py     # Rule adapters initialization
    ├── base.py         # Base adapter functionality
    ├── classifier.py   # Classifier-to-rule adapter
    └── guardrails_adapter.py  # Integration with Guardrails
```

## LangChain Adapter

### `LangChainAdapter`

Adapts Sifaka's models to be used with LangChain components.

```python
from sifaka.adapters.langchain import LangChainAdapter
from sifaka.models.anthropic import AnthropicProvider

model = AnthropicProvider(model="claude-3-haiku")
chain = LangChainAdapter.create_chain(model=model, verbose=True)

result = chain.invoke({"input": "Generate a poem about AI"})
print(result)
```

#### Methods

##### `create_chain(model, **kwargs)`

Creates a LangChain chain using a Sifaka model provider.

**Parameters:**
- `model` (ModelProvider): A Sifaka model provider
- `verbose` (bool, optional): Whether to enable verbose output
- `memory` (bool, optional): Whether to include memory
- `memory_key` (str, optional): Key to use for memory
- `**kwargs`: Additional arguments to pass to the LangChain chain

**Returns:**
- A LangChain chain wrapped in the adapter interface

##### `get_underlying_chain()`

Returns the underlying LangChain chain.

**Returns:**
- The LangChain chain instance

##### `invoke(inputs, **kwargs)`

Invokes the chain with the given inputs.

**Parameters:**
- `inputs` (Dict): The inputs to pass to the chain
- `**kwargs`: Additional arguments to pass to the chain run method

**Returns:**
- The output of the chain

## LangGraph Adapter

### `LangGraphAdapter`

Adapts Sifaka's models and rules to be used with LangGraph components.

```python
from sifaka.adapters.langgraph import LangGraphAdapter
from sifaka.models.anthropic import AnthropicProvider
from sifaka.rules.formatting.length import create_length_rule

model = AnthropicProvider(model="claude-3-haiku")
rule = create_length_rule(min_chars=10, max_chars=100)

graph = LangGraphAdapter.create_graph(
    model=model,
    rules=[rule],
    max_iterations=3
)

result = graph.invoke({"input": "Generate a short story"})
print(result)
```

#### Methods

##### `create_graph(model, rules=None, **kwargs)`

Creates a LangGraph graph using a Sifaka model provider and rules.

**Parameters:**
- `model` (ModelProvider): A Sifaka model provider
- `rules` (List[Rule], optional): Sifaka rules to validate output
- `max_iterations` (int, optional): Maximum iterations for the graph
- `**kwargs`: Additional arguments to pass to the LangGraph builder

**Returns:**
- A LangGraph graph wrapped in the adapter interface

##### `get_underlying_graph()`

Returns the underlying LangGraph graph.

**Returns:**
- The LangGraph graph instance

##### `invoke(inputs, **kwargs)`

Invokes the graph with the given inputs.

**Parameters:**
- `inputs` (Dict): The inputs to pass to the graph
- `**kwargs`: Additional arguments to pass to the graph invoke method

**Returns:**
- The output of the graph

## Rule Adapters

### `ClassifierAdapter`

Adapts Sifaka classifiers to be used as rules.

```python
from sifaka.adapters.rules import ClassifierAdapter
from sifaka.classifiers.toxicity import ToxicityClassifier

classifier = ToxicityClassifier()
rule = ClassifierAdapter(
    classifier=classifier,
    name="toxicity_rule",
    threshold=0.7
)

result = rule.validate("This is a test")
print(f"Passed: {result.passed}")
print(f"Message: {result.message}")
```

#### Constructor

```python
ClassifierAdapter(
    classifier,
    name="classifier_rule",
    description=None,
    threshold=0.5,
    target_label=None,
    **kwargs
)
```

**Parameters:**
- `classifier` (Classifier): The classifier to adapt
- `name` (str, optional): Name of the rule
- `description` (str, optional): Description of the rule
- `threshold` (float, optional): Classification threshold
- `target_label` (str, optional): Target label for classification
- `**kwargs`: Additional rule parameters

#### Methods

##### `validate(text, **kwargs)`

Validates text using the underlying classifier.

**Parameters:**
- `text` (str): Text to validate
- `**kwargs`: Additional parameters to pass to the classifier

**Returns:**
- RuleResult: Result of the validation

### `GuardrailsAdapter`

Adapts Sifaka rules to be used with Guardrails.

```python
from sifaka.adapters.rules.guardrails_adapter import GuardrailsAdapter
from sifaka.models.anthropic import AnthropicProvider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.content.toxicity import create_toxicity_rule

model = AnthropicProvider(model="claude-3-haiku")

# Create rules
rules = [
    create_length_rule(min_chars=20, max_chars=500),
    create_toxicity_rule(threshold=0.7)
]

# Create guardrails
guardrails = GuardrailsAdapter(
    name="content_guardrails",
    rules=rules
)

# Run model with guardrails
result = guardrails.run(
    model=model,
    prompt="Write a short story about robots"
)
print(result)
```

#### Constructor

```python
GuardrailsAdapter(
    rules,
    name="guardrails",
    description=None,
    fail_on_any=True,
    **kwargs
)
```

**Parameters:**
- `rules` (List[Rule]): Rules to use for validation
- `name` (str, optional): Name of the adapter
- `description` (str, optional): Description of the adapter
- `fail_on_any` (bool, optional): Whether to fail if any rule fails
- `**kwargs`: Additional parameters

#### Methods

##### `run(model, prompt, **kwargs)`

Runs the model with guardrails applied.

**Parameters:**
- `model` (ModelProvider): Model to use for generation
- `prompt` (str): Prompt to send to the model
- `max_attempts` (int, optional): Maximum retry attempts
- `**kwargs`: Additional parameters to pass to the model

**Returns:**
- str: The model output that passes the guardrails

##### `validate(text, **kwargs)`

Validates text against all rules.

**Parameters:**
- `text` (str): Text to validate
- `**kwargs`: Additional parameters to pass to the rules

**Returns:**
- RuleResult: Combined result of all rule validations

### Creating Custom Adapters

You can create custom adapters by extending the base classes:

```python
from sifaka.rules.base import Rule, RuleResult
from sifaka.adapters.rules.base import BaseAdapter
from typing import List, Dict, Any

class CustomAdapter(BaseAdapter):
    """Custom adapter implementation."""

    def __init__(
        self,
        name: str = "custom_adapter",
        description: str = "Custom adapter",
        **kwargs
    ):
        super().__init__(name=name, description=description, **kwargs)
        # Initialize custom adapter

    def adapt(self, component: Any) -> Rule:
        """Adapt external component to a Sifaka Rule."""
        # Implementation

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text using the adapter."""
        # Implementation
```

## Error Handling

Adapters provide translation of errors between Sifaka and external frameworks:

```python
from sifaka.adapters.langchain import LangChainAdapter
from sifaka.models.anthropic import AnthropicProvider

try:
    model = AnthropicProvider(model="claude-3-haiku")
    chain = LangChainAdapter.create_chain(model)
    result = chain.invoke({"input": "Generate a poem"})
except Exception as e:
    print(f"Error: {e}")

    # Check for specific error types
    if "ApiError" in str(e):
        print("API error occurred")
    elif "ValidationError" in str(e):
        print("Validation error occurred")
```

## Full Examples

See the [Adapter Examples](../examples/adapter_example.py) for complete working examples of using the adapter APIs.