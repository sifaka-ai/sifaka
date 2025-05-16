# Chain

The `Chain` class is the main orchestrator in Sifaka. It coordinates the generation, validation, and improvement of text using LLMs.

## Overview

The Chain class uses a builder pattern to create a fluent API for configuring and executing LLM operations. It allows you to:

- Specify which model to use
- Set the prompt for generation
- Add validators to check if the generated text meets requirements
- Add improvers to enhance the quality of the generated text
- Configure model options
- Execute the chain and get results

## Basic Usage

```python
from sifaka import Chain

# Create a simple chain
result = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .run())

print(result.text)
```

## API Reference

### Constructor

```python
Chain()
```

Creates a new Chain instance with no configuration.

### Methods

#### `with_model`

```python
with_model(model: Union[str, Model]) -> Chain
```

Sets the model to use for generation.

**Parameters:**
- `model`: Either a model instance or a string in the format "provider:model_name".

**Returns:**
- The chain instance for method chaining.

**Example:**
```python
# Using a string
chain = Chain().with_model("openai:gpt-4")

# Using a model instance
from sifaka.models import OpenAIModel
model = OpenAIModel("gpt-4", api_key="your-api-key")
chain = Chain().with_model(model)
```

#### `with_prompt`

```python
with_prompt(prompt: str) -> Chain
```

Sets the prompt to use for generation.

**Parameters:**
- `prompt`: The prompt to use for generation.

**Returns:**
- The chain instance for method chaining.

**Example:**
```python
chain = Chain().with_prompt("Write a short story about a robot.")
```

#### `validate_with`

```python
validate_with(validator: Validator) -> Chain
```

Adds a validator to the chain.

**Parameters:**
- `validator`: The validator to add.

**Returns:**
- The chain instance for method chaining.

**Example:**
```python
from sifaka.validators import length

chain = Chain().validate_with(length(min_words=50, max_words=200))
```

#### `improve_with`

```python
improve_with(improver: Improver) -> Chain
```

Adds an improver to the chain.

**Parameters:**
- `improver`: The improver to add.

**Returns:**
- The chain instance for method chaining.

**Example:**
```python
from sifaka.validators import clarity

chain = Chain().improve_with(clarity())
```

#### `with_options`

```python
with_options(**options: Any) -> Chain
```

Sets options for the model.

**Parameters:**
- `**options`: Options to pass to the model.

**Returns:**
- The chain instance for method chaining.

**Example:**
```python
chain = Chain().with_options(temperature=0.7, max_tokens=100)
```

#### `run`

```python
run() -> Result
```

Executes the chain and returns the result.

**Returns:**
- The result of the chain execution.

**Raises:**
- `ChainError`: If the chain is not properly configured.

**Example:**
```python
result = chain.run()
print(result.text)
print(f"Passed validation: {result.passed}")
```

## Complete Example

```python
from sifaka import Chain
from sifaka.validators import length, clarity, factual_accuracy

# Create and configure the chain
chain = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short explanation of quantum computing.")
    .validate_with(length(min_words=50, max_words=200))
    .validate_with(factual_accuracy())
    .improve_with(clarity())
    .with_options(temperature=0.7, max_tokens=500))

# Run the chain
result = chain.run()

# Process the result
print(f"Result passed validation: {result.passed}")
print(result.text)

# Print validation results
for i, validation_result in enumerate(result.validation_results):
    print(f"Validation {i+1}: {validation_result.message}")

# Print improvement results
for i, improvement_result in enumerate(result.improvement_results):
    print(f"Improvement {i+1}: {improvement_result.message}")
```

## Notes

- The chain executes validators in the order they are added. If any validator fails, the chain stops and returns a failed result.
- Improvers are executed in the order they are added, with each improver receiving the text from the previous improver.
- The chain requires both a model and a prompt to be set before running.
- Model options are passed directly to the model's generate method.
