# Chain Documentation

The Chain class is the central component of the Sifaka framework, orchestrating the process of generating text using language models, validating it against specified criteria, and improving it using specialized critics.

## Overview

The Chain follows a fluent interface pattern (builder pattern) for easy configuration, allowing you to chain method calls to set up the desired behavior. The typical workflow is:

1. Create a Chain instance
2. Configure it with a model, prompt, validators, and critics
3. Run the chain to generate, validate, and improve text
4. Process the result

## Basic Usage

Here's a simple example of using the Chain class:

```python
from sifaka import Chain
from sifaka.validators import length, prohibited_content
from sifaka.critics.reflexion import create_reflexion_critic

# Create a chain with validators and critics
chain = (Chain(max_improvement_iterations=3)
    .with_model("openai:gpt-4")  # Uses OPENAI_API_KEY environment variable
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=100, max_words=500))
    .validate_with(prohibited_content(prohibited=["violent", "harmful"]))
    .improve_with(create_reflexion_critic(model="openai:gpt-4"))
    .with_options(apply_improvers_on_validation_failure=True)  # Enable feedback loop
)

# Run the chain
result = chain.run()

# Check the result
if result.passed:
    print("Chain execution succeeded!")
    print(result.text)
else:
    print("Chain execution failed validation")
    print(result.validation_results[0].message)
```

## Chain Methods

### Initialization

```python
def __init__(
    self,
    config: Optional[SifakaConfig] = None,
    model_factory: Optional[Callable[[str, str], Model]] = None,
    max_improvement_iterations: int = 3,
):
```

Creates a new Chain instance with optional configuration and model factory.

- `config`: Optional configuration for the chain and its components
- `model_factory`: Optional factory function for creating models from strings
- `max_improvement_iterations`: Maximum number of improvement iterations when applying the feedback loop between validators and improvers. Default is 3.

### with_model

```python
def with_model(self, model: Union[str, Model]) -> "Chain":
```

Sets the model to use for text generation. You can provide either:
- A model instance (e.g., `OpenAIModel(model_name="gpt-4", api_key="your-api-key")`)
- A string in the format "provider:model_name" (e.g., "openai:gpt-4")

### with_prompt

```python
def with_prompt(self, prompt: str) -> "Chain":
```

Sets the prompt to use for text generation. This is the input text that guides the model's generation process.

### validate_with

```python
def validate_with(self, validator: Validator) -> "Chain":
```

Adds a validator to the chain. Validators check if the generated text meets certain criteria, such as length, content, or format requirements.

### improve_with

```python
def improve_with(self, improver: Improver) -> "Chain":
```

Adds an improver (critic) to the chain. Improvers enhance the quality of the text by applying various improvement strategies, such as clarity, coherence, or style improvements.

### with_options

```python
def with_options(self, **options: Any) -> "Chain":
```

Sets options to pass to the model during generation. These options can include parameters like temperature, max_tokens, and top_p, which control the behavior of the model.

Special options:
- `apply_improvers_on_validation_failure` (bool): If True, improvers will be applied when validation fails to create a feedback loop. Default is True.

### with_config

```python
def with_config(self, config: SifakaConfig) -> "Chain":
```

Sets the configuration for the chain. This method allows you to update the configuration for the chain and its components.

### run

```python
def run(self) -> Result:
```

Executes the chain and returns the result. This method runs the complete chain execution process with a feedback loop:
1. Checks that the chain is properly configured (has a model and prompt)
2. Generates text using the model and prompt
3. Validates the generated text using all configured validators
4. If validation fails and `apply_improvers_on_validation_failure` is True:
   - Gets feedback from critics on how to improve the text
   - Sends the original text + feedback to the model to generate improved text
   - Re-validates the improved text
   - Repeats steps 4a-4c until validation passes or max iterations is reached
5. If validation passes, improves the text using all configured improvers
6. Returns a Result object containing the final text and all validation/improvement results

## Chain Execution Process

The Chain execution process follows these steps:

1. **Configuration Check**: Ensures that a model and prompt have been specified
2. **Text Generation**: Uses the model to generate text from the prompt
3. **Validation**: Applies each validator in sequence to check if the text meets the specified criteria
   - If any validator fails and `apply_improvers_on_validation_failure` is True:
     - Collects feedback from validators on why validation failed
     - Gets feedback from critics on how to improve the text
     - Uses the model to generate improved text based on the combined feedback
     - Re-validates the improved text
     - Repeats this process until validation passes or max iterations is reached
   - If any validator fails and `apply_improvers_on_validation_failure` is False:
     - The process stops and returns a failed result
4. **Improvement**: If all validations pass, applies each improver in sequence to enhance the text quality
5. **Result Creation**: Returns a Result object containing the final text and all validation/improvement results

## Feedback Loop

The Chain class implements a feedback loop between validators and critics to improve text that fails validation. This feature is enabled by default and can be controlled with the `apply_improvers_on_validation_failure` option.

When the feedback loop is enabled:

1. If validation fails, the Chain collects feedback from validators on why validation failed
2. The Chain gets feedback from critics on how to improve the text
3. The combined feedback is sent to the model along with the original text to generate improved text
4. The improved text is re-validated
5. This process repeats until validation passes or the maximum number of iterations is reached

The maximum number of iterations can be set when creating the Chain instance:

```python
# Create a chain with a custom number of improvement iterations
chain = Chain(max_improvement_iterations=5)
```

To enable or disable the feedback loop:

```python
# Enable the feedback loop (default)
chain = chain.with_options(apply_improvers_on_validation_failure=True)

# Disable the feedback loop
chain = chain.with_options(apply_improvers_on_validation_failure=False)
```

## Using Configuration

You can provide a configuration object when creating a Chain instance to customize the behavior of the chain and its components:

```python
from sifaka import Chain
from sifaka.config import SifakaConfig, ModelConfig

# Create a custom configuration
config = SifakaConfig(
    model=ModelConfig(temperature=0.8, max_tokens=500),
    debug=True
)

# Create a chain with the configuration
chain = (Chain(config)
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .run())
```

## Working with Results

The `run()` method returns a `Result` object that contains:
- The final text after all validations and improvements
- Whether all validations passed
- The results of all validations
- The results of all improvements

You can use this object to check if the chain execution succeeded and to access the generated text:

```python
result = chain.run()

if result.passed:
    print("Chain execution succeeded!")
    print(result.text)
else:
    print("Chain execution failed validation")
    for validation_result in result.validation_results:
        if not validation_result.passed:
            print(f"Failed validation: {validation_result.message}")
            if validation_result.issues:
                for issue in validation_result.issues:
                    print(f"- Issue: {issue}")
            if validation_result.suggestions:
                for suggestion in validation_result.suggestions:
                    print(f"- Suggestion: {suggestion}")
```

## Advanced Usage

### Custom Model Factory

You can provide a custom model factory function when creating a Chain instance:

```python
def my_model_factory(provider: str, model_name: str) -> Model:
    # Custom logic to create models
    if provider == "my_provider":
        return MyCustomModel(model_name)
    else:
        # Fall back to default factory
        from sifaka.factories import create_model
        return create_model(provider, model_name)

chain = Chain(model_factory=my_model_factory)
```

### Chaining Multiple Improvers

You can add multiple improvers to the chain to apply different improvement strategies in sequence:

```python
from sifaka import Chain
from sifaka.critics.reflexion import create_reflexion_critic
from sifaka.critics.n_critics import create_n_critics_critic

chain = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .improve_with(create_reflexion_critic(model="openai:gpt-4"))
    .improve_with(create_n_critics_critic(model="openai:gpt-4", num_critics=3))
    .run())
```

## Best Practices

1. **Always specify a model and prompt** before running the chain
2. **Use validators** to ensure the generated text meets your requirements
3. **Use improvers** to enhance the quality of the generated text
4. **Check the result** to ensure the chain execution succeeded
5. **Use configuration** to customize the behavior of the chain and its components
6. **Handle errors** that may occur during chain execution
