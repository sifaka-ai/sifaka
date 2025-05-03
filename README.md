# Sifaka

Sifaka is a framework for improving large language model (LLM) outputs through validation, reflection, and refinement. It helps build more reliable AI systems by enforcing constraints and improving response quality.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

Sifaka can be installed with different sets of dependencies depending on your needs:

### Basic Installation
```bash
pip install sifaka
```

### Installation with Specific Features

```bash
# Install with OpenAI support
pip install "sifaka[openai]"

# Install with Anthropic support
pip install "sifaka[anthropic]"

# Install with all classifiers
pip install "sifaka[classifiers]"

# Install with benchmarking tools
pip install "sifaka[benchmark]"

# Install everything (except development tools)
pip install "sifaka[all]"
```

### Development Installation
```bash
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka
pip install -e ".[dev]"  # Install with development dependencies
```

## Optional Dependencies

Sifaka's functionality can be extended through optional dependencies:

### Model Providers
- `openai`: OpenAI API support
- `anthropic`: Anthropic Claude API support
- `google-generativeai`: Google Gemini API support

### Classifiers
- `toxicity`: Toxicity detection using Detoxify
- `sentiment`: Sentiment analysis using VADER
- `profanity`: Profanity detection
- `language`: Language detection
- `readability`: Text readability analysis

### Benchmarking
- `benchmark`: Tools for performance benchmarking and analysis

## Key Features

- ✅ **Validation Rules**: Enforce constraints like length limits and content restrictions
- ✅ **Response Critics**: Provide feedback to improve model outputs
- ✅ **Chain Architecture**: Create feedback loops for iterative improvement
- ✅ **Model Agnostic**: Works with Claude, OpenAI, and other LLM providers
- ✅ **Streamlined Configuration**: Unified configuration system using ClassifierConfig and RuleConfig

## API Key Handling

Sifaka provides a safe and flexible way to manage API keys for various model providers.

### Setting Up API Keys

You can provide API keys in several ways:

1. **Environment Variables** (recommended):
   ```python
   # In your .env file
   ANTHROPIC_API_KEY=your_anthropic_key_here
   OPENAI_API_KEY=your_openai_key_here
   GUARDRAILS_API_KEY=your_guardrails_key_here
   ```

   ```python
   # In your Python code
   from dotenv import load_dotenv
   load_dotenv()  # Load keys from .env file

   # Create provider without explicitly passing the key
   from sifaka.models.anthropic import AnthropicProvider
   model = AnthropicProvider(model_name="claude-3-haiku-20240307")
   # The API key will be loaded from the environment
   ```

2. **Explicit Configuration**:
   ```python
   from sifaka.models.base import ModelConfig
   from sifaka.models.anthropic import AnthropicProvider

   model = AnthropicProvider(
       model_name="claude-3-haiku-20240307",
       config=ModelConfig(
           api_key="your_api_key_here",
           temperature=0.7,
           max_tokens=1000,
       )
   )
   ```

### Handling Missing API Keys

Sifaka has built-in safety checks to prevent API calls when keys are missing:

```python
import os
from dotenv import load_dotenv
load_dotenv()

from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig

# Check if API key is available
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("Warning: No API key found in environment variables.")
    print("Using a placeholder for testing only.")
    api_key = "test_key_for_examples"  # This won't work for real API calls

# Initialize model with safety checks
try:
    model = AnthropicProvider(
        model_name="claude-3-haiku-20240307",
        config=ModelConfig(
            api_key=api_key,
            temperature=0.7,
            max_tokens=1000,
        )
    )

    # Will only run if a valid API key is available
    if api_key != "test_key_for_examples":
        response = model.generate("Hello, world!")
        print(f"Model response: {response}")
    else:
        print("Skipping API call due to missing API key")

except ValueError as e:
    print(f"Error: {e}")
    print("Please set your API key in the .env file or provide it directly")

## Core Components

### 1. Rules

Rules validate responses against specific criteria:

```python
# Length rule checks if text is within specified length bounds
from sifaka.rules.formatting.length import create_length_rule

length_rule = create_length_rule(
    min_words=100,   # Minimum word count
    max_words=500,   # Maximum word count
)
```

### 2. Critics

Critics analyze and provide feedback on model outputs:

#### PromptCritic

The standard critic that provides feedback based on a single prompt:

```python
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models.anthropic import AnthropicProvider

# Create critic that uses Claude to provide feedback
critic = PromptCritic(
    llm_provider=claude_model,
    config=PromptCriticConfig(
        name="length_critic",
        description="A critic that helps adjust text length",
        system_prompt="You are an editor who specializes in adjusting text length."
    )
)
```

#### ReflexionCritic

A critic that maintains a memory of past improvements and uses these reflections to guide future improvements:

```python
from sifaka.critics.reflexion import ReflexionCriticConfig, create_reflexion_critic
from sifaka.models.anthropic import AnthropicProvider

# Create a reflexion critic that uses Claude
reflexion_critic = create_reflexion_critic(
    llm_provider=claude_model,
    name="reflexion_critic",
    description="A critic that learns from past feedback",
    system_prompt="You are an editor who learns from past feedback to improve future edits.",
    memory_buffer_size=5,  # Store up to 5 reflections
    reflection_depth=1     # Perform 1 level of reflection
)
```

### 3. Chains

Chains orchestrate the validation and improvement process. Sifaka provides two ways to create chains:

#### Simple Chain Creation

```python
from sifaka.chain import create_simple_chain

# Create a chain with a model, rules, and critic
chain = create_simple_chain(
    model=model,            # LLM Provider
    rules=[length_rule],    # Validation rules
    critic=critic,          # Improvement critic
    max_attempts=3          # Max retries before giving up
)

# Run the chain with a prompt
result = chain.run("Write about artificial intelligence")
```

#### Advanced Chain Creation

For more control and customization, use the factory functions:

```python
from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.rules.formatting.length import create_length_rule
from sifaka.chain import create_simple_chain, create_backoff_chain

# First, create a proper model provider
model = AnthropicProvider(
    model_name="claude-3-haiku-20240307",
    config=ModelConfig(
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000,
    )
)

# Create a length rule
length_rule = create_length_rule(
    min_words=100,
    max_words=200
)

# Create a critic - make sure to pass the model object to llm_provider
critic = PromptCritic(
    llm_provider=model,  # Pass the model OBJECT here, not a string
    config=PromptCriticConfig(
        name="length_critic",
        description="A critic that helps adjust text length",
        system_prompt="You are an editor who specializes in adjusting text length."
    )
)

# Create a simple chain with fixed retry attempts
chain = create_simple_chain(
    model=model,
    rules=[length_rule],
    critic=critic,
    max_attempts=3,
)

# Create a chain with exponential backoff retry strategy
chain = create_backoff_chain(
    model=model,
    rules=[length_rule],
    critic=critic,
    max_attempts=5,
    initial_backoff=1.0,
    backoff_factor=2.0,
    max_backoff=60.0,
)
```

### 4. Classifiers

Classifiers analyze text and categorize it according to specific criteria:

```python
from sifaka.classifiers.toxicity import ToxicityClassifier
from sifaka.classifiers.base import ClassifierConfig

# Create a toxicity classifier with custom thresholds
classifier = ToxicityClassifier(
    config=ClassifierConfig(
        labels=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_toxic"],
        params={
            "general_threshold": 0.6,
            "severe_toxic_threshold": 0.8,
            "threat_threshold": 0.7,
        }
    )
)

# Classify text
result = classifier.classify("Your text to analyze")
print(f"Label: {result.label}, Confidence: {result.confidence}")
```

## Configuration System

Sifaka uses a streamlined configuration system with two main configuration classes:

### ClassifierConfig

Used by all classifiers to manage their configuration parameters:

```python
from sifaka.classifiers.base import ClassifierConfig

config = ClassifierConfig(
    labels=["positive", "neutral", "negative"],  # Available classification labels
    cost=1.0,                                   # Relative computational cost
    min_confidence=0.7,                         # Minimum confidence threshold
    params={                                    # All classifier-specific parameters
        "model_name": "default",
        "threshold": 0.5,
    }
)
```

### RuleConfig

Used by all rules to manage their configuration parameters:

```python
from sifaka.rules.base import RuleConfig, RulePriority

config = RuleConfig(
    priority=RulePriority.HIGH,      # Rule execution priority
    cache_size=100,                  # Cache size for validation results
    cost=1.0,                        # Relative computational cost
    params={                         # All rule-specific parameters
        "threshold": 0.7,
        "max_length": 500,
    }
)
```

## Usage Examples

### Example 1: Basic Length Rule

This example demonstrates how to use a simple length rule:

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a length rule to ensure text is within 100-500 words
length_rule = create_length_rule(
    min_words=100,   # Minimum word count
    max_words=500,   # Maximum word count
    rule_id="word_limit_rule",
    description="Ensures text is between 100-500 words"
)

# Validate some text
text = "This is a short test."
result = length_rule.validate(text)

# Check if validation passed
if result.passed:
    print("Text meets length requirements")
else:
    print(f"Text failed length check: {result.message}")
```

### Example 2: Using a PromptCritic with Claude

This example shows how to use a PromptCritic with Claude to improve text quality:

```python
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig

# Get API key from environment
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("Warning: ANTHROPIC_API_KEY not found in environment variables")
    print("Using a placeholder key for demonstration purposes only")
    print("Set your ANTHROPIC_API_KEY to run this example with real API calls")
    api_key = "placeholder_key"  # This won't work for actual API calls

# Configure Claude model
model = AnthropicProvider(
    model_name="claude-3-haiku-20240307",  # Fast, efficient model
    config=ModelConfig(
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000,
    )
)

# Create a critic to help improve text
critic = PromptCritic(
    llm_provider=model,
    config=PromptCriticConfig(
        name="writing_critic",
        description="A critic that improves writing clarity",
        system_prompt=(
            "You are an expert editor who helps improve writing clarity and structure. "
            "Focus on making text more concise, clear, and engaging."
        )
    )
)

# Example text to improve
text = "The thing is that writing can be improved in various ways and stuff."

# Only run if a valid API key is available (not the placeholder)
if api_key != "placeholder_key":
    try:
        # Improve the text
        improved_text = critic.improve(text)
        print(f"Original: {text}")
        print(f"Improved: {improved_text}")
    except Exception as e:
        print(f"Error: {e}")
        print("This may be due to an invalid API key or network issue")
else:
    print("\nDemo mode: Would improve this text if API key was provided:")
    print(f"Original: {text}")
    print("Possible improvement: \"Writing can be improved through conciseness, clarity, and proper structure.\"")
```

### Example 3: Simple Chain with a Length Rule

This example demonstrates a simple chain that uses a length rule and critic:

```python
import os
from dotenv import load_dotenv
load_dotenv()

from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import create_simple_chain

# Get API key (use a placeholder if not available)
api_key = os.environ.get("ANTHROPIC_API_KEY", "your_api_key_here")

# Configure Claude model
model = AnthropicProvider(
    model_name="claude-3-haiku-20240307",
    config=ModelConfig(
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000,
    )
)

# Create length rule (100-200 words)
length_rule = create_length_rule(
    min_words=100,
    max_words=200,
    rule_id="word_limit_rule",
    description="Ensures text is between 100-200 words"
)

# Create critic to help with length adjustments
critic = PromptCritic(
    llm_provider=model,
    config=PromptCriticConfig(
        name="length_critic",
        description="A critic that adjusts text length",
        system_prompt=(
            "You are an editor who specializes in adjusting text length. "
            "Make text more concise while preserving key information. "
            "Ensure the text is between 100-200 words."
        )
    )
)

# Create a simple chain with the model, rule, and critic
chain = create_simple_chain(
    model=model,
    rules=[length_rule],
    critic=critic,
    max_attempts=3
)

# Example prompt (would generate text likely over 200 words)
prompt = "Explain the concept of machine learning in detail, including supervised, unsupervised, and reinforcement learning"

# Only run if API key is available
if api_key != "your_api_key_here":
    # Run the chain
    try:
        result = chain.run(prompt)
        print(f"Word count: {len(result.output.split())}")
        print(f"Output: {result.output[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Set your ANTHROPIC_API_KEY to run this example")
```

### Example 4: Using a Toxicity Classifier

This example shows how to use a toxicity classifier to detect potentially harmful content:

```python
from sifaka.classifiers.toxicity import ToxicityClassifier
from sifaka.classifiers.base import ClassifierConfig

# Create a toxicity classifier
classifier = ToxicityClassifier(
    config=ClassifierConfig(
        labels=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_toxic"],
        params={
            "general_threshold": 0.5,
            "severe_toxic_threshold": 0.7,
            "threat_threshold": 0.7,
        }
    )
)

# Example texts to classify
texts = [
    "I really enjoyed reading this article, it was informative and well-written.",
    "This is absolutely terrible and I hate everything about it."
]

# Classify each text
for text in texts:
    try:
        result = classifier.classify(text)
        print(f"Text: {text}")
        print(f"Classification: {result.label}, Confidence: {result.confidence:.2f}")
        print()
    except Exception as e:
        print(f"Classification error: {e}")
```

### Example 5: Using ReflexionCritic with Concrete Implementation

This example demonstrates how to use ReflexionCritic by creating a concrete implementation:

```python
import os
from dotenv import load_dotenv
load_dotenv()

from sifaka.critics.reflexion import ReflexionCritic, ReflexionCriticConfig
from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig

# Get API key (use a placeholder if not available)
api_key = os.environ.get("ANTHROPIC_API_KEY", "your_api_key_here")

# Configure Claude model
model = AnthropicProvider(
    model_name="claude-3-haiku-20240307",
    config=ModelConfig(
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000,
    )
)

# Create a concrete implementation of ReflexionCritic
class ConcreteReflexionCritic(ReflexionCritic):
    """Concrete implementation of the ReflexionCritic abstract class."""

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Implement the required abstract method.

        Args:
            text: The text to improve
            feedback: Feedback to guide improvement

        Returns:
            str: The improved text
        """
        # Simply delegate to the improve method
        return self.improve(text, feedback)

# Create a reflexion critic with our concrete implementation
config = ReflexionCriticConfig(
    name="learning_critic",
    description="A critic that learns from past feedback",
    system_prompt=(
        "You are an expert editor who learns from past feedback to improve future edits. "
        "Focus on identifying patterns in feedback and applying those lessons."
    ),
    memory_buffer_size=3,  # Store up to 3 reflections
    reflection_depth=1     # How many levels of reflection to perform
)

reflexion_critic = ConcreteReflexionCritic(
    llm_provider=model,
    config=config
)

# Example text and feedback for multiple iterations
texts = [
    "The thing is that this sentence is not very good because it uses filler phrases.",
    "This explanation could be better but I'm not sure how to fix it properly.",
    "The report was completed yesterday and I think it turned out okay."
]

feedbacks = [
    "Remove filler phrases like 'the thing is' and make more direct",
    "Add specific details and examples to make the explanation clearer",
    "Be more specific about the report's content and use active voice"
]

# Only run if API key is available
if api_key != "your_api_key_here":
    # Demonstrate multiple improvements with reflection
    for i, (text, feedback) in enumerate(zip(texts, feedbacks)):
        print(f"\nIteration {i+1}:")
        print(f"Original: {text}")

        # Improve with feedback
        improved = reflexion_critic.improve(text, feedback)
        print(f"Feedback: {feedback}")
        print(f"Improved: {improved}")

        # The critic learns from each interaction
        print("(The critic is learning from this interaction to improve future edits)")
else:
    print("Set your ANTHROPIC_API_KEY to run this example")
```

### Example 6: Using Guardrails Integration

This example shows how to use a basic Guardrails validator with Sifaka:

```python
try:
    # Import Guardrails validator and registration
    from guardrails.validators import Validator
    from guardrails.validators import register_validator

    # Import Sifaka adapter for Guardrails
    from sifaka.adapters.rules.guardrails_adapter import create_guardrails_rule

    # Create and register a simple custom validator
    @register_validator(name="contains_word", data_type="string")
    class SimpleValidator(Validator):
        """Checks if text contains a specific word."""

        def __init__(self, word_to_find, on_fail="noop"):
            super().__init__(on_fail=on_fail)
            self.word_to_find = word_to_find

        def validate(self, value, metadata=None):
            if self.word_to_find in value:
                return self.pass_validation(value)
            else:
                return self.fail_validation(value, f"Text must contain '{self.word_to_find}'")

    # Create a Guardrails validator
    validator = SimpleValidator(word_to_find="important")

    # Create a Sifaka rule using the Guardrails validator
    rule = create_guardrails_rule(
        guardrails_validator=validator,
        rule_id="important_word_rule",
        name="Important Word Check",
        description="Checks if text contains the word 'important'"
    )

    # Validate some text
    valid_text = "This is an important point to consider."
    invalid_text = "This is something to consider."

    # Check valid text
    result = rule.validate(valid_text)
    print(f"Valid text: {result.passed}, Message: {result.message}")

    # Check invalid text
    result = rule.validate(invalid_text)
    print(f"Invalid text: {result.passed}, Message: {result.message}")

except ImportError:
    print("Guardrails is not installed. Install with: pip install guardrails-ai")
except Exception as e:
    print(f"Error using Guardrails: {str(e)}")
    print("This may be due to version compatibility issues.")
    print("For more complex examples, see the Guardrails documentation:")
    print("https://www.guardrailsai.com/docs/")
```

## Advanced Chain Example

This more advanced example shows how to create a chain with multiple rules and critics:

```python
import os
from dotenv import load_dotenv
load_dotenv()

from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.formatting.style import create_style_rule, CapitalizationStyle
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import create_simple_chain

# Get API key
api_key = os.environ.get("ANTHROPIC_API_KEY", "your_api_key_here")

# Configure model
model = AnthropicProvider(
    model_name="claude-3-haiku-20240307",
    config=ModelConfig(
        api_key=api_key,
        temperature=0.7,
        max_tokens=1200,
    )
)

# Create multiple rules
length_rule = create_length_rule(
    min_words=100,
    max_words=300,
    rule_id="length_rule"
)

style_rule = create_style_rule(
    capitalization=CapitalizationStyle.SENTENCE_CASE,
    rule_id="style_rule"
)

# Create critic
critic = PromptCritic(
    llm_provider=model,
    config=PromptCriticConfig(
        name="quality_critic",
        description="A critic that improves writing quality",
        system_prompt=(
            "You are an expert editor who improves writing quality. "
            "Ensure text uses sentence case capitalization and is between 100-300 words. "
            "Make writing clear, direct, and engaging."
        )
    )
)

# Create chain with both rules
chain = create_simple_chain(
    model=model,
    rules=[length_rule, style_rule],
    critic=critic,
    max_attempts=3
)

# Example prompt
prompt = "Write a short explanation of how neural networks function."

# Only run if API key is available
if api_key != "your_api_key_here":
    try:
        result = chain.run(prompt)
        print(f"Word count: {len(result.output.split())}")
        print(f"Output: {result.output[:150]}...")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Set your ANTHROPIC_API_KEY to run this example")
```

## Full Examples

For complete, runnable examples, see the `/examples` directory:

- `claude_length_critic.py`: Demonstrates reducing text length
- `claude_expand_length_critic.py`: Demonstrates expanding text length
- `toxicity_filtering.py`: Demonstrates content safety validation
- `reflexion_critic_example.py`: Demonstrates using the ReflexionCritic to improve text through learning from past feedback
- `advanced_chain_example.py`: Demonstrates the new chain architecture with different components and strategies

## Integration with Guardrails

While Sifaka is designed to be "batteries included" with its built-in classifiers and rules, it also provides seamless integration with [Guardrails AI](https://www.guardrailsai.com/). This integration allows you to:

- Use Guardrails' validation and transformation capabilities alongside Sifaka's native features
- Leverage Guardrails' extensive rule library while maintaining Sifaka's flexibility
- Combine both systems' strengths for robust content validation and transformation

Example integration:
```python
from sifaka.adapters.rules.guardrails_adapter import GuardrailsAdapter
from sifaka.domain import Domain

# Create a Guardrails adapter
guardrails_adapter = GuardrailsAdapter()

# Use it in your domain configuration
domain = Domain({
    "name": "text",
    "rules": {
        "guardrails": {
            "enabled": True,
            "adapter": guardrails_adapter
        }
    }
})
```

## License

Sifaka is licensed under the MIT License. See [LICENSE](LICENSE) for details.


