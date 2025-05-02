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

# Install with LangGraph integration
pip install "sifaka[langgraph]"

# Install with LangChain integration
pip install "sifaka[langchain]"

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

### Integrations
- `langgraph`: LangGraph integration for complex workflows
- `langchain`: LangChain integration for chain-based processing

### Benchmarking
- `benchmark`: Tools for performance benchmarking and analysis

## Key Features

- ✅ **Validation Rules**: Enforce constraints like length limits and content restrictions
- ✅ **Response Critics**: Provide feedback to improve model outputs
- ✅ **Chain Architecture**: Create feedback loops for iterative improvement
- ✅ **Model Agnostic**: Works with Claude, OpenAI, and other LLM providers
- ✅ **Streamlined Configuration**: Unified configuration system using ClassifierConfig and RuleConfig

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
    model=claude_model,
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

For more control and customization, use the factory functions or create chains with specialized components:

```python
from sifaka.chain import create_simple_chain, create_backoff_chain

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

### Example 1: Claude with Length Critic (Condense Text)

This example uses Claude with a critic to reduce verbose responses:

```python
# Load environment variables (.env file with ANTHROPIC_API_KEY)
from dotenv import load_dotenv
load_dotenv()

import os
from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import Chain

# Configure Claude model
model = AnthropicProvider(
    model_name="claude-3-sonnet-20240229",
    config=ModelConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=1500,
    )
)

# Create length rule (max 400 words)
length_rule = create_length_rule(
    min_words=100,
    max_words=500,
    rule_id="word_limit_rule"
)

# Create critic to help condense text
critic = PromptCritic(
    llm_provider=model,
    config=PromptCriticConfig(
        name="length_critic",
        description="A critic that helps condense text while preserving meaning",
        system_prompt=(
            "You are an editor who specializes in making text more concise "
            "while preserving core content and meaning."
        )
    )
)

# Create chain with model, rule, and critic
chain = Chain(model=model, rules=[length_rule], critic=critic, max_attempts=3)

# Run the chain with a prompt that would naturally generate verbose output
result = chain.run("Explain how large language models work in detail")
```

### Example 2: Content Safety Validation

This example uses a toxicity classifier with a rule to ensure content safety:

```python
from sifaka.classifiers.toxicity import ToxicityClassifier
from sifaka.classifiers.base import ClassifierConfig
from sifaka.rules.content.safety import create_toxicity_rule
from sifaka.rules.base import RuleConfig
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.chain import Chain
import os

# Configure OpenAI model for content generation
model = OpenAIProvider(
    model_name="gpt-4",
    config=ModelConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.7
    )
)

# Configure toxicity classifier
classifier = ToxicityClassifier(
    config=ClassifierConfig(
        labels=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_toxic"],
        params={
            "general_threshold": 0.5,
            "severe_toxic_threshold": 0.7,
        }
    )
)

# Create toxicity rule
toxicity_rule = create_toxicity_rule(
    config=RuleConfig(
        params={
            "threshold": 0.6,
            "indicators": [
                "hate",
                "offensive",
                "vulgar",
                "profanity",
            ]
        }
    )
)

# Create critic to help improve content that violates toxicity rules
critic = PromptCritic(
    llm_provider=model,
    config=PromptCriticConfig(
        name="content_safety_critic",
        description="A critic that helps ensure content is appropriate and non-toxic",
        system_prompt=(
            "You are an editor who specializes in ensuring content is appropriate, "
            "respectful, and free from offensive or harmful language."
        )
    )
)

# Create chain with model, rule, and critic
chain = Chain(model=model, rules=[toxicity_rule], critic=critic, max_attempts=3)

# Run the chain
result = chain.run("Write a social media post about community values")
```

## Full Examples

For complete, runnable examples, see the `/examples` directory:

- `claude_length_critic.py`: Demonstrates reducing text length
- `claude_expand_length_critic.py`: Demonstrates expanding text length
- `toxicity_filtering.py`: Demonstrates content safety validation
- `reflexion_critic_example.py`: Demonstrates using the ReflexionCritic to improve text through learning from past feedback
- `advanced_chain_example.py`: Demonstrates the new chain architecture with different components and strategies

### Example 3: Reflexion-Based Learning

This example uses the ReflexionCritic to improve text by learning from past feedback:

```python
from sifaka.critics.reflexion import ReflexionCriticConfig, create_reflexion_critic
from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.formatting.length import create_length_rule
from sifaka.chain import Chain
import os

# Configure OpenAI model
model = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    config=ModelConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=1500,
    )
)

# Create a reflexion critic using the factory function
reflexion_critic = create_reflexion_critic(
    model=model,
    name="reflexion_critic",
    description="A critic that learns from past feedback",
    system_prompt=(
        "You are an expert editor who learns from past feedback to improve future edits. "
        "Focus on identifying patterns in feedback and applying those lessons to new tasks."
    ),
    memory_buffer_size=5,  # Store up to 5 reflections
    reflection_depth=1     # Perform 1 level of reflection
)

# Create a length rule
length_rule = create_length_rule(
    min_words=50,
    max_words=225,
    rule_id="word_length_rule",
    description="Ensures text is between 50-300 words",
)

# Create a chain with the model, rule, and reflexion critic
chain = Chain(
    model=model,
    rules=[length_rule],
    critic=reflexion_critic,
    max_attempts=3,
)

# Process a series of prompts and observe how the critic improves over time
prompts = [
    "Explain the concept of machine learning in detail.",
    "Describe the process of photosynthesis in plants.",
    "Explain how the internet works.",
]

for prompt in prompts:
    try:
        result = chain.run(prompt)
        # Process successful result
        print(f"Output: {result.output}")
    except ValueError as e:
        # Handle validation failure
        print(f"Validation failed: {e}")


# The reflexion critic's memory buffer now contains reflections that will
# guide future improvements, making it more effective over time
```

## Integration with Guardrails

While Sifaka is designed to be "batteries included" with its built-in classifiers and rules, it also provides seamless integration with [Guardrails AI](https://www.guardrailsai.com/). This integration allows you to:

- Use Guardrails' validation and transformation capabilities alongside Sifaka's native features
- Leverage Guardrails' extensive rule library while maintaining Sifaka's flexibility
- Combine both systems' strengths for robust content validation and transformation

Example integration:
```python
from sifaka.rules.adapters.guardrails_adapter import GuardrailsAdapter
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
