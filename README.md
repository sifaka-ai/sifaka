# Sifaka

Sifaka is a framework for improving large language model (LLM) outputs through validation, reflection, and refinement. It helps build more reliable AI systems by enforcing constraints and improving response quality.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features

- ✅ **Validation Rules**: Enforce constraints like length limits and content restrictions
- ✅ **Response Critics**: Provide feedback to improve model outputs
- ✅ **Chain Architecture**: Create feedback loops for iterative improvement
- ✅ **Model Agnostic**: Works with Claude, OpenAI, and other LLM providers

## Installation

```bash
# Basic installation
pip install sifaka

# Development installation
git clone https://github.com/your-username/sifaka.git
cd sifaka
pip install -e .
```

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

```python
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models.anthropic import AnthropicProvider

# Create critic that uses Claude to provide feedback
critic = PromptCritic(
    llm_provider=claude_model,
    config=PromptCriticConfig(
        name="length_critic",
        system_prompt="You are an editor who specializes in adjusting text length."
    )
)
```

### 3. Chains

Chains orchestrate the validation and improvement process:

```python
from sifaka.chain import Chain

# Create a chain with a model, rules, and critic
chain = Chain(
    model=model,            # LLM Provider
    rules=[length_rule],    # Validation rules
    critic=critic,          # Improvement critic
    max_attempts=3          # Max retries before giving up
)

# Run the chain with a prompt
result = chain.run("Write about artificial intelligence")
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
    max_words=400,
    rule_id="word_limit_rule"
)

# Create critic to help condense text
critic = PromptCritic(
    llm_provider=model,
    config=PromptCriticConfig(
        name="length_critic",
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

### Example 2: Claude with Length Critic (Expand Text)

This example uses Claude with a critic to expand brief responses:

```python
# Same imports and initial setup as above

# Create length rule (minimum 500 words)
length_rule = create_length_rule(
    min_words=500,
    max_words=1000,
    rule_id="expanded_length_rule"
)

# Create critic to help expand text
critic = PromptCritic(
    llm_provider=model,
    config=PromptCriticConfig(
        name="expansion_critic",
        system_prompt=(
            "You are an editor who specializes in expanding text by adding relevant "
            "examples, elaborating on key points, and providing deeper context."
        )
    )
)

# Create chain with model, rule, and critic
chain = Chain(model=model, rules=[length_rule], critic=critic, max_attempts=4)

# Run the chain with a prompt that would naturally generate brief output
result = chain.run("Briefly explain what artificial intelligence is")
```

## Full Examples

For complete, runnable examples, see the `/examples` directory:

- `claude_length_critic.py`: Demonstrates reducing text length
- `claude_expand_length_critic.py`: Demonstrates expanding text length

## License

MIT
