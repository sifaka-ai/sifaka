# Sifaka Examples

This directory contains example scripts demonstrating various features and integrations of the Sifaka library.

## Core Components

1. **Chain**
The main component that orchestrates the validation and improvement process:

```python
from sifaka.chain import Chain
from sifaka.rules import LengthRule, SymmetryRule, RepetitionRule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models import OpenAIProvider
from sifaka.rules.base import RuleConfig, RulePriority

# Initialize provider
provider = OpenAIProvider(
    model_name="gpt-4-turbo-preview",
    config={"api_key": "your-api-key"}
)

# Set up rules
rules = [
    LengthRule(
        name="length_check",
        description="Checks if output length is within bounds",
        config={"min_length": 30, "max_length": 100}
    ),
    SymmetryRule(
        name="symmetry_check",
        description="Checks for text symmetry patterns",
        config=RuleConfig(
            priority=RulePriority.MEDIUM,
            metadata={"symmetry_threshold": 0.8}
        )
    )
]

# Set up critic
critic = PromptCritic(
    model=provider,
    config=PromptCriticConfig(
        name="text_improver",
        description="Improves text based on validation results"
    )
)

# Create chain
chain = Chain(
    model=provider,
    rules=rules,
    critic=critic,
    max_attempts=3
)

# Generate and validate content
result = chain.run("Write a professional email...")
```

⚠️ **Note**: The previously used `Reflector` class is deprecated and will be removed in version 2.0.0. Please use the `Chain` class as shown above.

## Available Examples

1. **Basic Usage** (`basic_usage.py`):
   - Demonstrates fundamental pattern analysis capabilities
   - Shows how to use LengthRule, ProhibitedContentRule, SymmetryRule, and RepetitionRule
   - Example text analysis with multiple validation rules

2. **OpenAI Integration** (`openai_example.py`):
   - Shows integration with OpenAI's GPT models
   - Demonstrates text improvement using PromptCritic

3. **Pydantic Integration** (`pydantic_integration.py`):
   - Demonstrates using Pydantic models with Sifaka
   - Shows validation of structured data

4. **Combined Classifiers** (`combined_classifiers.py`):
   - Shows how to combine multiple classifiers
   - Demonstrates advanced pattern detection

## Requirements

- Python 3.7+
- Sifaka library with dependencies installed
- API keys for respective providers (OpenAI/Anthropic) in environment variables
- Python dotenv for environment management

## Usage

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   # For OpenAI examples
   export OPENAI_API_KEY='your-api-key'

   # For Anthropic examples
   export ANTHROPIC_API_KEY='your-api-key'
   ```

3. Run any example:
   ```bash
   python openai_example.py
   python basic_usage.py
   # etc.
   ```

## Best Practices

1. Always use proper configuration for rules:
   - Set appropriate thresholds and parameters
   - Use RuleConfig for structured configuration
   - Set proper priorities using RulePriority enum

2. Handle validation results appropriately:
   - Check both the passed status and metadata
   - Log validation messages for debugging
   - Use the critic for text improvement when needed

3. API Key Management:
   - Never hardcode API keys
   - Use environment variables or secure configuration
   - Follow provider-specific best practices

## Contributing

Feel free to contribute additional examples by:
1. Creating a new example file
2. Adding appropriate documentation
3. Ensuring all dependencies are listed
4. Submitting a pull request

For more information, see the main Sifaka documentation.