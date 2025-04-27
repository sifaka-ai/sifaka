# Sifaka Examples

This directory contains example scripts demonstrating various features and integrations of the Sifaka library.

## Available Examples

1. **Basic Usage** (`basic_usage.py`):
   - Demonstrates fundamental pattern analysis capabilities
   - Shows how to use LengthRule, ProhibitedContentRule, SymmetryRule, and RepetitionRule
   - Example text analysis with multiple validation rules
   ```python
   from sifaka.rules import LengthRule, ProhibitedContentRule, SymmetryRule, RepetitionRule
   from sifaka.rules.base import RuleConfig, RulePriority

   # Create a length validation rule
   length_rule = LengthRule(
       name="length_check",
       description="Checks if output length is within bounds",
       config={
           "min_length": 30,
           "max_length": 100,
           "unit": "characters",
       },
   )

   # Create a symmetry rule with specific thresholds
   symmetry_rule = SymmetryRule(
       name="symmetry_check",
       description="Checks for text symmetry patterns",
       config=RuleConfig(
           priority=RulePriority.MEDIUM,
           metadata={
               "mirror_mode": "both",
               "symmetry_threshold": 0.8,
               "preserve_whitespace": True,
               "preserve_case": True,
               "ignore_punctuation": True,
           },
       ),
   )
   ```

2. **OpenAI Integration** (`openai_example.py`):
   - Shows integration with OpenAI's GPT models
   - Demonstrates text improvement using PromptCritic
   ```python
   from sifaka.models import OpenAIProvider
   from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
   from sifaka.models.base import ModelConfig

   # Initialize OpenAI provider
   openai_provider = OpenAIProvider(
       model_name="gpt-4-turbo-preview",
       config=ModelConfig(
           api_key=os.getenv("OPENAI_API_KEY"),
           temperature=0.7,
           max_tokens=1000,
       ),
   )

   # Create a critic with the OpenAI provider
   critic = PromptCritic(
       model=openai_provider,
       config=PromptCriticConfig(
           name="openai_critic",
           description="A critic that uses OpenAI to improve text",
           system_prompt="You are an expert editor that improves text.",
           temperature=0.7,
           max_tokens=1000,
       ),
   )
   ```

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

## Important Notes

⚠️ **Deprecation Warning**: The `Reflector` class is deprecated and will be removed in version 2.0.0. Please use `SymmetryRule` and `RepetitionRule` from `sifaka.rules.pattern_rules` instead.

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