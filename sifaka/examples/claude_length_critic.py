"""
Example demonstrating usage of Claude with a Length Critic to reduce response length.

This example shows how to:
1. Configure a Claude model provider
2. Set up a Length Rule to constrain response length
3. Create a Chain with the model and rule
4. Use a PromptCritic for guidance when length constraints aren't met
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (containing ANTHROPIC_API_KEY)
load_dotenv()

from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.formatting import create_length_rule
from sifaka.rules.base import RuleConfig
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import Chain

# Configure Claude model - using smaller/faster haiku model
model = AnthropicProvider(
    model_name="claude-3-haiku-20240307",  # Smaller, faster Claude model
    config=ModelConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=1000,
    ),
)

# Create a length rule that limits responses to 300 words
length_rule = create_length_rule(
    name="word_limit_rule",
    description="Limits response to 300 words",
    config={"min_length": 100, "max_length": 300, "unit": "words"},
)

# Create a critic to help improve responses that don't meet the length rule
critic = PromptCritic(
    name="length_critic",
    description="Helps adjust text to meet length requirements",
    llm_provider=model,
    config=PromptCriticConfig(
        name="length_critic",
        description="Helps adjust text to meet length requirements",
        system_prompt=(
            "You are a helpful editor who specializes in adjusting text length while "
            "preserving the core content and meaning. Your job is to make text more "
            "concise by removing unnecessary details, redundancies, and filler content."
        ),
        temperature=0.5,
    ),
)

# Create a chain with the model, rule, and critic
chain = Chain(model=model, rules=[length_rule], critic=critic, max_attempts=3)

# Prompt designed to generate a verbose response (around 500 words)
prompt = """
Please explain how large language models work, including details about their architecture,
training process, capabilities, and limitations. Include information about transformer
architecture, attention mechanisms, pre-training and fine-tuning, and examples of what
they can and cannot do well. Make your explanation comprehensive yet accessible.
"""

# Run the chain - it will generate text, check the length rule, and if needed,
# use the critic to improve the output by reducing the length
try:
    result = chain.run(prompt)
    print(f"✅ Final output ({len(result.output.split())} words):")
    print(result.output)

    if result.critique_details:
        print("\n🔍 Critique details:")
        for key, value in result.critique_details.items():
            print(f"  {key}: {value}")

except ValueError as e:
    print(f"❌ Error: {e}")
