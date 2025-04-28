"""
Example demonstrating usage of Claude with a Length Critic to expand response length.

This example shows how to:
1. Configure a Claude model provider
2. Set up a Length Rule that requires longer responses (1000-2000 words)
3. Create a Chain with the model and rule
4. Use a PromptCritic to guide Claude to produce more detailed responses
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (containing ANTHROPIC_API_KEY)
load_dotenv()

from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import Chain

# Configure Claude model - using smaller/faster haiku model
model = AnthropicProvider(
    model_name="claude-3-haiku-20240307",  # Smaller, faster Claude model
    config=ModelConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=4000,  # Increased to allow for longer responses
    ),
)

# Create a length rule that requires responses between 1000-2000 words
length_rule = create_length_rule(
    min_words=1000,  # Minimum 1000 words
    max_words=2000,  # Maximum 2000 words
    rule_id="expanded_length_rule",
)

# Create a critic to help improve responses that don't meet the length requirements
critic = PromptCritic(
    name="expansion_critic",
    description="Helps expand text to meet minimum length requirements",
    llm_provider=model,
    config=PromptCriticConfig(
        name="expansion_critic",
        description="Helps expand text to meet minimum length requirements",
        system_prompt=(
            "You are a helpful editor who specializes in expanding text while maintaining quality. "
            "Your job is to make text more detailed and comprehensive by adding relevant examples, "
            "elaborating on key points, providing context, exploring implications, and addressing "
            "potential questions readers might have. Never add fluff or redundant information."
        ),
        temperature=0.7,
    ),
)

# Create a chain with the model, rule, and critic
chain = Chain(model=model, rules=[length_rule], critic=critic, max_attempts=3)

# Prompt designed to generate a brief response (around 150 words)
prompt = """
Briefly explain what artificial intelligence is and some of its basic applications.
Keep your explanation short and to the point.
"""

# Run the chain - it will generate text, check the length rule, and if needed,
# use the critic to improve the output by expanding the content
try:
    result = chain.run(prompt)
    word_count = len(result.output.split())
    print(f"✅ Final output ({word_count} words):")
    print(f"First 200 words: {' '.join(result.output.split()[:200])}...")
    print(f"\n[Full response contains {word_count} words total]")

    if result.critique_details:
        print("\n🔍 Critique details:")
        for key, value in result.critique_details.items():
            if key == "feedback":
                print(f"  {key}: {value[:150]}..." if len(value) > 150 else f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

except ValueError as e:
    print(f"❌ Error: {e}")
