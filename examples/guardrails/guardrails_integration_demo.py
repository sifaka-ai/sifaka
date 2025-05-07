"""
Example demonstrating integration of Guardrails validators with Sifaka.

This example shows how to:
1. Create a properly registered Guardrails validator
2. Use the Guardrails adapter to create a Sifaka rule
3. Use the rule in a Sifaka chain
"""

import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file (containing ANTHROPIC_API_KEY)
load_dotenv()

# Verify API key is available
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError(
        "ANTHROPIC_API_KEY environment variable not set. "
        "Please set it in your environment or in a .env file."
    )

# Import Sifaka components
from sifaka.models.anthropic import create_anthropic_provider
from sifaka.models.base import ModelConfig
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import ChainOrchestrator

# Import Guardrails components
try:
    # This will work if you've installed guardrails with: pip install guardrails-ai
    from guardrails.validator_base import Validator, register_validator
    from guardrails.classes import ValidationResult, PassResult, FailResult

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    print("⚠️ Guardrails is not installed. Please install it with 'pip install guardrails-ai'")

# Import the Guardrails adapter
from sifaka.adapters.guardrails import create_guardrails_rule

# Configure Claude model using the factory function
model = create_anthropic_provider(
    model_name="claude-3-sonnet-20240229",
    temperature=0.7,
    max_tokens=1500,
    api_key=api_key,  # Use the verified API key
    trace_enabled=True,
)

# Only proceed if Guardrails is available
if GUARDRAILS_AVAILABLE:
    # Create a custom Guardrails validator with proper registration
    @register_validator(name="phone_number_validator", data_type="string")
    class PhoneNumberValidator(Validator):
        """Validator that checks if a value matches a US phone number pattern."""

        rail_alias = "phone_number_validator"

        def __init__(self, on_fail="exception"):
            """Initialize the validator."""
            super().__init__(on_fail=on_fail)
            self.pattern = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

        def _validate(self, value, metadata):
            """Validate if the value contains a valid US phone number."""
            if self.pattern.search(value):
                return PassResult(actual_value=value, validated_value=value)
            else:
                return FailResult(
                    actual_value=value,
                    error_message="Value must contain a valid US phone number (e.g., 555-123-4567)",
                )

    # Create an instance of our custom validator
    phone_validator = PhoneNumberValidator(on_fail="exception")

    # Create a Sifaka rule using the Guardrails validator
    phone_rule = create_guardrails_rule(
        guardrails_validator=phone_validator, rule_id="phone_number_format"
    )

    # Create a critic to help improve responses that don't meet the rule
    critic = PromptCritic(
        llm_provider=model,
        config=PromptCriticConfig(
            name="phone_format_critic",
            description="Helps ensure text contains a properly formatted phone number",
            system_prompt=(
                "You are a helpful editor who specializes in ensuring text contains "
                "properly formatted US phone numbers in the format (XXX) XXX-XXXX, "
                "XXX-XXX-XXXX, or similar standard formats. Make sure the response "
                "includes at least one properly formatted phone number."
            ),
            temperature=0.5,
        ),
    )

    # Create a chain with the model, rule, and critic
    chain = ChainOrchestrator(model=model, rules=[phone_rule], critic=critic, max_attempts=3)

    # Prompt designed to generate a response with a phone number
    prompt = """
    Please provide a fake customer service phone number for Acme Corporation.
    Include just the number in your response, no other text.
    """

    # Run the chain - it will generate text, check using the Guardrails validator,
    # and if needed, use the critic to improve the output
    try:
        result = chain.run(prompt)
        print(f"✅ Final output:")
        print(result.output)

        if result.critique_details:
            print("\n🔍 Critique details:")
            for key, value in result.critique_details.items():
                print(f"  {key}: {value}")

    except ValueError as e:
        print(f"❌ Error: {e}")
else:
    print("⚠️ Example skipped because Guardrails is not installed.")
