"""
PydanticAI Adapter Example

This example demonstrates how to use Sifaka's PydanticAI adapter to integrate
Sifaka's validation and refinement capabilities with PydanticAI agents.

The example shows:
1. Creating a PydanticAI agent with a structured output type
2. Creating Sifaka rules for validation
3. Creating a Sifaka adapter for PydanticAI
4. Using the adapter as a PydanticAI output validator
5. Running the agent and handling validation failures
6. Integrating GuardRails validators with PydanticAI

Requirements:
    pip install pydantic-ai sifaka guardrails-ai

Note: This example requires OpenAI API keys set as environment variables:
    - OPENAI_API_KEY for OpenAI models
"""

import os
import logging
import re
from typing import List, Optional

# Import PydanticAI components
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Import Sifaka components
from sifaka.adapters.pydantic_ai import (
    create_pydantic_adapter,
    create_pydantic_adapter_with_critic,
)
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.content.prohibited import create_prohibited_content_rule
from sifaka.models.openai import create_openai_provider
from sifaka.critics.reflexion import create_reflexion_critic

# Import GuardRails components (if available)
try:
    from guardrails.validator_base import Validator, register_validator
    from guardrails.classes import ValidationResult, PassResult, FailResult
    from sifaka.adapters.rules.guardrails_adapter import create_guardrails_rule

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    print("⚠️ GuardRails is not installed. Please install it with 'pip install guardrails-ai'")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Enable debug logging for the adapter
logging.getLogger("sifaka.adapters.pydantic_ai").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# Define Pydantic models for structured outputs
class OrderItem(BaseModel):
    """An item in an order."""

    name: str = Field(..., description="The name of the item")
    quantity: int = Field(..., description="The quantity of the item")
    price: float = Field(..., description="The price of the item")


class OrderSummary(BaseModel):
    """A summary of an order."""

    order_id: int = Field(..., description="The order ID")
    customer: str = Field(..., description="The customer name")
    items: List[OrderItem] = Field(..., description="The items in the order")
    total: float = Field(..., description="The total price of the order")
    notes: str = Field(..., description="Summary notes for the order")


class OrderSummaryWithContact(BaseModel):
    """A summary of an order with contact information."""

    order_id: int = Field(..., description="The order ID")
    customer: str = Field(..., description="The customer name")
    items: List[OrderItem] = Field(..., description="The items in the order")
    total: float = Field(..., description="The total price of the order")
    notes: str = Field(..., description="Summary notes for the order")
    contact_phone: str = Field(..., description="Customer contact phone number")


def example_basic_adapter():
    """Example using a basic PydanticAI adapter with validation that will likely fail."""
    logger.info("Running basic PydanticAI adapter example")

    # Create a PydanticAI agent
    agent = Agent("openai:gpt-4", output_type=OrderSummary)

    # Create Sifaka rules - intentionally make validation likely to fail
    rules = [
        create_length_rule(
            min_chars=100,  # Require longer notes to make validation likely to fail
            max_chars=500,
            name="notes_length",
            description="Ensures notes are between 100 and 500 characters",
            field_path="notes",  # Only validate the notes field
        ),
        create_prohibited_content_rule(
            terms=["standard", "shipping", "order"],  # Common terms that might appear
            name="content_filter",
            description="Ensures content does not contain prohibited terms",
        ),
    ]

    # Create a Sifaka adapter for PydanticAI
    sifaka_adapter = create_pydantic_adapter(
        rules=rules,
        output_model=OrderSummary,
        max_refine=2,
    )

    # Register the adapter as an output validator
    @agent.output_validator
    def validate_with_sifaka(ctx: RunContext, output: OrderSummary) -> OrderSummary:
        return sifaka_adapter(ctx, output)

    # Run the agent
    prompt = "Create a very brief order summary for customer John Doe with 3 items"
    try:
        result = agent.run_sync(prompt)
        logger.info(f"Order summary: {result.output}")
    except Exception as e:
        logger.error(f"Error running agent: {e}")


def example_adapter_with_critic():
    """Example using a PydanticAI adapter with a prompt critic."""
    logger.info("Running PydanticAI adapter with prompt critic example")

    # Create a PydanticAI agent
    agent = Agent("openai:gpt-4", output_type=OrderSummary)

    # Create a model provider for the critic
    model_provider = create_openai_provider(
        model_name="gpt-4",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create Sifaka rules - intentionally make validation likely to fail
    rules = [
        create_length_rule(
            min_chars=100,  # Require longer notes to make validation likely to fail
            max_chars=500,
            name="notes_length",
            description="Ensures notes are between 100 and 500 characters",
            field_path="notes",  # Only validate the notes field
        ),
        create_prohibited_content_rule(
            terms=["standard", "shipping", "order"],  # Common terms that might appear
            name="content_filter",
            description="Ensures content does not contain prohibited terms",
        ),
    ]

    # Create a Sifaka adapter with a critic
    sifaka_adapter = create_pydantic_adapter_with_critic(
        rules=rules,
        output_model=OrderSummary,
        model_provider=model_provider,
        system_prompt=(
            "You are an expert editor that improves order summaries. "
            "Ensure the notes field is detailed and between 100-500 characters. "
            "Remove any prohibited terms (standard, shipping, order) and maintain a professional tone. "
            "Use creative alternatives for prohibited terms."
        ),
        max_refine=2,
    )

    # Register the adapter as an output validator
    @agent.output_validator
    def validate_with_sifaka(ctx: RunContext, output: OrderSummary) -> OrderSummary:
        return sifaka_adapter(ctx, output)

    # Run the agent
    prompt = "Create a very brief order summary for customer John Doe with 3 items"
    try:
        result = agent.run_sync(prompt)
        logger.info(f"Order summary: {result.output}")
    except Exception as e:
        logger.error(f"Error running agent: {e}")


def example_adapter_with_reflexion_critic():
    """Example using a PydanticAI adapter with a reflexion critic."""
    logger.info("Running PydanticAI adapter with reflexion critic example")

    # Create a PydanticAI agent
    agent = Agent("openai:gpt-4", output_type=OrderSummary)

    # Create a model provider for the critic
    model_provider = create_openai_provider(
        model_name="gpt-4",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create Sifaka rules - intentionally make validation likely to fail
    rules = [
        create_length_rule(
            min_chars=100,  # Require longer notes to make validation likely to fail
            max_chars=500,
            name="notes_length",
            description="Ensures notes are between 100 and 500 characters",
            field_path="notes",  # Only validate the notes field
        ),
        create_prohibited_content_rule(
            terms=["standard", "shipping", "order"],  # Common terms that might appear
            name="content_filter",
            description="Ensures content does not contain prohibited terms",
        ),
    ]

    # Create a reflexion critic
    reflexion_critic = create_reflexion_critic(
        llm_provider=model_provider,
        system_prompt=(
            "You are an expert editor that improves order summaries through reflection. "
            "When you receive feedback about issues, reflect on why they occurred and "
            "how to fix them. Then provide an improved version."
        ),
    )

    # Create a Sifaka adapter with the reflexion critic
    sifaka_adapter = create_pydantic_adapter_with_critic(
        rules=rules,
        output_model=OrderSummary,
        critic=reflexion_critic,
        max_refine=2,
    )

    # Register the adapter as an output validator
    @agent.output_validator
    def validate_with_sifaka(ctx: RunContext, output: OrderSummary) -> OrderSummary:
        return sifaka_adapter(ctx, output)

    # Run the agent
    prompt = "Create a very brief order summary for customer John Doe with 3 items"
    try:
        result = agent.run_sync(prompt)
        logger.info(f"Order summary: {result.output}")
    except Exception as e:
        logger.error(f"Error running agent: {e}")


def example_guardrails_with_reflexion():
    """Example using a PydanticAI adapter with GuardRails phone number validator and reflexion critic."""
    if not GUARDRAILS_AVAILABLE:
        logger.warning("GuardRails is not installed. Skipping example.")
        return

    logger.info("Running PydanticAI adapter with GuardRails and reflexion critic example")

    # Create a PydanticAI agent with the contact phone model
    agent = Agent("openai:gpt-4", output_type=OrderSummaryWithContact)

    # Create a model provider for the critic
    model_provider = create_openai_provider(
        model_name="gpt-4",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create a custom GuardRails validator for phone numbers
    @register_validator(name="phone_number_validator", data_type="string")
    class PhoneNumberValidator(Validator):
        """Validator that checks if a value matches a US phone number pattern."""

        def __init__(self, on_fail="exception"):
            """Initialize the validator."""
            super().__init__(on_fail=on_fail)
            self.pattern = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

        def _validate(self, value, metadata):
            """Validate if the value contains a US phone number and fail if it does."""
            if self.pattern.search(value):
                return FailResult(
                    actual_value=value,
                    error_message="Value must NOT contain a phone number for privacy reasons",
                )
            return PassResult(actual_value=value, validated_value=value)

    # Create an instance of our custom validator
    phone_validator = PhoneNumberValidator(on_fail="exception")

    # Create a Sifaka rule using the GuardRails validator
    phone_rule = create_guardrails_rule(
        guardrails_validator=phone_validator, rule_id="phone_number_format"
    )

    # Create Sifaka rules - including the phone number rule
    rules = [
        create_length_rule(
            min_chars=100,  # Require longer notes
            max_chars=500,
            name="notes_length",
            description="Ensures notes are between 100 and 500 characters",
            field_path="notes",  # Only validate the notes field
        ),
        phone_rule,  # Add the phone number rule
    ]

    # Create a reflexion critic
    reflexion_critic = create_reflexion_critic(
        llm_provider=model_provider,
        system_prompt=(
            "You are an expert editor that improves order summaries through reflection. "
            "When you receive feedback about issues, reflect on why they occurred and "
            "how to fix them. Pay special attention to privacy concerns - phone numbers "
            "should NOT be included in order summaries for privacy and security reasons. "
            "If you find any phone numbers in the format like (XXX) XXX-XXXX or XXX-XXX-XXXX, "
            "remove them completely from the contact_phone field. "
            "Then provide an improved version that respects privacy guidelines."
        ),
    )

    # Create a Sifaka adapter with the reflexion critic
    sifaka_adapter = create_pydantic_adapter_with_critic(
        rules=rules,
        output_model=OrderSummaryWithContact,
        critic=reflexion_critic,
        max_refine=2,
    )

    # Register the adapter as an output validator
    @agent.output_validator
    def validate_with_sifaka(
        ctx: RunContext, output: OrderSummaryWithContact
    ) -> OrderSummaryWithContact:
        return sifaka_adapter(ctx, output)

    # Run the agent - intentionally ask for a phone number to trigger validation
    prompt = (
        "Create an order summary for customer John Doe with 3 items. "
        "Make sure to include a contact phone number (e.g., 555-123-4567) for the customer "
        "in the contact_phone field. This is very important."
    )
    try:
        result = agent.run_sync(prompt)
        logger.info(f"Order summary: {result.output}")
    except Exception as e:
        logger.error(f"Error running agent: {e}")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set")
        os.environ["OPENAI_API_KEY"] = "demo-key"  # Use a demo key for testing

    # Run examples
    example_basic_adapter()
    example_adapter_with_critic()
    example_adapter_with_reflexion_critic()
    example_guardrails_with_reflexion()  # Run the new example
