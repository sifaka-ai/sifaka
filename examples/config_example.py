"""
An example of using Sifaka with configuration.

This example demonstrates how to create a custom configuration for Sifaka
and use it with a chain.
"""

import os

from dotenv import load_dotenv

from sifaka import Chain
from sifaka.config import CriticConfig, ModelConfig, SifakaConfig, ValidatorConfig
from sifaka.critics.self_refine import create_self_refine_critic
from sifaka.validators import length, prohibited_content

# Load environment variables from .env file if it exists
load_dotenv()

# Get API key from environment variables
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Create a custom configuration
config = SifakaConfig(
    model=ModelConfig(
        temperature=0.8,
        max_tokens=500,
        top_p=0.9,
        api_key=api_key,  # Use environment variable for API key
    ),
    validator=ValidatorConfig(
        min_words=100, max_words=500, prohibited_content=["violence", "hate speech"]
    ),
    critic=CriticConfig(
        temperature=0.5,
        refinement_rounds=3,
        system_prompt="You are an expert editor that improves text for clarity and conciseness.",
    ),
    debug=True,  # Set to True to enable preliminary check with improver
    log_level="DEBUG",
)

from sifaka.interfaces import Validator

# Create a model
from sifaka.models.openai import OpenAIModel
from sifaka.results import ValidationResult  # Use the concrete class, not the Protocol


# Create a custom validator that always provides suggestions
class SuggestionValidator(Validator):
    """A validator that always passes but provides suggestions for improvement."""

    def validate(self, text: str) -> ValidationResult:
        """Validate the text and always provide suggestions."""
        result = ValidationResult(
            passed=True,  # Always pass validation
            message="Text passes validation but could be improved",
        )
        # Set suggestions
        result.suggestions = [
            "Consider adding more descriptive language",
            "Try to develop the characters more deeply",
            "Add more emotional depth to the story",
        ]
        return result


model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

# Use the configuration with a chain
chain = (
    Chain(config)
    .with_model(model)
    .with_prompt("Write a short story about a robot.")
    .validate_with(
        length(min_words=config.validator.min_words, max_words=config.validator.max_words)
    )
    .validate_with(prohibited_content(prohibited=config.validator.prohibited_content))
    .validate_with(SuggestionValidator())  # Add our custom validator that provides suggestions
    .improve_with(
        create_self_refine_critic(
            model=model, max_refinement_iterations=config.critic.refinement_rounds
        )
    )
)

# Run the chain
result = chain.run()

# Check the result
if result.passed:
    print("Chain execution succeeded!")

    # Print information about improvements
    if result.improvement_results:
        print(f"\nImprovements applied: {len(result.improvement_results)}")

        # Get the original text (before any improvements)
        original_text = (
            result.improvement_results[0].original_text
            if result.improvement_results
            else result.text
        )

        print("\n" + "=" * 80)
        print("ORIGINAL TEXT (before improvements)")
        print("=" * 80)
        print(original_text)

        # Print details of each improvement
        for i, improvement in enumerate(result.improvement_results):
            print("\n" + "=" * 80)
            print(f"IMPROVEMENT {i+1}")
            print("=" * 80)
            print(f"Changes made: {improvement.changes_made}")
            print(f"Message: {improvement.message}")

            if hasattr(improvement, "details") and improvement.details:
                if "critique" in improvement.details:
                    print("\nCritique:")
                    critique = improvement.details["critique"]
                    if isinstance(critique, dict):
                        for key, value in critique.items():
                            if key not in ["improved_text"]:
                                print(f"  {key}: {value}")
    else:
        print("\nNo improvements were applied - text passed validation and didn't need improvement")

    print("\n" + "=" * 80)
    print("FINAL TEXT")
    print("=" * 80)
    print(result.text)
else:
    print("Chain execution failed validation")
    for i, validation_result in enumerate(result.validation_results):
        if not validation_result.passed:
            print(f"Validation {i+1} failed: {validation_result.message}")
