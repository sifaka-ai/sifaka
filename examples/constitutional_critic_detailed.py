"""
A detailed example of using the ConstitutionalCritic in Sifaka.

This example demonstrates how the ConstitutionalCritic improves text based on
constitutional principles, showing the before and after results for each improvement.
"""

import logging
import os

from dotenv import load_dotenv

from sifaka import Chain
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.models.openai import OpenAIModel
from sifaka.results import ImprovementResult
from sifaka.validators import length

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_improvement_result(result: ImprovementResult, iteration: int = None) -> None:
    """Print the details of an improvement result.

    Args:
        result: The improvement result to print
        iteration: Optional iteration number to display
    """
    iteration_str = f" (Iteration {iteration})" if iteration is not None else ""

    print(f"\n{'=' * 80}")
    print(f"IMPROVEMENT RESULT{iteration_str}")
    print(f"{'=' * 80}")

    print(f"\nChanges made: {result.changes_made}")
    print(f"Message: {result.message}")

    if result.changes_made:
        print("\n----- ORIGINAL TEXT -----")
        print(result.original_text)
        print("\n----- IMPROVED TEXT -----")
        print(result.improved_text)

    # Print additional details if available
    if hasattr(result, "details") and result.details:
        if "critique" in result.details:
            print("\n----- CRITIQUE -----")
            critique = result.details["critique"]
            if isinstance(critique, dict):
                for key, value in critique.items():
                    if key != "improved_text":  # Skip improved_text as we already printed it
                        print(f"\n{key.upper()}:")
                        print(value)
            else:
                print(critique)


def main():
    """Run a detailed example with the ConstitutionalCritic."""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    # Create a model
    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

    # Define constitutional principles
    principles = [
        "Content should be factually accurate and well-supported with specific examples.",
        "Explanations should be clear and accessible to non-experts.",
        "Content should present a balanced view of both benefits and risks.",
        "Content should include diverse perspectives from different stakeholders.",
        "Historical information should be accurate and include key milestones with dates.",
        "Future predictions should acknowledge uncertainty and avoid overly deterministic claims.",
    ]

    # Create a constitutional critic
    constitutional_critic = create_constitutional_critic(
        model=model,
        principles=principles,
    )

    # Print the principles
    print("\nConstitutional Principles:")
    for i, principle in enumerate(principles, 1):
        print(f"{i}. {principle}")

    # Create a prompt that might need improvement
    prompt = """
    Explain the history of artificial intelligence and its potential future impact on society.
    """

    print(f"\nOriginal Prompt: {prompt}")

    # First, generate text with the model directly
    print("\nGenerating initial text...")
    initial_text = model.generate(prompt)

    print("\n----- INITIAL GENERATED TEXT -----")
    print(initial_text)

    # Now, apply the constitutional critic manually to see each step
    print("\nApplying ConstitutionalCritic...")

    # Get the critique and improved text
    improved_text, improvement_result = constitutional_critic.improve(initial_text)

    # Print the improvement result
    print_improvement_result(improvement_result)

    # Now run a full chain to see the complete process
    print("\n\nRunning a complete chain with ConstitutionalCritic...")

    # Create a custom validator that always provides suggestions
    from sifaka.interfaces import Validator
    from sifaka.results import ValidationResult

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
                "Consider adding more diverse perspectives",
                "Include more specific dates and milestones",
                "Acknowledge uncertainty in future predictions",
            ]
            return result

    # Create a chain with the constitutional critic and our custom validator
    chain = (
        Chain()
        .with_model(model)
        .with_prompt(prompt)
        .validate_with(length(min_words=50, max_words=500))
        .validate_with(SuggestionValidator())  # Add our custom validator that provides suggestions
        .improve_with(constitutional_critic)
    )

    # Run the chain
    result = chain.run()

    # Print the final result
    print("\n----- FINAL RESULT FROM CHAIN -----")
    if result.passed:
        print("Chain execution succeeded!")
        print(result.text)
    else:
        print("Chain execution failed validation")
        for i, validation_result in enumerate(result.validation_results):
            if not validation_result.passed:
                print(f"Validation {i+1} failed: {validation_result.message}")

    # Print improvement details
    if hasattr(result, "improvement_results") and result.improvement_results:
        print("\n----- IMPROVEMENT DETAILS -----")
        for i, imp_result in enumerate(result.improvement_results):
            print_improvement_result(imp_result, i + 1)


if __name__ == "__main__":
    main()
