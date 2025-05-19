"""
A detailed example of using the ConstitutionalCritic in Sifaka.

This example demonstrates how the ConstitutionalCritic improves text based on
constitutional principles, showing the before and after results for each improvement.

NOTE: After recent changes to the Chain class, critics are only applied when validation fails.
This example has been updated to demonstrate this behavior.
"""

import logging
import os
import sys

from dotenv import load_dotenv

from sifaka import Chain
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.models.openai import OpenAIModel
from sifaka.results import ImprovementResult, ValidationResult
from sifaka.validators import length

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enable debug logging for sifaka modules
sifaka_logger = logging.getLogger("sifaka")
sifaka_logger.setLevel(logging.DEBUG)

# Add a handler to output to console
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
sifaka_logger.addHandler(handler)


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
    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key, max_tokens=2500)

    # Define constitutional principles focused on addressing bias
    principles = [
        "Content should be inclusive and avoid gender-specific language when referring to all people.",
        "Content should acknowledge diverse perspectives from different cultures, regions, and socioeconomic backgrounds.",
        "Content should avoid deterministic language about the future and acknowledge uncertainty.",
        "Content should use neutral terminology and avoid outdated or potentially offensive terms.",
        "Content should avoid overgeneralizations and acknowledge exceptions and nuances.",
        "Content should present balanced views and avoid centering any particular cultural perspective.",
        "Content should avoid language that implies hierarchies between countries, cultures, or technologies.",
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

    # Create a prompt that might lead to biased content
    prompt = """
    Explain how mankind has developed artificial intelligence and how it will inevitably transform society in developed countries.
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

    # Create a custom validator that fails validation
    from sifaka.interfaces import Validator
    from sifaka.results import ValidationResult

    class BiasValidator(Validator):
        """A validator that checks for bias in the text and fails validation if bias is detected."""

        def validate(self, text: str) -> ValidationResult:
            """Validate the text and fail if bias is detected."""
            # Simple bias detection - check for certain keywords or phrases
            # In a real implementation, this would use a more sophisticated bias detection algorithm

            # Check for bias indicators in the text
            bias_indicators = [
                ("only men", "Gender bias - excludes women and non-binary individuals"),
                ("mankind", "Gender bias - uses male-centric language"),
                ("all people", "Overgeneralization without acknowledging diversity"),
                (
                    "inevitable",
                    "Deterministic language about the future without acknowledging uncertainty",
                ),
                (
                    "will definitely",
                    "Deterministic language about the future without acknowledging uncertainty",
                ),
                ("western", "Geographic/cultural bias - centers Western perspectives"),
                ("developed countries", "Economic bias - implies a hierarchy of development"),
                ("third world", "Outdated and potentially offensive terminology"),
                ("primitive", "Potentially offensive when describing cultures or technologies"),
                ("obviously", "Assumes universal agreement on subjective matters"),
            ]

            detected_biases = []
            for term, explanation in bias_indicators:
                if term.lower() in text.lower():
                    detected_biases.append(f"{explanation} (found term: '{term}')")

            # Always fail validation for this example to trigger the critic
            # In a real implementation, we would only fail if biases are detected
            result = ValidationResult(
                passed=False,
                message="Text contains bias that needs to be addressed",
            )

            # If no specific biases were detected, add some general issues
            if not detected_biases:
                detected_biases = [
                    "Text lacks diverse perspectives and considerations",
                    "Text contains potentially biased language or framing",
                    "Text makes deterministic claims about the future without acknowledging uncertainty",
                ]

            result.issues = detected_biases
            return result

    # Create a chain with the constitutional critic and our bias validator
    chain = (
        Chain()
        .with_model(model)
        .with_prompt(prompt)
        .validate_with(
            length(min_words=50, max_words=2000)
        )  # Increased max_words to avoid length issues
        .validate_with(BiasValidator())  # Add our bias validator that fails validation
        .improve_with(constitutional_critic)
        .with_options(
            apply_improvers_on_validation_failure=True
        )  # Ensure critics are applied when validation fails
    )

    # Print debug information about the chain
    print("\n----- CHAIN CONFIGURATION -----")
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of validators: {len(chain._validators)}")
    print(f"Number of improvers: {len(chain._improvers)}")
    print(
        f"apply_improvers_on_validation_failure: {chain._options.get('apply_improvers_on_validation_failure', True)}"
    )

    # Run the chain
    print("\n----- RUNNING CHAIN -----")
    result = chain.run()

    # Print the final result
    print("\n----- FINAL RESULT FROM CHAIN -----")
    if result.passed:
        print("Chain execution succeeded!")
        print(f"Initial text length: {len(result.initial_text)}")
        print(f"Final text length: {len(result.text)}")
        print("\n----- FINAL TEXT -----")
        print(result.text)
    else:
        print("Chain execution failed validation")
        for i, validation_result in enumerate(result.validation_results):
            if not validation_result.passed:
                print(f"Validation {i+1} failed: {validation_result.message}")
                if hasattr(validation_result, "issues") and validation_result.issues:
                    print("Issues:")
                    for issue in validation_result.issues:
                        print(f"  - {issue}")

    # Print validation results
    print("\n----- VALIDATION RESULTS -----")
    print(f"Number of validation results: {len(result.validation_results)}")
    for i, validation_result in enumerate(result.validation_results):
        print(
            f"Validation {i+1}: passed={validation_result.passed}, message={validation_result.message}"
        )

    # Print improvement details
    print("\n----- IMPROVEMENT RESULTS -----")
    print(f"Number of improvement results: {len(result.improvement_results)}")
    if result.improvement_results:
        for i, imp_result in enumerate(result.improvement_results):
            print_improvement_result(imp_result, i + 1)
    else:
        print("No improvements were made. This could be because:")
        print("1. All validators passed and critics were not applied")
        print("2. Validation failed but apply_improvers_on_validation_failure was set to False")
        print("3. Critics were applied but didn't make any changes")


if __name__ == "__main__":
    main()
