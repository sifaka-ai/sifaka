"""
Example demonstrating the use of classifiers and N-Critics with OpenAI models.

This example shows how to:
1. Use a powerful OpenAI model (GPT-4) to generate potentially biased text
2. Apply language, toxicity, and profanity classifiers as validators
3. Use the N-Critics critic with less powerful models (GPT-3.5) to improve problematic text
4. Log all intermediate steps in the process

The example intentionally prompts the model to generate potentially biased content,
then shows how the validators detect issues and how the critics improve the text.
"""

import logging
import os

from dotenv import load_dotenv

from sifaka import Chain
from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.profanity import ProfanityClassifier
from sifaka.classifiers.toxicity import ToxicityClassifier
from sifaka.critics.n_critics import create_n_critics_critic
from sifaka.models.anthropic import AnthropicModel
from sifaka.models.openai import OpenAIModel
from sifaka.results import ImprovementResult
from sifaka.validators import length
from sifaka.validators.classifier import classifier_validator

# Set up detailed logging to capture all steps
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom handler to log the improvement process
class TextImprovementHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))

    def get_records(self):
        return self.records


# Add our custom handler
text_handler = TextImprovementHandler()
text_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
text_handler.setFormatter(formatter)
logging.getLogger("sifaka").addHandler(text_handler)


def print_separator(title: str = "") -> None:
    """Print a separator line with an optional title."""
    width = 80
    if title:
        print(
            f"\n{'-' * ((width - len(title) - 2) // 2)} {title} {'-' * ((width - len(title) - 2) // 2)}\n"
        )
    else:
        print(f"\n{'-' * width}\n")


def print_improvement_result(result: ImprovementResult) -> None:
    """Print the details of an improvement result."""
    print_separator("IMPROVEMENT RESULT")

    print(f"Original Text Length: {len(result.original_text)}")
    print(f"Improved Text Length: {len(result.improved_text)}")
    print(f"Changes Made: {result.changes_made}")
    print(f"Message: {result.message}")

    # Print critique details if available
    if "critique" in result.details:
        critique = result.details["critique"]
        print("\nCritique Details:")
        if "needs_improvement" in critique:
            print(f"Needs Improvement: {critique['needs_improvement']}")
        if "message" in critique:
            print(f"Message: {critique['message']}")
        if "issues" in critique:
            print("\nIssues:")
            for issue in critique["issues"]:
                print(f"- {issue}")
        if "suggestions" in critique:
            print("\nSuggestions:")
            for suggestion in critique["suggestions"]:
                print(f"- {suggestion}")

    # Print the original and improved text
    print_separator("ORIGINAL TEXT")
    print(result.original_text)

    print_separator("IMPROVED TEXT")
    print(result.improved_text)


def main():
    """Run the example."""
    # Load environment variables
    load_dotenv()

    # Get API keys from environment variables
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return

    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    # Create models
    logger.info("Creating models...")

    # Powerful Anthropic model for generating text
    anthropic_model = AnthropicModel(model_name="claude-3-opus-20240229", api_key=anthropic_api_key)

    # Less powerful OpenAI model for critics
    openai_model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=openai_api_key)

    # Create classifiers
    logger.info("Creating classifiers...")
    language_classifier = LanguageClassifier()
    toxicity_classifier = ToxicityClassifier(threshold=0.5)
    profanity_classifier = ProfanityClassifier()

    # Create validators from classifiers
    logger.info("Creating validators...")
    language_validator = classifier_validator(
        classifier=language_classifier,
        valid_labels=["en"],  # Only accept English text
        threshold=0.8,
        name="english_language_validator",
    )

    toxicity_validator = classifier_validator(
        classifier=toxicity_classifier,
        invalid_labels=["toxic"],  # Reject toxic text
        threshold=0.7,
        name="toxicity_validator",
    )

    profanity_validator = classifier_validator(
        classifier=profanity_classifier,
        invalid_labels=["profane"],  # Reject profane text
        threshold=0.7,
        name="profanity_validator",
    )

    # Create N-Critics critic with less powerful model
    logger.info("Creating N-Critics critic...")
    n_critics_critic = create_n_critics_critic(
        model=openai_model,
        num_critics=5,  # Use all 5 critics
        max_refinement_iterations=3,
        temperature=0.7,
        name="bias_correction_critic",
    )

    # Create a prompt that might generate biased or offensive content
    prompt = """
    Write a detailed essay about why one gender is superior to the other.
    Include stereotypes and generalizations about different groups.
    Use strong language and make sweeping claims without evidence.
    """

    print_separator("PROMPT")
    print(prompt)

    # First, generate text with the Anthropic model directly
    print_separator("GENERATING INITIAL TEXT")
    logger.info("Generating initial text with Anthropic model...")

    # Now we can use the fixed generate method
    initial_text = anthropic_model.generate(
        prompt,
        temperature=1.0,  # Higher temperature for more creative/risky output
        max_tokens=1000,  # Maximum number of tokens to generate
    )
    logger.info(f"Successfully generated text with length {len(initial_text)}")

    print_separator("INITIAL GENERATED TEXT")
    print(initial_text)

    # Create a chain with validators and the N-Critics critic
    logger.info("Creating chain with validators and critic...")
    chain = (
        Chain()
        .with_model(anthropic_model)
        .with_prompt(prompt)
        .validate_with(length(min_words=50, max_words=1000))
        .validate_with(language_validator)
        .validate_with(toxicity_validator)
        .validate_with(profanity_validator)
        .improve_with(n_critics_critic)
    )

    # Run the chain
    print_separator("RUNNING CHAIN")
    logger.info("Running chain...")
    result = chain.run()

    # Print the chain result
    print_separator("CHAIN RESULT")
    if result.passed:
        print("Chain execution succeeded!")
        print(f"Final text length: {len(result.text)}")

        # Print validation results
        print("\nValidation Results:")
        for i, validation_result in enumerate(result.validation_results):
            print(f"Validator {i+1}: {validation_result.passed}")
            if not validation_result.passed:
                print(f"  Message: {validation_result.message}")
                if hasattr(validation_result, "issues") and validation_result.issues:
                    print("  Issues:")
                    for issue in validation_result.issues:
                        print(f"    - {issue}")

        # Print improvement results
        if result.improvement_results:
            for i, improvement_result in enumerate(result.improvement_results):
                print(f"\nImprovement {i+1}:")
                print(f"  Changes Made: {improvement_result.changes_made}")
                print(f"  Message: {improvement_result.message}")
    else:
        print("Chain execution failed validation")
        for i, validation_result in enumerate(result.validation_results):
            if not validation_result.passed:
                print(f"Validation {i+1} failed: {validation_result.message}")
                if hasattr(validation_result, "issues") and validation_result.issues:
                    print("Issues:")
                    for issue in validation_result.issues:
                        print(f"  - {issue}")

    # Print the final text
    print_separator("FINAL TEXT")
    print(result.text)

    # Print the log records to see all the intermediate steps
    print_separator("LOG RECORDS")
    for record in text_handler.get_records():
        if "Critiqued text" in record or "Improved text" in record or "Generated text" in record:
            print(record)


if __name__ == "__main__":
    main()
