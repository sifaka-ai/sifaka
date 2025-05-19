"""
Example demonstrating the use of GPT-4 with Claude critics and classifiers.

This example shows how to:
1. Use OpenAI's GPT-4 model to generate potentially biased text
2. Apply language, bias, and profanity classifiers as validators
3. Use a smaller Anthropic model (Claude Haiku) for critics to improve problematic text
4. Log all intermediate steps in the process

The example intentionally prompts the model to generate potentially biased content,
then shows how the validators detect issues and how the critics improve the text.
"""

import logging
import os

from dotenv import load_dotenv

from sifaka import Chain
from sifaka.classifiers.bias import BiasClassifier
from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.profanity import ProfanityClassifier
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
    # Set a timeout for the entire process
    import signal

    def timeout_handler(_, __):
        logger.error("Execution timed out after 300 seconds")
        raise TimeoutError("Execution timed out")

    # Set a 120-second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)

    try:
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

        # OpenAI GPT-4 model for generating text
        openai_model = OpenAIModel(model_name="gpt-4", api_key=openai_api_key)

        # Smaller Anthropic model for critics
        anthropic_model = AnthropicModel(
            model_name="claude-3-haiku-20240307",
            api_key=anthropic_api_key,  # Using a smaller Claude model
        )

        # Create classifiers
        logger.info("Creating classifiers...")
        language_classifier = LanguageClassifier()
        profanity_classifier = ProfanityClassifier()

        # Create validators from classifiers
        logger.info("Creating validators...")
        language_validator = classifier_validator(
            classifier=language_classifier,
            valid_labels=["en"],  # Only accept English text
            threshold=0.8,
            name="english_language_validator",
        )

        # Create a bias classifier
        bias_classifier = BiasClassifier(threshold=0.6)  # Lower threshold for better performance

        # Create a validator using the bias classifier
        bias_validator = classifier_validator(
            classifier=bias_classifier,
            invalid_labels=["biased"],  # Reject biased text
            threshold=0.6,  # Lower threshold for better performance
            name="bias_validator",
        )

        profanity_validator = classifier_validator(
            classifier=profanity_classifier,
            invalid_labels=["profane"],  # Reject profane text
            threshold=0.7,
            name="profanity_validator",
        )

        # Create N-Critics critic with Anthropic model
        logger.info("Creating N-Critics critic with Anthropic model...")
        n_critics_critic = create_n_critics_critic(
            model=anthropic_model,
            num_critics=5,  # Use all 5 critics
            temperature=0.7,
            name="bias_correction_critic",
            # Note: max_refinement_iterations parameter is not actually used in the implementation
        )

        # Create a prompt that will likely generate content with some bias terms
        prompt = """
        Write a detailed essay about the differences between cat owners and dog owners.
        Use strong language and make sweeping generalizations.
        Suggest that some dog owners are ignorant about proper pet care.
        Don't worry about being balanced - be strongly in favor of cats.
        """

        print_separator("PROMPT")
        print(prompt)

        # First, generate text with the OpenAI GPT-4 model
        print_separator("GENERATING INITIAL TEXT")
        logger.info("Generating initial text with OpenAI GPT-4 model...")

        initial_text = openai_model.generate(
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
            Chain(
                max_improvement_iterations=1
            )  # Limit to 1 improvement iteration for better performance
            .with_model(openai_model)
            .validate_with(length(min_words=50, max_words=1000))
            .validate_with(language_validator)
            .validate_with(bias_validator)
            .validate_with(profanity_validator)
            .improve_with(n_critics_critic)
        )

        # Run the chain with the pre-generated text
        print_separator("RUNNING CHAIN WITH PRE-GENERATED TEXT")
        logger.info("Running chain with pre-generated text...")
        result = chain.run_with_text(initial_text)

        # Print a comparison between initial and final text
        print_separator("COMPARISON")
        print("Initial text length:", len(result.initial_text))
        print("Final text length:", len(result.text))
        print("Text changed:", result.initial_text != result.text)

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
            if (
                "Critiqued text" in record
                or "Improved text" in record
                or "Generated text" in record
            ):
                print(record)

        # Reset the alarm
        signal.alarm(0)

    except TimeoutError:
        print_separator("TIMEOUT")
        print("The example execution timed out after 120 seconds.")
        print("This is expected behavior to prevent the example from running too long.")
        print("Try reducing the number of critics or iterations further if needed.")

    except Exception as e:
        print_separator("ERROR")
        print(f"An error occurred: {str(e)}")
        logger.exception("Unexpected error in example")

        # Reset the alarm
        signal.alarm(0)


if __name__ == "__main__":
    main()
