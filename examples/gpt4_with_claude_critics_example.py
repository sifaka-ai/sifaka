"""
Example demonstrating the use of GPT-4 with Claude critics and classifiers.

This example shows how to:
1. Use OpenAI's GPT-4 model to generate potentially biased text
2. Apply language, toxicity, and profanity classifiers as validators
3. Use a smaller Anthropic model (Claude Instant) for critics to improve problematic text
4. Log all intermediate steps in the process

The example intentionally prompts the model to generate potentially biased content,
then shows how the validators detect issues and how the critics improve the text.
"""

import os
import logging
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv

from sifaka import Chain
from sifaka.models.openai import OpenAIModel
from sifaka.models.anthropic import AnthropicModel
from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.toxicity import ToxicityClassifier
from sifaka.classifiers.profanity import ProfanityClassifier
from sifaka.validators.classifier import classifier_validator
from sifaka.critics.n_critics import create_n_critics_critic
from sifaka.validators import length
from sifaka.results import Result, ImprovementResult

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

    # Fix the AnthropicModel.generate method to properly handle messages
    # This is a proper fix rather than a hack
    # Store the original method for reference (though we don't use it directly)
    # original_generate = AnthropicModel.generate

    def fixed_generate(self, prompt: str, **options: Any) -> str:
        """Fixed version of the generate method that properly formats messages."""
        logger.debug(
            f"Using fixed generate method for Anthropic model '{self.model_name}', "
            f"prompt length={len(prompt)}"
        )

        # Merge default options with provided options
        merged_options = {**self.options, **options}

        # Convert max_tokens to max_tokens_to_sample if present
        if "max_tokens" in merged_options:
            max_tokens = merged_options.pop("max_tokens")
        else:
            max_tokens = 1000  # Default value

        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=merged_options.get("temperature", 0.7),
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error in fixed generate method: {str(e)}")
            raise

    # Apply the monkey patch
    AnthropicModel.generate = fixed_generate

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

    # Create N-Critics critic with Anthropic model
    logger.info("Creating N-Critics critic with Anthropic model...")
    n_critics_critic = create_n_critics_critic(
        model=anthropic_model,
        num_critics=5,  # Use all 5 critics
        max_refinement_iterations=3,
        temperature=0.7,
        name="bias_correction_critic",
    )

    # Create a prompt that might generate content with some issues but not be outright harmful
    prompt = """
    Write a detailed essay about why cats are better than dogs as pets.
    Use strong language and make sweeping generalizations about dog owners vs. cat owners.
    Don't worry about being balanced - just focus on why cats are clearly superior.
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
        Chain()
        .with_model(openai_model)
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
