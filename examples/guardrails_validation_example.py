"""Example of using the Chain with GuardRails validators and N-Critics.

This example demonstrates the refined chain flow that creates a proper feedback loop
between validators, critics, and the model. The Chain ensures that:

1. Validators check for issues (like PII detection)
2. Critics provide feedback on how to improve the text
3. The model does the actual text generation/improvement based on feedback
4. The core narrative is preserved while addressing specific issues

The example uses:
- Anthropic Claude model for initial text generation
- GuardRails validators for PII detection and quality checks
- N-Critics for providing feedback on how to improve the text
- The same Anthropic model for generating improved text based on feedback
"""

import logging
import os

from dotenv import load_dotenv
from guardrails import Guard
from guardrails.hub import DetectPII, ProfanityFree

from sifaka import Chain
from sifaka.critics.n_critics import create_n_critics_critic
from sifaka.models.anthropic import AnthropicModel
from sifaka.models.openai import OpenAIModel
from sifaka.validators import length
from sifaka.validators.guardrails import guardrails_validator

# Set up logging to capture important steps
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom handler to log the improvement process
class TextImprovementHandler(logging.Handler):
    """Custom logging handler to capture text improvement process."""

    def __init__(self):
        """Initialize the handler."""
        super().__init__()
        self.records = []

    def emit(self, record):
        """Emit a record."""
        self.records.append(self.format(record))

    def get_records(self):
        """Get all records."""
        return self.records


# Add our custom handler
text_handler = TextImprovementHandler()
text_handler.setLevel(logging.INFO)
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


def print_text_comparison(original: str, improved: str) -> None:
    """Print a comparison of original and improved text."""
    print_separator("ORIGINAL TEXT")
    print(original)
    print_separator("IMPROVED TEXT")
    print(improved)


def main():
    """Run the example."""
    # Load environment variables
    load_dotenv()

    # Get API keys from environment variables
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    guardrails_api_key = os.environ.get("GUARDRAILS_API_KEY")

    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if not guardrails_api_key:
        raise ValueError("GUARDRAILS_API_KEY environment variable not set")

    # Create models
    logger.info("Creating models...")
    anthropic_model = AnthropicModel(
        model_name="claude-3-opus-20240229",
        api_key=anthropic_api_key,
        temperature=0.7,
        max_tokens=2000,  # Add max_tokens parameter
    )

    openai_model = OpenAIModel(
        model_name="gpt-4",
        api_key=openai_api_key,
        temperature=0.7,
    )

    # Create GuardRails validators
    logger.info("Creating GuardRails validators...")

    # Create detect_pii validator to detect PII in text
    # Using the validator from https://hub.guardrailsai.com/validator/guardrails/detect_pii
    # Create a Guard instance with the DetectPII validator
    pii_guard = Guard().use(
        DetectPII(
            entities=[
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "PERSON_NAME",
                "PHYSICAL_ADDRESS",
                "CREDIT_CARD",
                "US_SSN",
            ],
            on_fail="fix",  # Changed from "exception" to "fix" to allow the chain to continue
        )
    )

    # Create a Sifaka validator that uses the Guard instance
    detect_pii_validator = guardrails_validator(
        guard=pii_guard,
        api_key=guardrails_api_key,
        name="detect_pii_validator",
        description="Validates that text does not contain personally identifiable information",
    )

    # Create profanity_free validator
    # Create a Guard instance with the ProfanityFree validator
    profanity_guard = Guard().use(ProfanityFree(on_fail="exception"))

    # Create a Sifaka validator that uses the Guard instance
    profanity_free_validator = guardrails_validator(
        guard=profanity_guard,
        api_key=guardrails_api_key,
        name="profanity_free_validator",
        description="Validates that text does not contain profanity",
    )

    # Create N-Critics critic with custom system prompt
    logger.info("Creating N-Critics critic...")

    # Custom system prompt that emphasizes providing feedback, not rewriting
    custom_system_prompt = """
    You are an expert language model that uses an ensemble of specialized critics
    to provide comprehensive feedback on text. You follow the N-Critics approach
    to provide structured guidance.

    Your job is to analyze the text and provide specific, actionable feedback on how
    to improve it. Focus on:

    1. Factual accuracy and coherence
    2. Style, tone, and readability
    3. Structure and organization
    4. Engagement and clarity
    5. Potential issues with bias, sensitivity, or appropriateness

    DO NOT rewrite the text yourself. Instead, provide clear feedback that would help
    the author improve the text while preserving the core narrative elements.

    Your feedback should be specific, constructive, and actionable.
    """

    n_critics_critic = create_n_critics_critic(
        model=openai_model,
        system_prompt=custom_system_prompt,
        num_critics=5,
        max_refinement_iterations=1,  # Only one iteration for feedback
        temperature=0.7,
        name="feedback_critic",
    )

    # Define the prompt for generating a fictional story
    prompt = """
    Write a fictional short story about a character who receives a mysterious message.
    The story should be creative and engaging, with a clear beginning, middle, and end.
    Include some character details and a setting.
    """

    # Create the Chain with feedback loop
    logger.info("Creating Chain with feedback loop...")
    chain = (
        Chain(max_improvement_iterations=3)
        .with_model(anthropic_model)  # Use Anthropic model now that we've fixed it
        .with_prompt(prompt)
        .validate_with(length(min_words=100, max_words=1000))
        .validate_with(detect_pii_validator)
        .validate_with(profanity_free_validator)
        .improve_with(n_critics_critic)
        .with_options(apply_improvers_on_validation_failure=True)  # Enable feedback loop
    )

    # Run the chain
    logger.info("Running Chain...")
    result = chain.run()

    # Print the result
    print_separator("CHAIN EXECUTION RESULT")
    print(f"Chain execution passed: {result.passed}")
    print(f"Execution time: {result.execution_time_ms / 1000:.2f} seconds")
    print(f"Number of validation results: {len(result.validation_results)}")
    print(f"Number of improvement results: {len(result.improvement_results)}")

    # Print the original and final text
    if result.improvement_results:
        original_text = result.improvement_results[0]._original_text
        final_text = result.text
        print_text_comparison(original_text, final_text)
    else:
        print_separator("FINAL TEXT")
        print(result.text)

    # Print the improvement process
    print_separator("IMPROVEMENT PROCESS")
    for record in text_handler.get_records():
        print(record)

    # Print detailed improvement results
    if result.improvement_results:
        print_separator("DETAILED IMPROVEMENT RESULTS")
        for i, improvement in enumerate(result.improvement_results):
            print(f"Improvement {i+1}:")
            print(f"Message: {improvement.message}")
            print(f"Changes made: {improvement.changes_made}")
            if hasattr(improvement, "details") and improvement.details:
                print("Details:")
                for key, value in improvement.details.items():
                    if key == "validation_feedback" or key == "critic_feedback":
                        print(f"  {key}:")
                        if isinstance(value, dict):
                            for k, v in value.items():
                                if k == "issues" or k == "suggestions":
                                    print(f"    {k}:")
                                    for item in v:
                                        print(f"      - {item}")
                                else:
                                    print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
            print()


if __name__ == "__main__":
    main()
