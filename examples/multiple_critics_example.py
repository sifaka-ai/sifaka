"""
An example of using Sifaka with multiple critics.

This example demonstrates how to create a chain with multiple critics for comprehensive
text improvement, using the ReflexionCritic and N-Critics approach.
"""

import logging
import os
from dotenv import load_dotenv
from sifaka import Chain
from sifaka.validators import length, prohibited_content
from sifaka.validators.guardrails import guardrails_validator
from sifaka.critics.reflexion import create_reflexion_critic
from sifaka.critics.n_critics import create_n_critics_critic
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.models.openai import OpenAIModel

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run an example with multiple critics and Guardrails validators."""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Get API keys from environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    guardrails_api_key = os.environ.get("GUARDRAILS_API_KEY")

    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    if not guardrails_api_key:
        logger.warning(
            "GUARDRAILS_API_KEY environment variable not set. Some validators may not work properly."
        )

    # Create a model
    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

    # Create critics
    reflexion_critic = create_reflexion_critic(
        model=model,
        reflection_rounds=2,
    )

    n_critics_critic = create_n_critics_critic(
        model=model,
        num_critics=3,
        max_refinement_iterations=1,
    )

    constitutional_critic = create_constitutional_critic(
        model=model,
        principles=[
            "Explanations should be clear and accessible to non-experts.",
            "Content should be factually accurate and well-supported.",
            "Examples should be relevant and help illustrate concepts.",
        ],
    )

    # Create Guardrails validators
    regex_match_validator = guardrails_validator(
        validators=["regex_match"],
        validator_args={
            "regex_match": {
                "regex": r"machine\s+learning",  # Ensure "machine learning" is mentioned
                "match_type": "search",
                "on_fail": "exception",
            }
        },
        api_key=guardrails_api_key,
        name="regex_match_validator",
    )

    profanity_free_validator = guardrails_validator(
        validators=["profanity_free"],
        validator_args={"profanity_free": {"on_fail": "exception"}},
        api_key=guardrails_api_key,
        name="profanity_free_validator",
    )

    # Create a chain with multiple critics and Guardrails validators
    chain = (
        Chain()
        .with_model(model)
        .with_prompt("Explain how machine learning works to someone with no technical background.")
        .validate_with(length(min_words=100, max_words=900))
        .validate_with(regex_match_validator)
        .validate_with(profanity_free_validator)
        .improve_with(constitutional_critic)  # First apply constitutional principles
        .improve_with(n_critics_critic)  # Then use n-critics for comprehensive feedback
        .improve_with(reflexion_critic)  # Finally reflect on improvements
    )

    # Run the chain
    logger.info("Running chain with multiple critics...")
    result = chain.run()

    # Check the result
    if result.passed:
        logger.info("Chain execution succeeded!")
        logger.info(f"Generated text:\n{result.text}")
    else:
        logger.error("Chain execution failed validation")
        for i, validation_result in enumerate(result.validation_results):
            if not validation_result.passed:
                logger.error(f"Validation {i+1} failed: {validation_result.message}")


if __name__ == "__main__":
    main()
