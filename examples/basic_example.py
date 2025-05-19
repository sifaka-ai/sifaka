"""
A basic example of using Sifaka with validators and critics.

This example demonstrates how to create a simple chain with a model, validators, and critics.
"""

import logging
import os

from dotenv import load_dotenv

from sifaka import Chain
from sifaka.models.openai import OpenAIModel
from sifaka.validators import length, prohibited_content

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run a basic example of Sifaka."""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    # Create a model
    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

    # Create a chain with validators only
    chain = (
        Chain()
        .with_model(model)
        .with_prompt("Write a short story about a robot learning to feel emotions.")
        .validate_with(length(min_words=50, max_words=500))
        .validate_with(prohibited_content(prohibited=["violent", "harmful"]))
    )

    # Run the chain
    logger.info("Running chain...")
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
