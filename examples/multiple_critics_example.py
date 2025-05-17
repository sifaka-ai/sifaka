"""
An example of using Sifaka with multiple critics.

This example demonstrates how to create a chain with multiple critics for comprehensive
text improvement.
"""

import logging
import os
from dotenv import load_dotenv
from sifaka import Chain
from sifaka.validators import length, prohibited_content
from sifaka.critics.reflexion import create_reflexion_critic
from sifaka.critics.self_refine import create_self_refine_critic
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.models.openai import OpenAIModel

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run an example with multiple critics."""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    # Create a model
    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

    # Create critics
    reflexion_critic = create_reflexion_critic(
        model=model,
        reflection_rounds=1,
    )

    self_refine_critic = create_self_refine_critic(
        model=model,
        max_refinement_iterations=2,
    )

    constitutional_critic = create_constitutional_critic(
        model=model,
        principles=[
            "Explanations should be clear and accessible to non-experts.",
            "Content should be factually accurate and well-supported.",
            "Examples should be relevant and help illustrate concepts.",
        ],
    )

    # Create a chain with multiple critics
    chain = (
        Chain()
        .with_model(model)
        .with_prompt("Explain how machine learning works to someone with no technical background.")
        .validate_with(length(min_words=100, max_words=300))
        .validate_with(prohibited_content(prohibited=["complex", "difficult"]))
        .improve_with(constitutional_critic)  # First apply constitutional principles
        .improve_with(self_refine_critic)  # Then refine the text
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
