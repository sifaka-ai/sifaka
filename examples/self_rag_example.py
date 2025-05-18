"""
An example of using Sifaka with the Self-RAG critic.

This example demonstrates how to use the Self-RAG critic with a retriever
for knowledge-enhanced text generation.
"""

import logging
import os
from dotenv import load_dotenv
from sifaka import Chain
from sifaka.validators import length
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.models.openai import OpenAIModel
from sifaka.retrievers import SimpleRetriever

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Sample documents for retrieval
DOCUMENTS = [
    "The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. "
    "As the largest optical telescope in space, its high infrared resolution and sensitivity allow it to view objects "
    "too old, distant, or faint for the Hubble Space Telescope.",
    "The JWST was launched on 25 December 2021 on an Ariane 5 rocket from Kourou, French Guiana, and arrived at the "
    "Sun–Earth L2 Lagrange point in January 2022. The telescope is named after James E. Webb, who was the administrator "
    "of NASA from 1961 to 1968 during the Mercury, Gemini, and Apollo programs.",
    "The JWST is the successor to the Hubble Space Telescope and is a collaboration between NASA, the European Space "
    "Agency, and the Canadian Space Agency. The telescope features a segmented 6.5-meter-diameter mirror, and operates "
    "in the infrared spectrum, allowing it to observe objects that are too distant or too cool for Hubble.",
    "The primary mirror of the JWST consists of 18 hexagonal mirror segments made of gold-plated beryllium. The mirror "
    "has a collecting area of 25.4 m², compared to Hubble's 4.5 m², giving JWST a significant advantage in terms of "
    "light-gathering capability.",
]


def main():
    """Run an example with the Self-RAG critic."""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    # Create a model
    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

    # Create a simple retriever with our documents
    retriever = SimpleRetriever(documents=DOCUMENTS)

    # Create a Self-RAG critic
    self_rag_critic = create_self_rag_critic(
        model=model,
        retriever=retriever,
        reflection_enabled=True,
        max_passages=2,
    )

    # Create a chain with the Self-RAG critic
    chain = (
        Chain()
        .with_model(model)
        .with_prompt("Explain the key features and capabilities of the James Webb Space Telescope.")
        .validate_with(length(min_words=50, max_words=500))
        .improve_with(self_rag_critic)
    )

    # Run the chain
    logger.info("Running chain with Self-RAG critic...")
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
