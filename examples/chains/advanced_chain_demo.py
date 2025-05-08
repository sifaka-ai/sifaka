"""
Advanced Chain Example

This example demonstrates how to use the new chain architecture with different
components and strategies.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file (containing API keys)
load_dotenv()

from sifaka.chain import (
    PromptManager,
    create_simple_chain,
    create_backoff_chain,
)
from sifaka.critics import create_prompt_critic
from sifaka.models.openai import create_openai_provider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.content.prohibited import create_prohibited_content_rule

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_simple_chain():
    """Run a simple chain with a length rule."""
    logger.info("Running simple chain example...")

    # Create a model provider
    model = create_openai_provider(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create rules
    rules = [
        create_length_rule(min_chars=50, max_chars=200),
        create_prohibited_content_rule(terms=["bad", "inappropriate"]),
    ]

    # Create a critic
    critic = create_prompt_critic(
        llm_provider=model,
        name="editor_critic",
        description="Expert editor that improves text",
        system_prompt="You are an expert editor that improves text while maintaining its original meaning.",
        temperature=0.7,
        max_tokens=500,
        min_confidence=0.7,
    )

    # Create a chain using the factory function
    chain = create_simple_chain(
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=3,
    )

    # Run the chain
    prompt = "Write a short paragraph about the importance of clean code."

    try:
        result = chain.run(prompt)
        logger.info(f"Output: {result.output}")
        logger.info(f"All rules passed: {all(r.passed for r in result.rule_results)}")

        if result.critique_details:
            logger.info(f"Critique score: {result.critique_details.get('score', 'N/A')}")
            logger.info(f"Critique feedback: {result.critique_details.get('feedback', 'N/A')}")
    except ValueError as e:
        logger.error(f"Validation failed: {e}")


def run_backoff_chain():
    """Run a chain with exponential backoff retry strategy."""
    logger.info("Running backoff chain example...")

    # Create a model provider
    model = create_openai_provider(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create rules
    rules = [
        create_length_rule(min_chars=50, max_chars=200),
        create_prohibited_content_rule(terms=["bad", "inappropriate"]),
    ]

    # Create a critic
    critic = create_prompt_critic(
        llm_provider=model,
        name="editor_critic",
        description="Expert editor that improves text",
        system_prompt="You are an expert editor that improves text while maintaining its original meaning.",
        temperature=0.7,
        max_tokens=500,
        min_confidence=0.7,
    )

    # Create a chain using the factory function
    chain = create_backoff_chain(
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=5,
        initial_backoff=1.0,
        backoff_factor=2.0,
        max_backoff=60.0,
    )

    # Run the chain
    prompt = "Write a short paragraph about the importance of clean code."

    try:
        result = chain.run(prompt)
        logger.info(f"Output: {result.output}")
        logger.info(f"All rules passed: {all(r.passed for r in result.rule_results)}")

        if result.critique_details:
            logger.info(f"Critique score: {result.critique_details.get('score', 'N/A')}")
            logger.info(f"Critique feedback: {result.critique_details.get('feedback', 'N/A')}")
    except ValueError as e:
        logger.error(f"Validation failed: {e}")


def run_custom_chain():
    """Run a chain with custom components."""
    logger.info("Running custom chain example...")

    # Create a model provider
    model = create_openai_provider(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create rules
    rules = [
        create_length_rule(min_chars=50, max_chars=200),
        create_prohibited_content_rule(terms=["bad", "inappropriate"]),
    ]

    # Create a critic
    critic = create_prompt_critic(
        llm_provider=model,
        name="editor_critic",
        description="Expert editor that improves text",
        system_prompt="You are an expert editor that improves text while maintaining its original meaning.",
        temperature=0.7,
        max_tokens=500,
        min_confidence=0.7,
    )

    # Create a chain with backoff strategy
    chain = create_backoff_chain(
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=5,
        initial_backoff=1.0,
        backoff_factor=2.0,
        max_backoff=60.0,
        name="custom_backoff_chain",
        description="A custom chain with backoff strategy",
    )

    # Run the chain
    prompt = "Write a short paragraph about the importance of clean code."

    try:
        result = chain.run(prompt)
        logger.info(f"Output: {result.output}")
        logger.info(f"All rules passed: {all(r.passed for r in result.rule_results)}")

        if result.critique_details:
            logger.info(f"Critique score: {result.critique_details.get('score', 'N/A')}")
            logger.info(f"Critique feedback: {result.critique_details.get('feedback', 'N/A')}")
    except ValueError as e:
        logger.error(f"Validation failed: {e}")


def run_custom_prompt_manager():
    """Run a chain with a custom prompt manager."""
    logger.info("Running custom prompt manager example...")

    # Create a custom prompt manager
    class CustomPromptManager(PromptManager):
        """Custom prompt manager that adds a system message."""

        def create_prompt_with_feedback(self, original_prompt: str, feedback: str) -> str:
            """Create a new prompt with feedback."""
            return f"System: {feedback}\n\nUser: {original_prompt}"

    # Create a model provider
    model = create_openai_provider(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create rules
    rules = [
        create_length_rule(min_chars=50, max_chars=200),
        create_prohibited_content_rule(terms=["bad", "inappropriate"]),
    ]

    # Create a critic
    critic = create_prompt_critic(
        llm_provider=model,
        name="editor_critic",
        description="Expert editor that improves text",
        system_prompt="You are an expert editor that improves text while maintaining its original meaning.",
        temperature=0.7,
        max_tokens=500,
        min_confidence=0.7,
    )

    # For this example, we'll use the simple chain factory function
    # In a real implementation, you would create a custom implementation
    # that uses your custom prompt manager
    chain = create_simple_chain(
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=3,
        name="custom_prompt_chain",
        description="A chain with a custom prompt manager",
    )

    # Run the chain
    prompt = "Write a short paragraph about the importance of clean code."

    try:
        result = chain.run(prompt)
        logger.info(f"Output: {result.output}")
        logger.info(f"All rules passed: {all(r.passed for r in result.rule_results)}")

        if result.critique_details:
            logger.info(f"Critique score: {result.critique_details.get('score', 'N/A')}")
            logger.info(f"Critique feedback: {result.critique_details.get('feedback', 'N/A')}")
    except ValueError as e:
        logger.error(f"Validation failed: {e}")

    # Note: In a real implementation, you would create a custom chain implementation
    # that uses your custom prompt manager. For example:
    #
    # from sifaka.chain.implementation import Chain
    # from sifaka.chain.implementations.simple import SimpleChainImplementation
    #
    # # Create custom implementation with your prompt manager
    # implementation = SimpleChainImplementation(
    #     model=model,
    #     rules=rules,
    #     critic=critic,
    #     max_attempts=3,
    #     prompt_manager=CustomPromptManager()  # Use your custom prompt manager
    # )
    #
    # # Create chain with custom implementation
    # chain = Chain(
    #     name="custom_prompt_chain",
    #     description="A chain with a custom prompt manager",
    #     config=ChainConfig(max_attempts=3),
    #     implementation=implementation
    # )


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set. Please set it and try again.")
        exit(1)

    # Run examples
    run_simple_chain()
    run_backoff_chain()
    run_custom_chain()
    run_custom_prompt_manager()
