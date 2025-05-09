"""
Modern Chain Example

This example demonstrates how to use the current chain architecture with different
components and strategies according to the latest structure recommendations.

The example includes:
1. Simple chain with OpenAI provider
2. Backoff chain with Anthropic provider
3. Custom chain with specialized components
4. Chain with custom prompt manager
5. Advanced chain with multiple rules and custom critic

Requirements:
    pip install "sifaka[openai,anthropic]"

Note: This example requires API keys set as environment variables:
    - OPENAI_API_KEY for OpenAI models
    - ANTHROPIC_API_KEY for Anthropic models
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file (containing API keys)
load_dotenv()

# Import Sifaka components using the correct imports
from sifaka.chain import (
    ChainOrchestrator,
    create_simple_chain,
    create_backoff_chain,
    PromptManager,
    ValidationManager,
    ResultFormatter,
    SimpleRetryStrategy,
    BackoffRetryStrategy,
)
from sifaka.critics.implementations import create_prompt_critic
from sifaka.models import create_openai_provider, create_anthropic_provider
from sifaka.rules.formatting.length import create_length_rule

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_simple_chain_with_openai():
    """Run a simple chain with OpenAI provider and basic rules."""
    logger.info("Running simple chain with OpenAI example...")

    # Create a model provider using the factory function
    model = create_openai_provider(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create rules using factory functions
    rules = [
        create_length_rule(min_chars=50, max_chars=1000),
    ]

    # Create a critic using the factory function
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


def run_backoff_chain_with_anthropic():
    """Run a chain with Anthropic provider and exponential backoff retry strategy."""
    logger.info("Running backoff chain with Anthropic example...")

    # Create a model provider using the factory function
    model = create_anthropic_provider(
        model_name="claude-3-opus-20240229",
        temperature=0.7,
        max_tokens=500,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Create rules using factory functions
    rules = [
        create_length_rule(min_chars=50, max_chars=1000),
    ]

    # Create a critic using the factory function
    critic = create_prompt_critic(
        llm_provider=model,
        name="editor_critic",
        description="Expert editor that improves text",
        system_prompt="You are an expert editor that improves text while maintaining its original meaning.",
        temperature=0.7,
        max_tokens=500,
        min_confidence=0.7,
    )

    # Create a chain using the factory function with backoff strategy
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

    # Create a model provider using the factory function
    model = create_openai_provider(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create rules using factory functions
    rules = [
        create_length_rule(min_chars=50, max_chars=1000),
    ]

    # Create a critic using the factory function
    critic = create_prompt_critic(
        llm_provider=model,
        name="editor_critic",
        description="Expert editor that improves text",
        system_prompt="You are an expert editor that improves text while maintaining its original meaning.",
        temperature=0.7,
        max_tokens=500,
        min_confidence=0.7,
    )

    # Create custom components
    validation_manager = ValidationManager(rules)
    prompt_manager = PromptManager()
    retry_strategy = BackoffRetryStrategy(
        max_attempts=5,
        initial_backoff=1.0,
        backoff_factor=2.0,
        max_backoff=60.0,
    )
    result_formatter = ResultFormatter()

    # Create a chain orchestrator with custom components
    chain = ChainOrchestrator(
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=5,
    )

    # Replace default retry strategy with custom strategy
    chain._core._retry_strategy = retry_strategy

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

    # Create a model provider using the factory function
    model = create_openai_provider(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create rules using factory functions
    rules = [
        create_length_rule(min_chars=50, max_chars=1000),
    ]

    # Create a critic using the factory function
    critic = create_prompt_critic(
        llm_provider=model,
        name="editor_critic",
        description="Expert editor that improves text",
        system_prompt="You are an expert editor that improves text while maintaining its original meaning.",
        temperature=0.7,
        max_tokens=500,
        min_confidence=0.7,
    )

    # Create custom components
    validation_manager = ValidationManager(rules)
    prompt_manager = CustomPromptManager()
    retry_strategy = SimpleRetryStrategy(max_attempts=3)
    result_formatter = ResultFormatter()

    # Create a chain orchestrator with custom components
    chain = ChainOrchestrator(
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=3,
    )

    # Replace default prompt manager with custom prompt manager
    chain._core._prompt_manager = prompt_manager

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


def run_advanced_chain_example():
    """Run an advanced chain with multiple rules and a custom critic."""
    logger.info("Running advanced chain with multiple rules and custom critic example...")

    # Create a model provider using the factory function
    model = create_openai_provider(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create a critic using the factory function with a custom system prompt
    critic = create_prompt_critic(
        llm_provider=model,
        name="detailed_critic",
        description="Provides detailed feedback on text quality",
        system_prompt="""You are an expert editor that provides detailed feedback on text quality.

        Analyze the text and provide feedback in the following format:
        - Overall quality (1-10)
        - Strengths
        - Areas for improvement
        - Specific suggestions

        Be thorough and specific in your analysis.
        """,
        temperature=0.7,
        max_tokens=500,
        min_confidence=0.7,
    )

    # Test the critic directly
    logger.info("Testing critic directly...")
    test_text = "Clean code is important for software development."
    critique_result = critic.critique(test_text)
    logger.info(f"Critique result: {critique_result}")

    # Create a simple chain with a length rule
    logger.info("Creating a chain with a length rule and critic...")
    chain = create_simple_chain(
        model=model,
        rules=[create_length_rule(min_chars=50, max_chars=1000)],
        critic=critic,
        max_attempts=3,
    )

    # Run the chain with a simple prompt
    prompt = "Write a short paragraph about clean code."

    try:
        result = chain.run(prompt)
        logger.info(f"Output: {result.output}")
        logger.info(f"All rules passed: {all(r.passed for r in result.rule_results)}")

        # Print the critique details if available
        if hasattr(result, "critique_details") and result.critique_details:
            logger.info(f"Critique details: {result.critique_details}")
        else:
            logger.info("No critique details available")

    except ValueError as e:
        logger.error(f"Validation failed: {e}")

    # Create a chain with a backoff retry strategy
    logger.info("Creating a chain with a backoff retry strategy...")

    from sifaka.chain.strategies.retry import BackoffRetryStrategy

    # Create a backoff retry strategy
    retry_strategy = BackoffRetryStrategy(
        max_attempts=3,
        initial_backoff=1.0,
        backoff_factor=2.0,
        max_backoff=10.0,
    )

    # Create a chain with the backoff retry strategy
    chain = create_simple_chain(
        model=model,
        rules=[create_length_rule(min_chars=50, max_chars=1000)],
        critic=critic,
        max_attempts=3,
    )

    # Replace the default retry strategy with our custom one
    chain._core._retry_strategy = retry_strategy

    # Run the chain with a simple prompt
    prompt = "Write a short paragraph about clean code that includes specific examples."

    try:
        result = chain.run(prompt)
        logger.info(f"Output: {result.output}")
        logger.info(f"All rules passed: {all(r.passed for r in result.rule_results)}")

        # Print the critique details if available
        if hasattr(result, "critique_details") and result.critique_details:
            logger.info(f"Critique details: {result.critique_details}")
        else:
            logger.info("No critique details available")

    except ValueError as e:
        logger.error(f"Validation failed: {e}")


if __name__ == "__main__":
    # Check if API keys are set
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error(
            "OPENAI_API_KEY environment variable not set. OpenAI examples will be skipped."
        )
    else:
        # Run OpenAI examples
        run_simple_chain_with_openai()
        run_custom_chain()
        run_custom_prompt_manager()
        run_advanced_chain_example()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY environment variable not set. Anthropic examples will be skipped."
        )
    else:
        # Run Anthropic examples
        run_backoff_chain_with_anthropic()
