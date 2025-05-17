"""
Example of using retrieval-enhanced critics.

This example demonstrates how to use retrieval-enhanced versions of various critics
to improve text with retrieval-based information.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from sifaka.models import get_model
from sifaka.critics import (
    create_retrieval_enhanced_constitutional_critic,
    create_retrieval_enhanced_n_critics_critic,
    create_retrieval_enhanced_reflexion_critic,
    create_retrieval_enhanced_self_refine_critic,
    create_retrieval_enhanced_prompt_critic,
)
from sifaka.retrievers import create_retrieval_augmenter

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def dummy_retriever(query: str) -> List[str]:
    """
    Dummy retriever for demonstration purposes.
    In a real application, you would use a proper retriever.

    Args:
        query: Query to retrieve documents for

    Returns:
        A list of retrieved documents
    """
    # This is just a dummy implementation
    # In a real application, you would use a proper retriever
    logger.info(f"Retrieving documents for query: {query}")

    # Return some dummy documents based on the query
    if "history" in query.lower():
        return [
            "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
            "artificial beings endowed with intelligence or consciousness by master craftsmen.",
            "The field of AI research was founded at a workshop held on the campus of Dartmouth College "
            "during the summer of 1956.",
        ]
    elif "application" in query.lower():
        return [
            "AI applications include advanced web search engines, recommendation systems, "
            "understanding human speech, self-driving cars, automated decision-making and "
            "competing at the highest level in strategic game systems.",
            "Many tools are used in AI, including versions of search and mathematical optimization, "
            "artificial neural networks, and methods based on statistics, probability and economics.",
        ]
    elif "ethics" in query.lower():
        return [
            "AI ethics is a set of values, principles, and techniques that employ widely accepted standards "
            "of right and wrong to guide moral conduct in the development and use of AI technologies.",
            "The ethics of artificial intelligence is the branch of the ethics of technology specific to "
            "artificially intelligent systems. It is sometimes divided into a concern with the moral behavior "
            "of humans as they design, make, use and treat artificially intelligent systems, and a concern "
            "with the behavior of machines, in machine ethics.",
        ]
    else:
        return [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, "
            "as opposed to natural intelligence displayed by animals including humans.",
            "Leading AI textbooks define the field as the study of 'intelligent agents': "
            "any system that perceives its environment and takes actions that maximize "
            "its chance of achieving its goals.",
        ]


def example_retrieval_enhanced_constitutional_critic():
    """Example of using a retrieval-enhanced Constitutional critic."""
    logger.info("Example of using a retrieval-enhanced Constitutional critic")

    # Create a model
    model = get_model("gpt-4")

    # Create a retrieval augmenter
    retrieval_augmenter = create_retrieval_augmenter(
        retriever=dummy_retriever, model=model, max_passages=3, max_queries=3
    )

    # Define constitutional principles
    principles = [
        "Ensure factual accuracy by incorporating relevant information from retrieved sources.",
        "Provide comprehensive context by including information from retrieved passages when relevant.",
        "Maintain clarity and coherence in the text while integrating new information.",
        "Cite or reference external information appropriately.",
    ]

    # Create a retrieval-enhanced Constitutional critic
    critic = create_retrieval_enhanced_constitutional_critic(
        model=model, retrieval_augmenter=retrieval_augmenter, principles=principles, temperature=0.7
    )

    # Use the critic to improve text
    text = "Artificial intelligence is a technology that can solve many problems."
    improved_text = critic.improve(text)

    logger.info(f"Original text: {text}")
    logger.info(f"Improved text: {improved_text}")


def example_retrieval_enhanced_n_critics_critic():
    """Example of using a retrieval-enhanced N-Critics critic."""
    logger.info("Example of using a retrieval-enhanced N-Critics critic")

    # Create a model
    model = get_model("gpt-4")

    # Create a retrieval augmenter
    retrieval_augmenter = create_retrieval_augmenter(
        retriever=dummy_retriever, model=model, max_passages=3, max_queries=3
    )

    # Create a retrieval-enhanced N-Critics critic
    critic = create_retrieval_enhanced_n_critics_critic(
        model=model,
        retrieval_augmenter=retrieval_augmenter,
        num_critics=3,
        max_refinement_iterations=2,
        temperature=0.7,
    )

    # Use the critic to improve text
    text = "Artificial intelligence is a technology that can solve many problems."
    improved_text = critic.improve(text)

    logger.info(f"Original text: {text}")
    logger.info(f"Improved text: {improved_text}")


def example_retrieval_enhanced_reflexion_critic():
    """Example of using a retrieval-enhanced Reflexion critic."""
    logger.info("Example of using a retrieval-enhanced Reflexion critic")

    # Create a model
    model = get_model("gpt-4")

    # Create a retrieval augmenter
    retrieval_augmenter = create_retrieval_augmenter(
        retriever=dummy_retriever, model=model, max_passages=3, max_queries=3
    )

    # Create a retrieval-enhanced Reflexion critic
    critic = create_retrieval_enhanced_reflexion_critic(
        model=model, retrieval_augmenter=retrieval_augmenter, reflection_rounds=2, temperature=0.7
    )

    # Use the critic to improve text
    text = "Artificial intelligence is a technology that can solve many problems."
    improved_text = critic.improve(text)

    logger.info(f"Original text: {text}")
    logger.info(f"Improved text: {improved_text}")


def example_retrieval_enhanced_self_refine_critic():
    """Example of using a retrieval-enhanced Self-Refine critic."""
    logger.info("Example of using a retrieval-enhanced Self-Refine critic")

    # Create a model
    model = get_model("gpt-4")

    # Create a retrieval augmenter
    retrieval_augmenter = create_retrieval_augmenter(
        retriever=dummy_retriever, model=model, max_passages=3, max_queries=3
    )

    # Create a retrieval-enhanced Self-Refine critic
    critic = create_retrieval_enhanced_self_refine_critic(
        model=model, retrieval_augmenter=retrieval_augmenter, max_iterations=3, temperature=0.7
    )

    # Use the critic to improve text
    text = "Artificial intelligence is a technology that can solve many problems."
    improved_text = critic.improve(text)

    logger.info(f"Original text: {text}")
    logger.info(f"Improved text: {improved_text}")


def example_retrieval_enhanced_prompt_critic():
    """Example of using a retrieval-enhanced Prompt critic."""
    logger.info("Example of using a retrieval-enhanced Prompt critic")

    # Create a model
    model = get_model("gpt-4")

    # Create a retrieval augmenter
    retrieval_augmenter = create_retrieval_augmenter(
        retriever=dummy_retriever, model=model, max_passages=3, max_queries=3
    )

    # Define custom prompts that incorporate retrieved information
    critique_prompt = """
    Please analyze the following text and provide a detailed critique:

    ```
    {text}
    ```

    Retrieved information:
    {formatted_passages}

    Focus on:
    1. Factual accuracy compared to the retrieved information
    2. Completeness of the information
    3. Clarity and coherence
    4. Style and tone

    Format your response as JSON with the following fields:
    - "issues": a list of specific issues
    - "suggestions": a list of specific suggestions for improvement
    """

    improve_prompt = """
    Please improve the following text based on the critique and retrieved information:

    Original text:
    ```
    {text}
    ```

    Critique:
    {critique}

    Retrieved information:
    {formatted_passages}

    Instructions:
    1. Address all issues mentioned in the critique
    2. Incorporate relevant information from the retrieved passages
    3. Maintain the original style and tone
    4. Ensure factual accuracy and logical coherence

    Improved text:
    """

    # Create a retrieval-enhanced Prompt critic
    critic = create_retrieval_enhanced_prompt_critic(
        model=model,
        retrieval_augmenter=retrieval_augmenter,
        critique_prompt=critique_prompt,
        improve_prompt=improve_prompt,
        temperature=0.7,
    )

    # Use the critic to improve text
    text = "Artificial intelligence is a technology that can solve many problems."
    improved_text = critic.improve(text)

    logger.info(f"Original text: {text}")
    logger.info(f"Improved text: {improved_text}")


def compare_critics():
    """Compare different retrieval-enhanced critics."""
    logger.info("Comparing different retrieval-enhanced critics")

    # Create a model
    model = get_model("gpt-4")

    # Create a retrieval augmenter
    retrieval_augmenter = create_retrieval_augmenter(
        retriever=dummy_retriever, model=model, max_passages=3, max_queries=3
    )

    # Create critics
    constitutional_critic = create_retrieval_enhanced_constitutional_critic(
        model=model, retrieval_augmenter=retrieval_augmenter, temperature=0.7
    )

    n_critics_critic = create_retrieval_enhanced_n_critics_critic(
        model=model,
        retrieval_augmenter=retrieval_augmenter,
        num_critics=3,
        max_refinement_iterations=2,
        temperature=0.7,
    )

    reflexion_critic = create_retrieval_enhanced_reflexion_critic(
        model=model, retrieval_augmenter=retrieval_augmenter, reflection_rounds=2, temperature=0.7
    )

    self_refine_critic = create_retrieval_enhanced_self_refine_critic(
        model=model, retrieval_augmenter=retrieval_augmenter, max_iterations=3, temperature=0.7
    )

    prompt_critic = create_retrieval_enhanced_prompt_critic(
        model=model, retrieval_augmenter=retrieval_augmenter, temperature=0.7
    )

    # Text to improve
    text = "Artificial intelligence is a technology that can solve many problems."

    # Improve text with each critic
    constitutional_improved = constitutional_critic.improve(text)
    n_critics_improved = n_critics_critic.improve(text)
    reflexion_improved = reflexion_critic.improve(text)
    self_refine_improved = self_refine_critic.improve(text)
    prompt_improved = prompt_critic.improve(text)

    # Print results
    logger.info(f"Original text: {text}")
    logger.info(f"Constitutional critic: {constitutional_improved}")
    logger.info(f"N-Critics critic: {n_critics_improved}")
    logger.info(f"Reflexion critic: {reflexion_improved}")
    logger.info(f"Self-Refine critic: {self_refine_improved}")
    logger.info(f"Prompt critic: {prompt_improved}")


if __name__ == "__main__":
    # Run the examples
    example_retrieval_enhanced_constitutional_critic()
    print("\n" + "-" * 80 + "\n")
    example_retrieval_enhanced_n_critics_critic()
    print("\n" + "-" * 80 + "\n")
    example_retrieval_enhanced_reflexion_critic()
    print("\n" + "-" * 80 + "\n")
    example_retrieval_enhanced_self_refine_critic()
    print("\n" + "-" * 80 + "\n")
    example_retrieval_enhanced_prompt_critic()
    print("\n" + "-" * 80 + "\n")
    compare_critics()
