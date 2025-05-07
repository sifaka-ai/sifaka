"""
Critic Comparison Example

This example compares the performance of different critics in Sifaka:
- PromptCritic
- ReflexionCritic
- SelfRefineCritic
- SelfRAGCritic
- ConstitutionalCritic
- LACCritic

The example measures:
1. Time spent (execution time)
2. Number of revisions/iterations
3. Quality of output (using a simple heuristic)
4. Output length

Requirements:
    pip install "sifaka[openai,anthropic]"

Note: This example requires API keys set as environment variables:
    - OPENAI_API_KEY for OpenAI models
    - ANTHROPIC_API_KEY for Anthropic models
"""

import os
import time
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Import Sifaka components
from sifaka.models.openai import create_openai_provider
from sifaka.models.anthropic import create_anthropic_provider
from sifaka.critics.factories import create_prompt_critic, create_reflexion_critic
from sifaka.critics.self_refine import create_self_refine_critic
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.critics.lac import create_lac_critic
from sifaka.retrieval import SimpleRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CriticPerformance:
    """Class to store performance metrics for a critic."""

    name: str
    model_provider: str
    time_spent: float
    revisions: int
    output_quality: float
    output_length: int
    output_text: str


def evaluate_quality(text: str) -> float:
    """
    Evaluate the quality of text using simple heuristics.

    Returns a score between 0.0 and 1.0.
    """
    # Simple heuristics for quality evaluation
    # In a real application, you might use a more sophisticated approach
    # or another LLM to evaluate quality

    # Check length (longer texts might be more detailed)
    length_score = min(len(text) / 1000, 1.0)

    # Check for structure (paragraphs, bullet points)
    structure_markers = ["\n\n", "- ", "• ", "1. ", "2. ", "3. "]
    structure_score = min(sum(text.count(marker) for marker in structure_markers) / 10, 1.0)

    # Check for complexity (average word length)
    words = text.split()
    if words:
        avg_word_length = sum(len(word) for word in words) / len(words)
        complexity_score = min(avg_word_length / 8, 1.0)
    else:
        complexity_score = 0.0

    # Combine scores (weighted average)
    return 0.4 * length_score + 0.4 * structure_score + 0.2 * complexity_score


def test_prompt_critic(llm_provider: Any, input_text: str, task: str) -> Tuple[str, int]:
    """Test the PromptCritic and return the improved text and number of revisions."""
    # Note: 'task' parameter is included for consistent interface with other test functions
    critic = create_prompt_critic(
        llm_provider=llm_provider,
        name="test_prompt_critic",
        description="A critic for testing",
        system_prompt=f"You are an expert editor that improves text for this task: {task}",
        temperature=0.7,
        max_tokens=1000,
    )

    # PromptCritic doesn't track revisions internally, so we'll do it manually
    revisions = 1

    # First get a critique to use as feedback
    critique = critic.critique(input_text)
    feedback = critique.get("feedback", "Please improve this text to be more comprehensive.")

    # Then improve the text with the feedback
    improved_text = critic.improve(input_text, feedback)

    return improved_text, revisions


def test_reflexion_critic(llm_provider: Any, input_text: str, task: str) -> Tuple[str, int]:
    """Test the ReflexionCritic and return the improved text and number of revisions."""
    # Note: 'task' parameter is included for consistent interface with other test functions
    critic = create_reflexion_critic(
        llm_provider=llm_provider,
        name="test_reflexion_critic",
        description="A critic for testing with reflection",
        system_prompt=f"You are an expert editor that learns from past feedback. Task: {task}",
        temperature=0.7,
        max_tokens=1000,
        memory_buffer_size=3,
        reflection_depth=1,
    )

    # ReflexionCritic doesn't track revisions internally, so we'll do it manually
    revisions = 1

    # First get a critique to use as feedback
    critique = critic.critique(input_text)
    feedback = critique.get("feedback", "Please improve this text to be more comprehensive.")

    # Then improve the text with the feedback
    improved_text = critic.improve(input_text, feedback)

    return improved_text, revisions


def test_self_refine_critic(llm_provider: Any, input_text: str, task: str) -> Tuple[str, int]:
    """Test the SelfRefineCritic and return the improved text and number of revisions."""
    critic = create_self_refine_critic(
        llm_provider=llm_provider,
        name="test_self_refine_critic",
        description="A critic for testing with self-refinement",
        system_prompt="You are an expert editor that critiques and revises content.",
        temperature=0.7,
        max_tokens=1000,
        max_iterations=3,  # Allow up to 3 iterations
    )

    # SelfRefineCritic has internal iterations, but doesn't expose the count
    # We'll use the max_iterations as an approximation
    metadata = {"task": task}
    improved_text = critic.improve(input_text, metadata)

    # Estimate revisions based on difference between input and output
    # This is a rough approximation
    if improved_text == input_text:
        revisions = 0
    else:
        # Assume at least one revision if text changed
        revisions = 1

    return improved_text, revisions


def test_self_rag_critic(
    llm_provider: Any, input_text: str, task: str, documents: Dict[str, str]
) -> Tuple[str, int]:
    """Test the SelfRAGCritic and return the improved text and number of revisions."""
    # Create a simple retriever with the provided documents
    retriever = SimpleRetriever(documents=documents)

    critic = create_self_rag_critic(
        llm_provider=llm_provider,
        retriever=retriever,
        name="test_self_rag_critic",
        description="A critic for testing with self-RAG",
        system_prompt="You are an expert at retrieving and using information.",
        temperature=0.7,
        max_tokens=1000,
    )

    # SelfRAGCritic doesn't track revisions internally
    revisions = 1
    metadata = {"task": task}
    improved_text = critic.improve(input_text, metadata)

    return improved_text, revisions


def test_constitutional_critic(
    llm_provider: Any, input_text: str, task: str, principles: List[str]
) -> Tuple[str, int]:
    """Test the ConstitutionalCritic and return the improved text and number of revisions."""
    critic = create_constitutional_critic(
        llm_provider=llm_provider,
        principles=principles,
        name="test_constitutional_critic",
        description="A critic for testing with constitutional principles",
        system_prompt="You are an expert at evaluating content against principles.",
        temperature=0.7,
        max_tokens=1000,
    )

    # ConstitutionalCritic doesn't track revisions internally
    revisions = 1
    metadata = {"task": task}
    improved_text = critic.improve(input_text, metadata)

    return improved_text, revisions


def test_lac_critic(llm_provider: Any, input_text: str, task: str) -> Tuple[str, int]:
    """Test the LACCritic and return the improved text and number of revisions."""
    critic = create_lac_critic(
        llm_provider=llm_provider,
        name="test_lac_critic",
        description="A critic for testing with LAC approach",
        system_prompt="You are an expert at evaluating and improving text.",
        temperature=0.7,
        max_tokens=1000,
    )

    # LACCritic doesn't track revisions internally
    revisions = 1
    metadata = {"task": task}
    improved_text = critic.improve(input_text, metadata)

    return improved_text, revisions


def run_critic_comparison():
    """Run the comparison of different critics."""
    logger.info("Starting critic comparison")

    # Check for API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_api_key:
        logger.warning("OPENAI_API_KEY environment variable not set")
        openai_api_key = "demo-key"  # Use a demo key for testing

    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY environment variable not set")
        anthropic_api_key = "demo-key"  # Use a demo key for testing

    # Create model providers
    openai_provider = create_openai_provider(
        model_name="gpt-4",
        api_key=openai_api_key,
    )

    anthropic_provider = create_anthropic_provider(
        model_name="claude-3-opus-20240229",
        api_key=anthropic_api_key,
    )

    # Define a common task and input text
    task = "Write a comprehensive explanation of quantum computing for beginners."
    input_text = "Quantum computing uses qubits."

    # Define documents for SelfRAGCritic
    documents = {
        "quantum_basics": """
        Quantum computing is a type of computation that harnesses the collective properties of quantum states,
        such as superposition, interference, and entanglement, to perform calculations. The basic unit of
        quantum computing is the quantum bit or qubit, which can exist in multiple states simultaneously,
        unlike classical bits that can only be in one state (0 or 1) at a time.
        """,
        "quantum_applications": """
        Quantum computers are believed to be able to solve certain computational problems, such as integer
        factorization (which underlies RSA encryption), substantially faster than classical computers.
        They are also expected to be useful for simulating quantum systems in physics and chemistry.
        """,
        "quantum_challenges": """
        Building practical quantum computers is extremely challenging due to the need to maintain quantum
        coherence. Quantum systems are highly sensitive to noise and environmental interactions, which can
        cause decoherence and errors in calculations. Current quantum computers are still in early stages
        with limited qubits and high error rates.
        """,
    }

    # Define principles for ConstitutionalCritic
    principles = [
        "Explanations should be accurate and based on scientific facts.",
        "Content should be accessible to beginners without prior knowledge.",
        "Avoid unnecessary jargon and explain technical terms when used.",
        "Include practical examples or analogies to aid understanding.",
    ]

    # List to store performance results
    results = []

    # Test critics with OpenAI
    for critic_name, test_func, extra_args in [
        ("PromptCritic", test_prompt_critic, {}),
        ("ReflexionCritic", test_reflexion_critic, {}),
        ("SelfRefineCritic", test_self_refine_critic, {}),
        ("SelfRAGCritic", test_self_rag_critic, {"documents": documents}),
        ("ConstitutionalCritic", test_constitutional_critic, {"principles": principles}),
        ("LACCritic", test_lac_critic, {}),
    ]:
        logger.info(f"Testing {critic_name} with OpenAI")
        try:
            start_time = time.time()
            improved_text, revisions = test_func(openai_provider, input_text, task, **extra_args)
            time_spent = time.time() - start_time
            quality = evaluate_quality(improved_text)
            results.append(
                CriticPerformance(
                    name=critic_name,
                    model_provider="OpenAI GPT-4",
                    time_spent=time_spent,
                    revisions=revisions,
                    output_quality=quality,
                    output_length=len(improved_text),
                    output_text=improved_text,
                )
            )
        except Exception as e:
            logger.error(f"Error testing {critic_name} with OpenAI: {e}")

    # Test critics with Anthropic
    for critic_name, test_func, extra_args in [
        ("PromptCritic", test_prompt_critic, {}),
        ("ReflexionCritic", test_reflexion_critic, {}),
        ("SelfRefineCritic", test_self_refine_critic, {}),
        ("SelfRAGCritic", test_self_rag_critic, {"documents": documents}),
        ("ConstitutionalCritic", test_constitutional_critic, {"principles": principles}),
        ("LACCritic", test_lac_critic, {}),
    ]:
        logger.info(f"Testing {critic_name} with Anthropic")
        try:
            start_time = time.time()
            improved_text, revisions = test_func(anthropic_provider, input_text, task, **extra_args)
            time_spent = time.time() - start_time
            quality = evaluate_quality(improved_text)
            results.append(
                CriticPerformance(
                    name=critic_name,
                    model_provider="Anthropic Claude-3-Opus",
                    time_spent=time_spent,
                    revisions=revisions,
                    output_quality=quality,
                    output_length=len(improved_text),
                    output_text=improved_text,
                )
            )
        except Exception as e:
            logger.error(f"Error testing {critic_name} with Anthropic: {e}")

    # Display results
    logger.info("\n\nCritic Comparison Results:")
    logger.info("-" * 100)
    logger.info(
        f"{'Critic':<20} {'Model':<20} {'Time (s)':<10} {'Revisions':<10} {'Quality':<10} {'Length':<10}"
    )
    logger.info("-" * 100)

    for result in results:
        logger.info(
            f"{result.name:<20} {result.model_provider:<20} {result.time_spent:<10.2f} "
            f"{result.revisions:<10} {result.output_quality:<10.2f} {result.output_length:<10}"
        )

    logger.info("-" * 100)

    # Calculate and display average metrics by critic type
    logger.info("\nAverage Metrics by Critic Type:")
    logger.info("-" * 100)
    logger.info(f"{'Critic Type':<20} {'Time (s)':<10} {'Quality':<10} {'Length':<10}")
    logger.info("-" * 100)

    # Group results by critic type
    critic_types = {}
    for result in results:
        if result.name not in critic_types:
            critic_types[result.name] = []
        critic_types[result.name].append(result)

    # Calculate averages
    for critic_type, critic_results in critic_types.items():
        avg_time = sum(r.time_spent for r in critic_results) / len(critic_results)
        avg_quality = sum(r.output_quality for r in critic_results) / len(critic_results)
        avg_length = sum(r.output_length for r in critic_results) / len(critic_results)

        logger.info(f"{critic_type:<20} {avg_time:<10.2f} {avg_quality:<10.2f} {avg_length:<10.0f}")

    logger.info("-" * 100)

    # Print the output for each critic
    for result in results:
        logger.info(f"\n\n{result.name} with {result.model_provider} Output:")
        logger.info("-" * 100)
        logger.info(result.output_text)
        logger.info("-" * 100)


if __name__ == "__main__":
    run_critic_comparison()
