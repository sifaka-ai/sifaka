"""Integration tests for SelfConsistencyCritic with structured output.

This module tests the SelfConsistencyCritic to ensure it properly returns
CriticResult objects with structured feedback instead of the old dictionary format.
"""

import pytest
from datetime import datetime
from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()

from sifaka.agents.config import ChainConfig
from sifaka.agents import create_pydantic_chain
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.models import create_model
from sifaka.models.critic_results import CriticResult, CritiqueFeedback
from sifaka.core.thought import Thought


def test_SelfConsistencyCritic_inits():

    model_provide_name = "groq:llama3-8b-8192"
    model = create_model(model_provide_name)
    assert model is not None
    assert model.model_name == model_provide_name

    keywords_arguments = {
        "num_iterations": 4,
        "consensus_threshold": 0.6,
        "aggregation_method": "majority_vote",
        "use_chain_of_thought": True,
    }

    critic = SelfConsistencyCritic(model=model, **keywords_arguments)
    assert critic is not None
    assert critic.num_iterations == 4
    assert critic.use_chain_of_thought
    assert critic.use_chain_of_thought == True

    assert critic.critique_prompt_template == critic._default_critique_template()


@pytest.mark.asyncio
async def test_create_pydantic_chain_with_self_consistency_critic():
    model_provide_name = "groq:llama3-8b-8192"
    model = create_model(model_provide_name)

    keywords_arguments = {
        "num_iterations": 4,
        "consensus_threshold": 0.6,
        "aggregation_method": "majority_vote",
        "use_chain_of_thought": True,
    }

    agent = Agent(
        "groq:llama-3.1-8b-instant",
        system_prompt=(
            "You are a helpful assistant that can answer questions and help " "with tasks."
        ),
    )

    critic = SelfConsistencyCritic(model=model, **keywords_arguments)

    chain = create_pydantic_chain(
        agent=agent,
    )
    assert chain is not None
    chain = create_pydantic_chain(agent=agent, critics=[critic], max_improvement_iterations=2)
    assert chain is not None

    assert chain.agent == agent
    assert chain.config == ChainConfig(
        validators=[],
        critics=[critic],
        model_retrievers=[],
        critic_retrievers=[],
    )

    user_prompt = (
        "Create a detailed Q1 2025 financial report for a hypothetical company, "
        "'TechTrend Innovations,' integrating revenue, net income, EBITDA, and "
        "stock price trends from a simulated Alpha Vantage API, a market "
        "sentiment analysis from 50 hypothetical X posts, and three risk "
        "factors with logical reasoning, ensuring the report is professional, "
        "under 1,000 words, validated for completeness and accuracy using "
        "sifaka-ai’s Self-Consistency Critic with three critiques, "
        "automatically correcting failures (e.g., missing data) with MongoDB "
        "logging, and providing a confidence score, all within 5 minutes. "
        "Demonstrate the process in a Flask-based interface, showing data "
        "integration, critique feedback, and corrections for a reliable, "
        "high-impact output."
    )

    print("The following chat completion takes about 2.5 minutes, so please wait...")
    response = await chain.run(user_prompt)
    # type coroutine
    print("type(response)", type(response))

    assert isinstance(response, Thought)
    print(response.iteration)
    print(response.text)
    print("len(response.text)", len(response.text))

    # Show critic feedback if available
    if response.critic_feedback:
        print(f"\n🔍 Critic feedback ({len(response.critic_feedback)} entries):")
        print("-" * 30)
        for i, feedback in enumerate(response.critic_feedback, 1):
            print(f"\nFeedback {i}:")
            print(f"  Critic: {feedback.critic_name}")
            print(f"  Needs improvement: {feedback.needs_improvement}")
            print(f"  Message: {feedback.feedback}")  # Show full feedback

            # Show confidence if available (SelfConsistency specific)
            if hasattr(feedback, "confidence"):
                print(f"  Confidence: {feedback.confidence:.2f}")

            # Debug: Show raw individual critiques if available
            if hasattr(feedback, "metadata") and feedback.metadata:
                individual_critiques = feedback.metadata.get("individual_critiques", [])
                if individual_critiques:
                    print(
                        f"\n  🔍 DEBUG: Raw model responses ({len(individual_critiques)} critiques):"
                    )
                    for j, critique in enumerate(individual_critiques[:2], 1):  # Show first 2
                        print(f"    Raw Critique {j}:")
                        print(f"    {critique.get('message', 'No message')[:300]}...")
                        print(f"    Parsed Issues: {critique.get('issues', [])}")
                        print(f"    Parsed Suggestions: {critique.get('suggestions', [])}")
                        print()

    # Show improvement history if available
    if hasattr(response, "improvement_history") and response.improvement_history:
        print(f"\n📈 Improvement history ({len(response.improvement_history)} iterations):")
        print("-" * 30)
        for i, iteration in enumerate(response.improvement_history):
            print(f"  Iteration {i + 1}: {len(iteration)} characters")


@pytest.mark.asyncio
async def test_self_consistency_critic_structured_output():
    """Test that SelfConsistencyCritic returns proper CriticResult objects."""
    critic = SelfConsistencyCritic(
        model_name="groq:llama3-8b-8192",
        num_iterations=2,  # Keep low for testing
        consensus_threshold=0.6,
        aggregation_method="majority_vote",
        use_chain_of_thought=True,
    )

    # Create test thought
    thought = Thought(
        prompt="Write a brief explanation of photosynthesis",
        text="Plants make food from sunlight. This process is called photosynthesis and it's very important for life on Earth.",
        timestamp=datetime.now(),
        id="test_self_consistency_001",
    )

    # Test the critic
    result = await critic.critique(thought)

    # Verify it returns a CriticResult object
    assert isinstance(result, CriticResult)

    # Verify the structure
    assert hasattr(result, "feedback")
    assert isinstance(result.feedback, CritiqueFeedback)
    assert hasattr(result.feedback, "message")
    assert hasattr(result.feedback, "needs_improvement")
    assert isinstance(result.feedback.message, str)
    assert isinstance(result.feedback.needs_improvement, bool)

    # Verify operation metadata
    assert result.operation_type == "critique"
    assert isinstance(result.success, bool)
    assert result.critic_name == "SelfConsistencyCritic"

    # Verify timing information if present
    if result.processing_time_ms is not None:
        assert result.processing_time_ms >= 0

    print(f"✅ SelfConsistencyCritic returned proper CriticResult")
    print(f"   Message: {result.feedback.message}")
    print(f"   Needs improvement: {result.feedback.needs_improvement}")
    print(f"   Success: {result.success}")
    print(f"   Processing time: {result.processing_time_ms}ms")
