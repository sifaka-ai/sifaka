from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()

from sifaka.agents.config import ChainConfig
from sifaka.agents import create_pydantic_chain
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.models import create_model
from sifaka.core.thought import Thought

import pytest

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
            "You are a helpful assistant that can answer questions and help "
            "with tasks."
        ),
    )

    critic = SelfConsistencyCritic(model=model, **keywords_arguments)

    chain = create_pydantic_chain(agent=agent,)
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
        "high-impact output.")

    print(
        "The following chat completion takes about 2.5 minutes, so please wait...")
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
        for i, iteration in enumerate(result.improvement_history):
            print(f"  Iteration {i + 1}: {len(iteration)} characters")
