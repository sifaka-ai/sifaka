"""Integration tests for SelfConsistencyCritic with structured output.

This module tests the SelfConsistencyCritic to ensure it properly returns
CriticResult objects with structured feedback instead of the old dictionary format.
"""


import pytest
from dotenv import load_dotenv

load_dotenv()

from sifaka.core.thought import SifakaThought
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.models.critic_results import CriticResult, CritiqueFeedback


def test_SelfConsistencyCritic_inits():
    """Test SelfConsistencyCritic initialization."""
    keywords_arguments = {
        "num_consistency_attempts": 4,
        "consistency_threshold": 0.6,
    }

    critic = SelfConsistencyCritic(model_name="groq:llama3-8b-8192", **keywords_arguments)
    assert critic is not None
    assert critic.num_consistency_attempts == 4
    assert critic.consistency_threshold == 0.6


@pytest.mark.asyncio
async def test_self_consistency_critic_structured_output():
    """Test that SelfConsistencyCritic returns proper CriticResult objects."""
    critic = SelfConsistencyCritic(
        model_name="groq:llama3-8b-8192",
        num_consistency_attempts=2,  # Keep low for testing
        consistency_threshold=0.6,
    )

    # Create test thought
    thought = SifakaThought(prompt="Write a brief explanation of photosynthesis")
    thought.current_text = "Plants make food from sunlight. This process is called photosynthesis and it's very important for life on Earth."

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
    assert result.feedback.critic_name == "SelfConsistencyCritic"

    # Verify timing information if present
    if hasattr(result, "total_processing_time_ms") and result.total_processing_time_ms is not None:
        assert result.total_processing_time_ms >= 0

    print(f"✅ SelfConsistencyCritic returned proper CriticResult")
    print(f"   Message: {result.feedback.message}")
    print(f"   Needs improvement: {result.feedback.needs_improvement}")
    print(f"   Success: {result.success}")
    if hasattr(result, "total_processing_time_ms"):
        print(f"   Processing time: {result.total_processing_time_ms}ms")
