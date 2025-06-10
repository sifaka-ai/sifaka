"""Comprehensive integration tests for all critics.

This module tests multiple critics working together and ensures they all
follow the same interface and return consistent results.
"""


import pytest
from dotenv import load_dotenv

load_dotenv()

from sifaka.core.thought import SifakaThought
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.prompt import PromptCritic
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.models.critic_results import CriticResult, CritiqueFeedback


class TestCriticInterface:
    """Test that all critics follow the same interface."""

    @pytest.mark.parametrize(
        "critic_class,init_kwargs",
        [
            (ConstitutionalCritic, {"model_name": "openai:gpt-4o-mini"}),
            (ReflexionCritic, {"model_name": "openai:gpt-4o-mini"}),
            (
                SelfConsistencyCritic,
                {"model_name": "openai:gpt-4o-mini", "num_consistency_attempts": 2},
            ),
            (
                PromptCritic,
                {
                    "model_name": "openai:gpt-4o-mini",
                    "critique_prompt": "Evaluate this text for clarity.",
                },
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_critic_interface_consistency(self, critic_class, init_kwargs):
        """Test that all critics follow the same interface."""
        # Create critic
        critic = critic_class(**init_kwargs)

        # Create test thought
        thought = SifakaThought(prompt="Explain artificial intelligence")
        thought.current_text = "AI is a technology that makes computers smart. It can help with many tasks and is becoming more popular."

        # Test the critique method
        result = await critic.critique(thought)

        # Verify interface consistency
        assert isinstance(result, CriticResult)
        assert isinstance(result.feedback, CritiqueFeedback)
        assert isinstance(result.feedback.message, str)
        assert isinstance(result.feedback.needs_improvement, bool)
        assert isinstance(result.feedback.critic_name, str)
        assert result.operation_type == "critique"
        assert isinstance(result.success, bool)
        assert isinstance(result.input_text_length, int)

        # Verify critic name matches class
        expected_name = critic_class.__name__
        assert result.feedback.critic_name == expected_name

        print(f"✅ {expected_name} follows consistent interface")


class TestCriticQuality:
    """Test the quality and usefulness of critic feedback."""

    @pytest.mark.asyncio
    async def test_constitutional_critic_identifies_issues(self):
        """Test that ConstitutionalCritic identifies constitutional violations."""
        critic = ConstitutionalCritic(model_name="openai:gpt-4o-mini")

        # Create thought with clear constitutional issues
        thought = SifakaThought(prompt="Write about AI safety")
        thought.current_text = "AI is completely safe and never needs oversight. Trust AI systems blindly without any safety measures or regulations."

        result = await critic.critique(thought)

        # Should identify issues
        assert result.success
        assert result.feedback.needs_improvement  # Should flag this as problematic
        assert len(result.feedback.message) > 50  # Should provide detailed feedback

        print(f"✅ ConstitutionalCritic identified constitutional issues")

    @pytest.mark.asyncio
    async def test_reflexion_critic_provides_reflection(self):
        """Test that ReflexionCritic provides thoughtful reflection."""
        critic = ReflexionCritic(model_name="openai:gpt-4o-mini")

        # Create thought that could benefit from reflection
        thought = SifakaThought(prompt="Explain quantum computing")
        thought.current_text = (
            "Quantum computers are fast. They use quantum bits. They will solve everything."
        )

        result = await critic.critique(thought)

        # Should provide reflection
        assert result.success
        assert len(result.feedback.message) > 50  # Should provide detailed reflection

        print(f"✅ ReflexionCritic provided thoughtful reflection")

    @pytest.mark.asyncio
    async def test_self_consistency_critic_evaluates_consistency(self):
        """Test that SelfConsistencyCritic evaluates consistency."""
        critic = SelfConsistencyCritic(model_name="openai:gpt-4o-mini", num_consistency_attempts=2)

        # Create thought with potential consistency issues
        thought = SifakaThought(prompt="Explain machine learning")
        thought.current_text = "Machine learning is supervised learning. It's also unsupervised. All ML is deep learning."

        result = await critic.critique(thought)

        # Should evaluate consistency
        assert result.success
        assert len(result.feedback.message) > 30  # Should provide feedback

        print(f"✅ SelfConsistencyCritic evaluated consistency")


class TestCriticRobustness:
    """Test critic robustness with edge cases."""

    @pytest.mark.parametrize(
        "critic_class,init_kwargs",
        [
            (ConstitutionalCritic, {"model_name": "openai:gpt-4o-mini"}),
            (ReflexionCritic, {"model_name": "openai:gpt-4o-mini"}),
            (
                SelfConsistencyCritic,
                {"model_name": "openai:gpt-4o-mini", "num_consistency_attempts": 2},
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_critic_handles_empty_text(self, critic_class, init_kwargs):
        """Test that critics handle empty text gracefully."""
        critic = critic_class(**init_kwargs)

        # Create thought with empty text
        thought = SifakaThought(prompt="Write something")
        thought.current_text = ""

        result = await critic.critique(thought)

        # Should handle gracefully
        assert isinstance(result, CriticResult)
        assert isinstance(result.feedback, CritiqueFeedback)
        # May or may not need improvement, but should not crash

        print(f"✅ {critic_class.__name__} handled empty text gracefully")

    @pytest.mark.parametrize(
        "critic_class,init_kwargs",
        [
            (ConstitutionalCritic, {"model_name": "openai:gpt-4o-mini"}),
            (ReflexionCritic, {"model_name": "openai:gpt-4o-mini"}),
            (
                SelfConsistencyCritic,
                {"model_name": "openai:gpt-4o-mini", "num_consistency_attempts": 2},
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_critic_handles_long_text(self, critic_class, init_kwargs):
        """Test that critics handle long text appropriately."""
        critic = critic_class(**init_kwargs)

        # Create thought with long text
        long_text = "This is a test sentence. " * 100  # 2500 characters
        thought = SifakaThought(prompt="Write a long explanation")
        thought.current_text = long_text

        result = await critic.critique(thought)

        # Should handle gracefully
        assert isinstance(result, CriticResult)
        assert isinstance(result.feedback, CritiqueFeedback)
        assert result.input_text_length == len(long_text)

        print(f"✅ {critic_class.__name__} handled long text appropriately")


class TestCriticPerformance:
    """Test critic performance characteristics."""

    @pytest.mark.asyncio
    async def test_critic_processing_time_recorded(self):
        """Test that critics record processing time."""
        critic = ConstitutionalCritic(model_name="openai:gpt-4o-mini")

        thought = SifakaThought(prompt="Test prompt")
        thought.current_text = "This is a test text for performance measurement."

        result = await critic.critique(thought)

        # Should record processing time
        assert hasattr(result, "total_processing_time_ms")
        if result.total_processing_time_ms is not None:
            assert result.total_processing_time_ms >= 0

        print(f"✅ Processing time recorded: {result.total_processing_time_ms}ms")


@pytest.mark.asyncio
async def test_multiple_critics_on_same_thought():
    """Test running multiple critics on the same thought."""
    # Create critics
    constitutional = ConstitutionalCritic(model_name="openai:gpt-4o-mini")
    reflexion = ReflexionCritic(model_name="openai:gpt-4o-mini")

    # Create thought
    thought = SifakaThought(prompt="Explain climate change")
    thought.current_text = "Climate change is when the weather gets different. It's caused by pollution and stuff. We should probably do something about it."

    # Run both critics
    constitutional_result = await constitutional.critique(thought)
    reflexion_result = await reflexion.critique(thought)

    # Both should succeed
    assert constitutional_result.success
    assert reflexion_result.success

    # Should have different perspectives
    assert constitutional_result.feedback.critic_name == "ConstitutionalCritic"
    assert reflexion_result.feedback.critic_name == "ReflexionCritic"

    # Both should provide feedback
    assert len(constitutional_result.feedback.message) > 20
    assert len(reflexion_result.feedback.message) > 20

    print("✅ Multiple critics provided different perspectives on the same thought")
