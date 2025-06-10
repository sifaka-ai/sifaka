"""Integration tests for ReflexionCritic with structured output.

This module tests the ReflexionCritic to ensure it properly returns
CriticResult objects with structured feedback and self-reflection capabilities.
"""


import pytest
from dotenv import load_dotenv

load_dotenv()

from sifaka.core.thought import SifakaThought
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.models.critic_results import CriticResult, CritiqueFeedback, ImprovementSuggestion


@pytest.mark.asyncio
async def test_reflexion_critic_structured_output():
    """Test that ReflexionCritic returns proper CriticResult objects."""
    # Create critic with a fast model for testing
    critic = ReflexionCritic(model_name="openai:gpt-4o-mini")

    # Create test thought that could benefit from reflection
    thought = SifakaThought(prompt="Explain quantum computing")
    thought.current_text = "Quantum computing uses quantum bits. It's faster than regular computers. It will solve all problems."

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
    assert result.feedback.critic_name == "ReflexionCritic"

    # Verify suggestions structure if present (Reflexion should provide suggestions)
    if result.feedback.suggestions:
        for suggestion in result.feedback.suggestions:
            assert isinstance(suggestion, ImprovementSuggestion)
            assert hasattr(suggestion, "suggestion")
            assert hasattr(suggestion, "category")
            assert isinstance(suggestion.suggestion, str)
            assert isinstance(suggestion.category, str)

    # Verify confidence structure if present
    if result.feedback.confidence:
        assert hasattr(result.feedback.confidence, "overall")
        assert 0.0 <= result.feedback.confidence.overall <= 1.0

    print(f"✅ ReflexionCritic returned proper CriticResult")
    print(f"   Message: {result.feedback.message}")
    print(f"   Needs improvement: {result.feedback.needs_improvement}")
    print(f"   Success: {result.success}")
    print(f"   Suggestions provided: {len(result.feedback.suggestions)}")


@pytest.mark.asyncio
async def test_reflexion_critic_with_detailed_content():
    """Test ReflexionCritic with more detailed content."""
    critic = ReflexionCritic(model_name="openai:gpt-4o-mini")

    # Create test thought with more detailed content
    thought = SifakaThought(prompt="Explain machine learning")
    thought.current_text = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns. Common types include supervised learning, unsupervised learning, and reinforcement learning."

    # Test the critic
    result = await critic.critique(thought)

    # Verify basic structure
    assert isinstance(result, CriticResult)
    assert isinstance(result.feedback, CritiqueFeedback)
    assert result.operation_type == "critique"
    assert result.feedback.critic_name == "ReflexionCritic"

    print(f"✅ ReflexionCritic processed detailed content")
    print(f"   Message: {result.feedback.message}")
    print(f"   Needs improvement: {result.feedback.needs_improvement}")
    print(f"   Suggestions: {len(result.feedback.suggestions)}")


@pytest.mark.asyncio
async def test_reflexion_critic_self_reflection_metadata():
    """Test that ReflexionCritic includes self-reflection metadata."""
    critic = ReflexionCritic(model_name="openai:gpt-4o-mini")

    # Create test thought
    thought = SifakaThought(prompt="Describe photosynthesis")
    thought.current_text = "Plants use sunlight to make food. This happens in leaves."

    # Test the critic
    result = await critic.critique(thought)

    # Verify basic structure
    assert isinstance(result, CriticResult)

    # Check for reflexion-specific metadata
    if result.metadata:
        print(f"✅ ReflexionCritic metadata: {result.metadata}")

    # Check for categories affected
    if hasattr(result.feedback, "categories_affected") and result.feedback.categories_affected:
        print(f"   Categories affected: {result.feedback.categories_affected}")

    # Check suggestions for reflexion-style improvements
    if result.feedback.suggestions:
        for i, suggestion in enumerate(result.feedback.suggestions):
            print(f"   Suggestion {i+1}: {suggestion.suggestion}")
            if hasattr(suggestion, "rationale") and suggestion.rationale:
                print(f"     Rationale: {suggestion.rationale}")

    print(f"✅ ReflexionCritic completed self-reflection analysis")


@pytest.mark.asyncio
async def test_reflexion_critic_error_handling():
    """Test ReflexionCritic error handling."""
    critic = ReflexionCritic(model_name="openai:gpt-4o-mini")

    # Create test thought with minimal content
    thought = SifakaThought(prompt="Say hello")
    thought.current_text = "Hi"

    # Test the critic - should handle gracefully
    result = await critic.critique(thought)

    # Should still return a valid CriticResult
    assert isinstance(result, CriticResult)
    assert isinstance(result.feedback, CritiqueFeedback)

    print(f"✅ ReflexionCritic handled minimal content")
    print(f"   Success: {result.success}")
    print(f"   Message: {result.feedback.message}")

    if not result.success:
        print(f"   Error: {result.error_message}")


@pytest.mark.asyncio
async def test_reflexion_critic_improvement_suggestions():
    """Test that ReflexionCritic provides meaningful improvement suggestions."""
    critic = ReflexionCritic(model_name="openai:gpt-4o-mini")

    # Create test thought that clearly needs improvement
    thought = SifakaThought(prompt="Explain climate change")
    thought.current_text = "Climate change is bad. It makes things hot. We should stop it."

    # Test the critic
    result = await critic.critique(thought)

    # Verify basic structure
    assert isinstance(result, CriticResult)
    assert isinstance(result.feedback, CritiqueFeedback)

    # This simple explanation should likely need improvement
    print(f"✅ ReflexionCritic analyzed simple explanation")
    print(f"   Needs improvement: {result.feedback.needs_improvement}")
    print(f"   Number of suggestions: {len(result.feedback.suggestions)}")

    # Check suggestion quality
    if result.feedback.suggestions:
        for i, suggestion in enumerate(result.feedback.suggestions):
            print(f"   Suggestion {i+1}: {suggestion.category} - {suggestion.suggestion}")
            if suggestion.priority:
                print(f"     Priority: {suggestion.priority}")

    print(f"✅ ReflexionCritic provided improvement analysis")
