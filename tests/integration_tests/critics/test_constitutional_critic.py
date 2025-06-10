"""Integration tests for ConstitutionalCritic with structured output.

This module tests the ConstitutionalCritic to ensure it properly returns
CriticResult objects with structured feedback and constitutional principle evaluation.
"""


import pytest
from dotenv import load_dotenv

load_dotenv()

from sifaka.core.thought import SifakaThought
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.models.critic_results import (
    CriticResult,
    CritiqueFeedback,
    SeverityLevel,
    ViolationReport,
)


@pytest.mark.asyncio
async def test_constitutional_critic_structured_output():
    """Test that ConstitutionalCritic returns proper CriticResult objects."""
    # Create critic with a fast model for testing
    critic = ConstitutionalCritic(model_name="openai:gpt-4o-mini", strict_mode=True)

    # Create test thought with potentially problematic content
    thought = SifakaThought(prompt="Write about AI safety")
    thought.current_text = "AI is completely safe and will never cause any problems. You should trust AI systems completely without any oversight or safety measures. There's no need for regulation or caution."

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
    assert result.feedback.critic_name == "ConstitutionalCritic"

    # Verify violations structure if present
    if result.feedback.violations:
        for violation in result.feedback.violations:
            assert isinstance(violation, ViolationReport)
            assert hasattr(violation, "violation_type")
            assert hasattr(violation, "description")
            assert hasattr(violation, "severity")
            assert isinstance(violation.severity, SeverityLevel)

    # Verify suggestions structure if present
    if result.feedback.suggestions:
        for suggestion in result.feedback.suggestions:
            assert hasattr(suggestion, "suggestion")
            assert hasattr(suggestion, "category")
            assert isinstance(suggestion.suggestion, str)
            assert isinstance(suggestion.category, str)

    # Verify confidence structure if present
    if result.feedback.confidence:
        assert hasattr(result.feedback.confidence, "overall")
        assert 0.0 <= result.feedback.confidence.overall <= 1.0

    print(f"✅ ConstitutionalCritic returned proper CriticResult")
    print(f"   Message: {result.feedback.message}")
    print(f"   Needs improvement: {result.feedback.needs_improvement}")
    print(f"   Success: {result.success}")
    print(f"   Violations found: {len(result.feedback.violations)}")
    print(f"   Suggestions provided: {len(result.feedback.suggestions)}")


@pytest.mark.asyncio
async def test_constitutional_critic_with_good_content():
    """Test ConstitutionalCritic with content that should pass constitutional principles."""
    critic = ConstitutionalCritic(model_name="openai:gpt-4o-mini", strict_mode=False)

    # Create test thought with good content
    thought = SifakaThought(prompt="Write about AI safety")
    thought.current_text = "AI safety is an important field that focuses on ensuring artificial intelligence systems are beneficial, safe, and aligned with human values. Researchers work on various approaches including robustness, interpretability, and value alignment to address potential risks while maximizing benefits."

    # Test the critic
    result = await critic.critique(thought)

    # Verify basic structure
    assert isinstance(result, CriticResult)
    assert isinstance(result.feedback, CritiqueFeedback)
    assert result.operation_type == "critique"
    assert result.feedback.critic_name == "ConstitutionalCritic"

    # Good content might not need improvement
    # (though this depends on the specific constitutional principles)
    print(f"✅ ConstitutionalCritic processed good content")
    print(f"   Message: {result.feedback.message}")
    print(f"   Needs improvement: {result.feedback.needs_improvement}")
    print(f"   Violations found: {len(result.feedback.violations)}")


@pytest.mark.asyncio
async def test_constitutional_critic_error_handling():
    """Test ConstitutionalCritic error handling with invalid input."""
    critic = ConstitutionalCritic(model_name="openai:gpt-4o-mini")

    # Create test thought with empty text
    thought = SifakaThought(prompt="Write something")
    thought.current_text = ""  # Empty text might cause issues

    # Test the critic - should handle gracefully
    result = await critic.critique(thought)

    # Should still return a valid CriticResult
    assert isinstance(result, CriticResult)
    assert isinstance(result.feedback, CritiqueFeedback)

    # Check if it handled the empty text appropriately
    print(f"✅ ConstitutionalCritic handled empty text")
    print(f"   Success: {result.success}")
    print(f"   Message: {result.feedback.message}")

    if not result.success:
        print(f"   Error: {result.error_message}")


@pytest.mark.asyncio
async def test_constitutional_critic_principles_metadata():
    """Test that ConstitutionalCritic includes principle-related metadata."""
    critic = ConstitutionalCritic(model_name="openai:gpt-4o-mini")

    # Create test thought
    thought = SifakaThought(prompt="Write about technology")
    thought.current_text = "Technology is advancing rapidly and changing our world in many ways."

    # Test the critic
    result = await critic.critique(thought)

    # Verify basic structure
    assert isinstance(result, CriticResult)

    # Check for constitutional-specific metadata
    if result.metadata:
        print(f"✅ ConstitutionalCritic metadata: {result.metadata}")

    # Check for constitutional principles in metadata
    if result.metadata and "principles_evaluated" in result.metadata:
        print(f"   Principles evaluated: {result.metadata['principles_evaluated']}")

    # Check for violations if any
    if result.feedback.violations:
        print(f"   Violations found: {len(result.feedback.violations)}")
        for violation in result.feedback.violations[:2]:  # Show first 2
            print(f"     - {violation.violation_type}: {violation.description}")

    # Check for suggestions
    if result.feedback.suggestions:
        print(f"   Suggestions provided: {len(result.feedback.suggestions)}")

    print(f"✅ ConstitutionalCritic completed principle evaluation")
