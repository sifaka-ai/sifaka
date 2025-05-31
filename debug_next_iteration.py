#!/usr/bin/env python3
"""Debug script to test next_iteration() behavior."""

from sifaka.core.thought import CriticFeedback, Thought


def test_next_iteration():
    """Test that next_iteration() properly resets critic_feedback."""

    # Create initial thought
    thought = Thought(prompt="Test prompt", text="Test text", iteration=0)

    print(
        f"Initial thought - iteration: {thought.iteration}, critic_feedback: {thought.critic_feedback}"
    )

    # Add some critic feedback
    feedback = CriticFeedback(
        critic_name="TestCritic",
        feedback="Test feedback",
        confidence=0.8,
        violations=["test issue"],
        suggestions=["test suggestion"],
        needs_improvement=True,
    )

    thought = thought.add_critic_feedback(feedback)
    print(
        f"After adding feedback - iteration: {thought.iteration}, critic_feedback count: {len(thought.critic_feedback or [])}"
    )

    # Create next iteration
    next_thought = thought.next_iteration()
    print(
        f"Next iteration - iteration: {next_thought.iteration}, critic_feedback: {next_thought.critic_feedback}"
    )

    # Verify the reset worked
    if next_thought.critic_feedback is None:
        print("✅ SUCCESS: critic_feedback was properly reset to None")
    else:
        print(
            f"❌ FAILURE: critic_feedback was not reset, contains: {next_thought.critic_feedback}"
        )

    # Test adding feedback to next iteration
    feedback2 = CriticFeedback(
        critic_name="TestCritic2",
        feedback="Test feedback 2",
        confidence=0.9,
        violations=["test issue 2"],
        suggestions=["test suggestion 2"],
        needs_improvement=True,
    )

    next_thought = next_thought.add_critic_feedback(feedback2)
    print(
        f"After adding feedback to next iteration - critic_feedback count: {len(next_thought.critic_feedback or [])}"
    )

    # Verify only new feedback is present
    if len(next_thought.critic_feedback or []) == 1:
        print("✅ SUCCESS: Only new feedback is present in next iteration")
    else:
        print(f"❌ FAILURE: Expected 1 feedback, got {len(next_thought.critic_feedback or [])}")


if __name__ == "__main__":
    test_next_iteration()
