"""
Integration tests for the ReflexionCritic class.

These tests validate the integration between ReflexionCritic
and its component dependencies (without extensive mocking).
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
from typing import Dict, Any, List

from sifaka.critics.reflexion import (
    ReflexionCritic,
    ReflexionCriticConfig,
    ReflexionPromptFactory,
    create_reflexion_critic
)
from sifaka.critics.managers.memory import MemoryManager
from sifaka.critics.managers.response import ResponseParser
from sifaka.critics.base import CriticMetadata


# Add a concrete implementation of the abstract method
class ConcreteReflexionCritic(ReflexionCritic):
    """Concrete implementation of ReflexionCritic for testing."""

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Implement the abstract method required by the interface."""
        # Just delegate to improve
        return self.improve(text, feedback)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Implement the async version of the abstract method."""
        # Just delegate to aimprove
        return await self.aimprove(text, feedback)


class TestModelWithMemoryTracking:
    """Test language model that tracks input prompts and maintains memory of its responses."""

    def __init__(self, responses=None):
        """Initialize with optional predefined responses."""
        self.responses = responses or {}
        self.invoked_prompts = []
        self.default_response = "REFLECTION: This is a default reflection."
        self.reflection_count = 0

    def invoke(self, prompt):
        """Record the prompt and return an appropriate response."""
        self.invoked_prompts.append(prompt)

        if "validate" in prompt.lower():
            return self._get_response("validate", "VALID: true\nREASON: Valid content")
        elif "critique" in prompt.lower():
            return self._get_response("critique",
                "SCORE: 0.8\nFEEDBACK: Good content\nISSUES:\n- Issue 1\n- Issue 2\nSUGGESTIONS:\n- Suggestion 1\n- Suggestion 2")
        elif "improve" in prompt.lower():
            return self._get_response("improve", "IMPROVED_TEXT: This is improved text")
        elif "reflect" in prompt.lower():
            # Generate unique reflections to test memory
            self.reflection_count += 1
            return self._get_response("reflection", f"REFLECTION: This is reflection #{self.reflection_count}")

        return self.default_response

    async def ainvoke(self, prompt):
        """Async version of invoke."""
        return self.invoke(prompt)

    def _get_response(self, key, default):
        """Get a response for the given key or return default."""
        if isinstance(self.responses, dict) and key in self.responses:
            return self.responses[key]
        return default

    def get_prompt_count(self):
        """Get the number of prompts invoked."""
        return len(self.invoked_prompts)

    def get_prompt_containing(self, text):
        """
        Get all prompts containing the given text.

        This uses more precise matching to avoid false positives:
        - For 'improve', matches only prompts containing 'TEXT TO IMPROVE'
        - For 'reflect', matches only prompts containing 'REFLECTION:'
        - For 'validate', matches only prompts containing 'TEXT TO VALIDATE'
        - For 'critique', matches only prompts containing 'TEXT TO CRITIQUE'
        """
        if text == "improve":
            return [p for p in self.invoked_prompts if "TEXT TO IMPROVE:" in p]
        elif text == "reflect":
            return [p for p in self.invoked_prompts if "ORIGINAL TEXT:" in p and "REFLECTION:" in p]
        elif text == "validate":
            return [p for p in self.invoked_prompts if "TEXT TO VALIDATE:" in p]
        elif text == "critique":
            return [p for p in self.invoked_prompts if "TEXT TO CRITIQUE:" in p]
        else:
            return [p for p in self.invoked_prompts if text.lower() in p.lower()]

    def reset(self):
        """Reset the model for a new test."""
        self.invoked_prompts = []
        self.reflection_count = 0


class TestReflexionCriticIntegration(unittest.TestCase):
    """Integration tests for ReflexionCritic."""

    def setUp(self):
        """Set up the test environment."""
        self.test_model = TestModelWithMemoryTracking()

        # Create a real ReflexionCritic instance
        self.config = ReflexionCriticConfig(
            name="test_reflexion_critic",
            description="Test reflexion critic for integration tests",
            system_prompt="You are a test critic",
            temperature=0.7,
            max_tokens=500,
            memory_buffer_size=3,
            reflection_depth=1
        )

        # Create the critic with concrete implementation
        self.critic = ConcreteReflexionCritic(
            name=self.config.name,
            description=self.config.description,
            llm_provider=self.test_model,
            config=self.config,
            prompt_factory=ReflexionPromptFactory()
        )

    def tearDown(self):
        """Tear down the test environment."""
        # Reset the model for the next test
        self.test_model.reset()

    def test_real_critique_workflow(self):
        """Test the full critique workflow with real components."""
        # Perform a critique
        result = self.critic.critique("This is test content")

        # Verify that the model was called with a critique prompt
        critique_prompts = self.test_model.get_prompt_containing("critique")
        self.assertEqual(len(critique_prompts), 1)
        self.assertIn("This is test content", critique_prompts[0])

        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertIn("feedback", result)
        self.assertIn("issues", result)
        self.assertIn("suggestions", result)

        # Verify specific values
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good content")
        self.assertEqual(len(result["issues"]), 2)
        self.assertEqual(len(result["suggestions"]), 2)

    def test_real_improve_workflow(self):
        """Test the full improve workflow with real components."""
        # Perform an improvement
        result = self.critic.improve("This is content to improve", "Make it better")

        # Verify that the model was called with an improve prompt
        improve_prompts = self.test_model.get_prompt_containing("improve")
        self.assertEqual(len(improve_prompts), 1)
        self.assertIn("This is content to improve", improve_prompts[0])
        self.assertIn("Make it better", improve_prompts[0])

        # Verify that a reflection was generated
        reflection_prompts = self.test_model.get_prompt_containing("reflect")
        self.assertEqual(len(reflection_prompts), 1)

        # Verify the result
        self.assertEqual(result, "This is improved text")

    def test_memory_usage_in_real_critic(self):
        """Test that reflections are stored in memory and used in subsequent improvements."""
        # Perform multiple improvements to generate multiple reflections
        self.critic.improve("Content 1", "Feedback 1")
        self.critic.improve("Content 2", "Feedback 2")
        self.critic.improve("Content 3", "Feedback 3")

        # Verify that reflections were generated
        reflection_prompts = self.test_model.get_prompt_containing("reflect")
        self.assertEqual(len(reflection_prompts), 3)

        # Verify that memory was used in the last improvement
        improve_prompts = self.test_model.get_prompt_containing("improve")
        self.assertEqual(len(improve_prompts), 3)

        # The last improve prompt should contain reflections
        last_improve_prompt = improve_prompts[-1]
        self.assertIn("Feedback 3", last_improve_prompt)

    def test_violations_to_feedback_conversion(self):
        """Test conversion of rule violations to feedback text."""
        violations = [
            {"rule_name": "Grammar Rule", "message": "Fix grammar issues"},
            {"rule_name": "Style Rule", "message": "Improve style"}
        ]

        # Call improve with violations
        result = self.critic.improve("Test content", violations)

        # Verify that the model was called with an improve prompt containing the violations
        improve_prompts = self.test_model.get_prompt_containing("improve")
        self.assertEqual(len(improve_prompts), 1)
        last_prompt = improve_prompts[0]

        # Check that the violations were converted to feedback
        self.assertIn("Grammar Rule", last_prompt)
        self.assertIn("Fix grammar issues", last_prompt)
        self.assertIn("Style Rule", last_prompt)
        self.assertIn("Improve style", last_prompt)

    def test_validate_with_real_components(self):
        """Test validation with real components."""
        # Set up specific responses for this test
        self.test_model.responses = {
            "validate": "VALID: true\nREASON: This is valid content."
        }

        # Perform validation
        result = self.critic.validate("This is valid content")

        # Verify that the model was called with a validation prompt
        validate_prompts = self.test_model.get_prompt_containing("validate")
        self.assertEqual(len(validate_prompts), 1)
        self.assertIn("This is valid content", validate_prompts[0])

        # Verify the result
        self.assertTrue(result)

        # Test with invalid content
        self.test_model.responses = {
            "validate": "VALID: false\nREASON: This is invalid content."
        }

        result = self.critic.validate("This is invalid content")
        self.assertFalse(result)


@pytest.mark.asyncio
async def test_async_methods_integration():
    """Test async methods of ReflexionCritic with real components."""
    # Create test model and critic
    test_model = TestModelWithMemoryTracking()
    config = ReflexionCriticConfig(
        name="async_test_critic",
        description="Async test critic",
        system_prompt="You are an async test critic",
        memory_buffer_size=2
    )

    critic = ConcreteReflexionCritic(
        llm_provider=test_model,
        config=config,
        prompt_factory=ReflexionPromptFactory()
    )

    # Test async validate
    test_model.responses = {
        "validate": "VALID: true\nREASON: This is valid async content."
    }
    result = await critic.avalidate("This is async content")
    assert result is True

    # Test async critique
    test_model.responses = {
        "critique": "SCORE: 0.9\nFEEDBACK: Great async content\nISSUES:\n- None\nSUGGESTIONS:\n- None"
    }
    result = await critic.acritique("This is async content")
    assert isinstance(result, dict)
    assert result["score"] == 0.9
    assert result["feedback"] == "Great async content"

    # Test async improve
    test_model.responses = {
        "improve": "IMPROVED_TEXT: This is improved async content",
        "reflection": "REFLECTION: Async reflection"
    }
    result = await critic.aimprove("This is async content", "Make it better asynchronously")
    assert result == "This is improved async content"

    # Verify prompts were called
    assert len(test_model.get_prompt_containing("validate")) == 1
    assert len(test_model.get_prompt_containing("critique")) == 1
    assert len(test_model.get_prompt_containing("improve")) == 1
    assert len(test_model.get_prompt_containing("reflect")) == 1


# Patch create_reflexion_critic to return ConcreteReflexionCritic
@patch('sifaka.critics.reflexion.ReflexionCritic', ConcreteReflexionCritic)
def test_create_reflexion_critic_factory():
    """Test the create_reflexion_critic factory function with real components."""
    test_model = TestModelWithMemoryTracking()

    # Create a critic using the factory function
    critic = create_reflexion_critic(
        llm_provider=test_model,
        name="factory_test_critic",
        description="Created by factory function",
        system_prompt="You are a factory-created critic",
        temperature=0.5,
        max_tokens=800,
        memory_buffer_size=4,
        reflection_depth=2
    )

    # Verify that the critic was created with the correct configuration
    assert isinstance(critic, ReflexionCritic)
    assert critic.config.name == "factory_test_critic"
    assert critic.config.description == "Created by factory function"
    assert critic.config.system_prompt == "You are a factory-created critic"
    assert critic.config.temperature == 0.5
    assert critic.config.max_tokens == 800
    assert critic.config.memory_buffer_size == 4
    assert critic.config.reflection_depth == 2

    # Verify that the critic works
    result = critic.validate("Test factory content")
    assert result is True

    # Verify the model was called
    assert len(test_model.get_prompt_containing("validate")) == 1