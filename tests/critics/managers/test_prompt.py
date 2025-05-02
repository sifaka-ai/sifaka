"""
Tests for the critics PromptManager class.

This module contains comprehensive tests for the PromptManager abstract base class
and its DefaultPromptManager implementation. It focuses on ensuring proper
functionality for creating validation, critique, improvement, and reflection prompts.

The tests cover both the base class functionality and implementation-specific details.

Created during the test coverage improvement initiative.
"""

import unittest
from typing import List, Optional
import pytest

from sifaka.critics.models import CriticConfig
from sifaka.critics.managers.prompt import PromptManager, DefaultPromptManager


class TestDefaultPromptManager(unittest.TestCase):
    """Tests for the DefaultPromptManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CriticConfig(
            name="test_critic",
            description="Test critic",
            min_confidence=0.7,
            max_attempts=3
        )
        self.prompt_manager = DefaultPromptManager(self.config)

    def test_create_validation_prompt(self):
        """Test creating a validation prompt."""
        text = "This is a test text"
        prompt = self.prompt_manager.create_validation_prompt(text)

        # Check that the prompt contains the text
        self.assertIn(text, prompt)

        # Check required format components
        self.assertIn("TEXT TO VALIDATE", prompt)
        self.assertIn("VALID: [true/false]", prompt)
        self.assertIn("REASON:", prompt)
        self.assertIn("VALIDATION:", prompt)

    def test_create_critique_prompt(self):
        """Test creating a critique prompt."""
        text = "This is a test text"
        prompt = self.prompt_manager.create_critique_prompt(text)

        # Check that the prompt contains the text
        self.assertIn(text, prompt)

        # Check required format components
        self.assertIn("TEXT TO CRITIQUE", prompt)
        self.assertIn("SCORE:", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("ISSUES:", prompt)
        self.assertIn("SUGGESTIONS:", prompt)
        self.assertIn("CRITIQUE:", prompt)

    def test_create_improvement_prompt_without_reflections(self):
        """Test creating an improvement prompt without reflections."""
        text = "This is a test text"
        feedback = "This needs improvement"
        prompt = self.prompt_manager.create_improvement_prompt(text, feedback)

        # Check that the prompt contains the text and feedback
        self.assertIn(text, prompt)
        self.assertIn(feedback, prompt)

        # Check required format components
        self.assertIn("TEXT TO IMPROVE", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("IMPROVED_TEXT:", prompt)
        self.assertIn("IMPROVED VERSION:", prompt)

        # Verify reflections are not included
        self.assertNotIn("REFLECTIONS FROM PREVIOUS IMPROVEMENTS", prompt)

    def test_create_improvement_prompt_with_reflections(self):
        """Test creating an improvement prompt with reflections."""
        text = "This is a test text"
        feedback = "This needs improvement"
        reflections = ["First reflection", "Second reflection"]
        prompt = self.prompt_manager.create_improvement_prompt(text, feedback, reflections)

        # Check that the prompt contains the text, feedback, and reflections
        self.assertIn(text, prompt)
        self.assertIn(feedback, prompt)
        self.assertIn("REFLECTIONS FROM PREVIOUS IMPROVEMENTS", prompt)

        # Check that each reflection is included
        for i, reflection in enumerate(reflections):
            self.assertIn(f"{i+1}. {reflection}", prompt)

        # Check required format components
        self.assertIn("TEXT TO IMPROVE", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("IMPROVED_TEXT:", prompt)
        self.assertIn("IMPROVED VERSION:", prompt)

    def test_create_improvement_prompt_with_empty_reflections(self):
        """Test creating an improvement prompt with empty reflections list."""
        text = "This is a test text"
        feedback = "This needs improvement"
        reflections = []
        prompt = self.prompt_manager.create_improvement_prompt(text, feedback, reflections)

        # Check that the prompt contains the text and feedback
        self.assertIn(text, prompt)
        self.assertIn(feedback, prompt)

        # Verify reflections are not included
        self.assertNotIn("REFLECTIONS FROM PREVIOUS IMPROVEMENTS", prompt)

    def test_create_reflection_prompt(self):
        """Test creating a reflection prompt."""
        original_text = "This is the original text"
        feedback = "This needs improvement"
        improved_text = "This is the improved text"
        prompt = self.prompt_manager.create_reflection_prompt(original_text, feedback, improved_text)

        # Check that the prompt contains all components
        self.assertIn(original_text, prompt)
        self.assertIn(feedback, prompt)
        self.assertIn(improved_text, prompt)

        # Check required format components
        self.assertIn("ORIGINAL TEXT:", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("IMPROVED TEXT:", prompt)
        self.assertIn("REFLECTION:", prompt)


class CustomPromptManager(PromptManager):
    """Custom implementation of PromptManager for testing abstract base class."""

    def _create_validation_prompt_impl(self, text: str) -> str:
        """Custom implementation of create_validation_prompt."""
        return f"VALIDATION PROMPT: {text}"

    def _create_critique_prompt_impl(self, text: str) -> str:
        """Custom implementation of create_critique_prompt."""
        return f"CRITIQUE PROMPT: {text}"

    def _create_improvement_prompt_impl(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """Custom implementation of create_improvement_prompt."""
        result = f"IMPROVEMENT PROMPT: {text} | {feedback}"
        if reflections:
            result += f" | Reflections: {','.join(reflections)}"
        return result

    def _create_reflection_prompt_impl(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """Custom implementation of create_reflection_prompt."""
        return f"REFLECTION PROMPT: {original_text} | {feedback} | {improved_text}"


class TestPromptManagerAbstract(unittest.TestCase):
    """Tests for the abstract PromptManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CriticConfig(
            name="test_critic",
            description="Test critic",
            min_confidence=0.7,
            max_attempts=3
        )
        self.prompt_manager = CustomPromptManager(self.config)

    def test_create_validation_prompt(self):
        """Test that create_validation_prompt calls the implementation."""
        text = "This is a test text"
        prompt = self.prompt_manager.create_validation_prompt(text)
        self.assertEqual(prompt, f"VALIDATION PROMPT: {text}")

    def test_create_critique_prompt(self):
        """Test that create_critique_prompt calls the implementation."""
        text = "This is a test text"
        prompt = self.prompt_manager.create_critique_prompt(text)
        self.assertEqual(prompt, f"CRITIQUE PROMPT: {text}")

    def test_create_improvement_prompt_without_reflections(self):
        """Test that create_improvement_prompt calls the implementation without reflections."""
        text = "This is a test text"
        feedback = "This needs improvement"
        prompt = self.prompt_manager.create_improvement_prompt(text, feedback)
        self.assertEqual(prompt, f"IMPROVEMENT PROMPT: {text} | {feedback}")

    def test_create_improvement_prompt_with_reflections(self):
        """Test that create_improvement_prompt calls the implementation with reflections."""
        text = "This is a test text"
        feedback = "This needs improvement"
        reflections = ["First reflection", "Second reflection"]
        prompt = self.prompt_manager.create_improvement_prompt(text, feedback, reflections)
        self.assertEqual(
            prompt,
            f"IMPROVEMENT PROMPT: {text} | {feedback} | Reflections: {','.join(reflections)}"
        )

    def test_create_reflection_prompt(self):
        """Test that create_reflection_prompt calls the implementation."""
        original_text = "This is the original text"
        feedback = "This needs improvement"
        improved_text = "This is the improved text"
        prompt = self.prompt_manager.create_reflection_prompt(original_text, feedback, improved_text)
        self.assertEqual(
            prompt,
            f"REFLECTION PROMPT: {original_text} | {feedback} | {improved_text}"
        )


if __name__ == "__main__":
    unittest.main()