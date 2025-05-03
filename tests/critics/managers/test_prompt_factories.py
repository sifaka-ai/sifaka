"""
Tests for the critics prompt factory classes.

This module contains comprehensive tests for the PromptCriticPromptManager and
ReflexionCriticPromptManager classes. It focuses on ensuring proper functionality
for creating specialized prompts for different critic types.

The tests verify that both basic and specialized prompt managers correctly handle
their respective inputs and produce appropriate outputs, including optional parameters
like reflections.

Created during the test coverage improvement initiative.
"""

import unittest
from typing import List, Optional

from sifaka.critics.models import CriticConfig
from sifaka.critics.managers.prompt_factories import (
    PromptCriticPromptManager,
    ReflexionCriticPromptManager
)


class TestPromptCriticPromptManager(unittest.TestCase):
    """Tests for the PromptCriticPromptManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CriticConfig(
            name="test_critic",
            description="Test critic",
            min_confidence=0.7,
            max_attempts=3
        )
        self.prompt_manager = PromptCriticPromptManager(self.config)

    def test_create_validation_prompt(self):
        """Test creating a validation prompt."""
        text = "This is a test text"
        prompt = self.prompt_manager.create_validation_prompt(text)

        # Check that the prompt contains the text
        self.assertIn(text, prompt)

        # Check required format components
        self.assertIn("Please Validate the following text", prompt)
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
        self.assertIn("Please critique the following text", prompt)
        self.assertIn("TEXT TO CRITIQUE", prompt)
        self.assertIn("SCORE:", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("ISSUES:", prompt)
        self.assertIn("SUGGESTIONS:", prompt)
        self.assertIn("CRITIQUE:", prompt)

    def test_create_improvement_prompt(self):
        """Test creating an improvement prompt."""
        text = "This is a test text"
        feedback = "This needs improvement"
        prompt = self.prompt_manager.create_improvement_prompt(text, feedback)

        # Check that the prompt contains the text and feedback
        self.assertIn(text, prompt)
        self.assertIn(feedback, prompt)

        # Check required format components
        self.assertIn("Please improve the following text", prompt)
        self.assertIn("TEXT TO IMPROVE", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("IMPROVED_TEXT:", prompt)
        self.assertIn("IMPROVED VERSION:", prompt)

        # Verify reflections are not used in this implementation
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
        self.assertIn("Please reflect on the following text improvement", prompt)
        self.assertIn("ORIGINAL TEXT:", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("IMPROVED TEXT:", prompt)
        self.assertIn("REFLECTION:", prompt)


class TestReflexionCriticPromptManager(unittest.TestCase):
    """Tests for the ReflexionCriticPromptManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CriticConfig(
            name="test_critic",
            description="Test critic",
            min_confidence=0.7,
            max_attempts=3
        )
        self.prompt_manager = ReflexionCriticPromptManager(self.config)

    def test_create_validation_prompt(self):
        """Test creating a validation prompt."""
        text = "This is a test text"
        prompt = self.prompt_manager.create_validation_prompt(text)

        # Check that the prompt contains the text
        self.assertIn(text, prompt)

        # Check required format components
        self.assertIn("Please Validate the following text", prompt)
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
        self.assertIn("Please critique the following text", prompt)
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
        self.assertIn("Please improve the following text", prompt)
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
        self.assertIn("Please improve the following text", prompt)
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
        self.assertIn("Please reflect on the following text improvement", prompt)
        self.assertIn("ORIGINAL TEXT:", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("IMPROVED TEXT:", prompt)
        self.assertIn("REFLECTION:", prompt)


if __name__ == "__main__":
    unittest.main()