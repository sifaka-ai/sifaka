"""
Tests for the reflexion critic module.

This module tests the components in sifaka.critics.reflexion.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest

from sifaka.critics.reflexion import (
    ReflexionCritic,
    ReflexionCriticConfig,
    ReflexionPromptFactory,
    create_reflexion_critic,
)
from sifaka.critics.base import CriticMetadata


class TestReflexionPromptFactory(unittest.TestCase):
    """Tests for the ReflexionPromptFactory."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = ReflexionPromptFactory()

    def test_create_validation_prompt(self):
        """Test creating a validation prompt."""
        prompt = self.factory.create_validation_prompt("Text to validate")
        self.assertIn("TEXT TO VALIDATE:", prompt)
        self.assertIn("Text to validate", prompt)
        self.assertIn("VALID:", prompt)
        self.assertIn("REASON:", prompt)

    def test_create_critique_prompt(self):
        """Test creating a critique prompt."""
        prompt = self.factory.create_critique_prompt("Text to critique")
        self.assertIn("TEXT TO CRITIQUE:", prompt)
        self.assertIn("Text to critique", prompt)
        self.assertIn("SCORE:", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("ISSUES:", prompt)
        self.assertIn("SUGGESTIONS:", prompt)

    def test_create_improvement_prompt_without_reflections(self):
        """Test creating an improvement prompt without reflections."""
        prompt = self.factory.create_improvement_prompt("Text to improve", "Feedback")
        self.assertIn("TEXT TO IMPROVE:", prompt)
        self.assertIn("Text to improve", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("Feedback", prompt)
        self.assertIn("IMPROVED_TEXT:", prompt)
        self.assertNotIn("PREVIOUS REFLECTIONS:", prompt)

    def test_create_improvement_prompt_with_reflections(self):
        """Test creating an improvement prompt with reflections."""
        reflections = ["Reflection 1", "Reflection 2"]
        prompt = self.factory.create_improvement_prompt("Text to improve", "Feedback", reflections)
        self.assertIn("TEXT TO IMPROVE:", prompt)
        self.assertIn("Text to improve", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("Feedback", prompt)
        self.assertIn("IMPROVED_TEXT:", prompt)
        self.assertIn("PREVIOUS REFLECTIONS:", prompt)
        self.assertIn("1. Reflection 1", prompt)
        self.assertIn("2. Reflection 2", prompt)

    def test_create_reflection_prompt(self):
        """Test creating a reflection prompt."""
        prompt = self.factory.create_reflection_prompt(
            "Original text", "Feedback received", "Improved text"
        )
        self.assertIn("ORIGINAL TEXT:", prompt)
        self.assertIn("Original text", prompt)
        self.assertIn("FEEDBACK RECEIVED:", prompt)
        self.assertIn("Feedback received", prompt)
        self.assertIn("IMPROVED TEXT:", prompt)
        self.assertIn("Improved text", prompt)
        self.assertIn("REFLECTION:", prompt)


class TestReflexionHelperMethods(unittest.TestCase):
    """Tests for the helper methods in ReflexionCritic."""

    def setUp(self):
        """Set up common test fixtures."""
        self.mock_model = MagicMock()
        self.critic = ReflexionCritic(
            name="test_critic",
            description="Test critic",
            llm_provider=self.mock_model,
        )

    def test_violations_to_feedback_empty(self):
        """Test converting empty violations to feedback."""
        result = self.critic._violations_to_feedback([])
        self.assertEqual(result, "No issues found.")

    def test_violations_to_feedback_with_data(self):
        """Test converting violations to feedback."""
        violations = [
            {"rule_name": "Rule1", "message": "Violation 1"},
            {"rule_name": "Rule2", "message": "Violation 2"},
        ]
        result = self.critic._violations_to_feedback(violations)
        self.assertIn("Rule1: Violation 1", result)
        self.assertIn("Rule2: Violation 2", result)

    def test_parse_critique_response_empty(self):
        """Test parsing empty critique response."""
        result = self.critic._parse_critique_response("")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

    def test_parse_critique_response_score(self):
        """Test parsing critique response with score."""
        response = "SCORE: 0.8\nFEEDBACK: Good text"
        result = self.critic._parse_critique_response(response)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good text")

    def test_parse_critique_response_issues(self):
        """Test parsing critique response with issues."""
        response = "SCORE: 0.8\nFEEDBACK: Good text\nISSUES:\n- Issue 1\n- Issue 2"
        result = self.critic._parse_critique_response(response)
        self.assertEqual(result["issues"], ["Issue 1", "Issue 2"])

    def test_parse_critique_response_suggestions(self):
        """Test parsing critique response with suggestions."""
        response = "SCORE: 0.8\nFEEDBACK: Good text\nISSUES:\n- Issue 1\nSUGGESTIONS:\n- Suggestion 1\n- Suggestion 2"
        result = self.critic._parse_critique_response(response)
        self.assertEqual(result["suggestions"], ["Suggestion 1", "Suggestion 2"])

    def test_get_relevant_reflections(self):
        """Test getting relevant reflections."""
        # Mock memory manager
        self.critic._memory_manager = MagicMock()
        self.critic._memory_manager.get_memory.return_value = ["Reflection 1", "Reflection 2"]
        reflections = self.critic._get_relevant_reflections()
        self.assertEqual(reflections, ["Reflection 1", "Reflection 2"])
        self.critic._memory_manager.get_memory.assert_called_once()