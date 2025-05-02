"""Tests for the improvement module."""

import unittest
from unittest.mock import MagicMock
from typing import Dict, Any, Optional, List
from sifaka.improvement import Improver, ImprovementResult
from sifaka.validation import ValidationResult
from sifaka.critics.base import BaseCritic, CriticConfig, CriticMetadata
from tests.base.test_base import BaseTestCase


class MockCritic(BaseCritic):
    """Mock critic for testing."""

    def __init__(self, feedback: Optional[Dict[str, Any]] = None):
        """Initialize mock critic."""
        super().__init__(CriticConfig(
            name="mock_critic",
            description="Mock critic for testing"
        ))
        self.feedback = feedback or {}

    def validate(self, text: str) -> bool:
        """Mock validate method."""
        return True

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Mock improve method."""
        return text

    def critique(self, text: str) -> CriticMetadata:
        """Mock critique method."""
        if isinstance(self.feedback, CriticMetadata):
            return self.feedback
        elif isinstance(self.feedback, dict):
            return CriticMetadata(
                score=self.feedback.get("confidence", 0.8),
                feedback=self.feedback.get("feedback", ""),
                suggestions=self.feedback.get("suggestions", [])
            )
        return CriticMetadata(
            score=0.8,
            feedback="",
            suggestions=[]
        )


class TestImprover(BaseTestCase):
    """Tests for Improver class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.critic = MockCritic()
        self.improver = Improver(self.critic)

    def test_improve_no_improvement_needed(self):
        """Test improvement when validation passed."""
        output = "test output"
        validation_result = ValidationResult(output=output, rule_results=[], all_passed=True)
        result = self.improver.improve(output, validation_result)

        self.assertIsInstance(result, ImprovementResult)
        self.assertEqual(result.output, output)
        self.assertFalse(result.improved)
        self.assertIsNone(result.critique_details)

    def test_improve_with_critic_metadata(self):
        """Test improvement with CriticMetadata feedback."""
        output = "test output"
        validation_result = ValidationResult(output=output, rule_results=[], all_passed=False)

        # Create critic with metadata feedback
        metadata = CriticMetadata(
            score=0.8,
            feedback="Test feedback",
            suggestions=["suggestion1", "suggestion2"]
        )
        self.critic = MockCritic(metadata)  # Use instance variable
        result = self.improver.improve(output, validation_result)

        self.assertIsInstance(result, ImprovementResult)
        self.assertEqual(result.output, output)
        self.assertTrue(result.improved)
        self.assertIsNotNone(result.critique_details)
        self.assertEqual(result.critique_details["feedback"], "Test feedback")
        self.assertEqual(result.critique_details["score"], 0.8)
        self.assertEqual(result.critique_details["suggestions"], ["suggestion1", "suggestion2"])

    def test_improve_with_dict_feedback(self):
        """Test improvement with dictionary feedback."""
        output = "test output"
        validation_result = ValidationResult(output=output, rule_results=[], all_passed=False)

        # Create critic with dictionary feedback
        feedback = {
            "feedback": "Test feedback",
            "confidence": 0.8,
            "suggestions": ["suggestion1", "suggestion2"]
        }
        self.critic = MockCritic(feedback)  # Use instance variable
        result = self.improver.improve(output, validation_result)

        self.assertIsInstance(result, ImprovementResult)
        self.assertEqual(result.output, output)
        self.assertTrue(result.improved)
        self.assertIsNotNone(result.critique_details)
        self.assertEqual(result.critique_details["feedback"], "Test feedback")
        self.assertEqual(result.critique_details["confidence"], 0.8)
        self.assertEqual(result.critique_details["suggestions"], ["suggestion1", "suggestion2"])

    def test_improve_no_feedback(self):
        """Test improvement when critic provides no feedback."""
        output = "test output"
        validation_result = ValidationResult(output=output, rule_results=[], all_passed=False)

        # Create critic that returns empty feedback
        critic = MockCritic({})
        improver = Improver(critic)

        result = improver.improve(output, validation_result)

        self.assertIsInstance(result, ImprovementResult)
        self.assertEqual(result.output, output)
        self.assertFalse(result.improved)
        self.assertIsNone(result.critique_details)

    def test_get_feedback_with_feedback(self):
        """Test getting feedback when it exists."""
        critique_details = {"feedback": "Test feedback"}
        feedback = self.improver.get_feedback(critique_details)
        self.assertEqual(feedback, "Test feedback")

    def test_get_feedback_without_feedback(self):
        """Test getting feedback when it doesn't exist."""
        critique_details = {"other": "value"}
        feedback = self.improver.get_feedback(critique_details)
        self.assertEqual(feedback, "")

    def test_get_feedback_non_dict(self):
        """Test getting feedback with non-dict input."""
        critique_details = "not a dict"
        feedback = self.improver.get_feedback(critique_details)
        self.assertEqual(feedback, "")


if __name__ == "__main__":
    unittest.main()