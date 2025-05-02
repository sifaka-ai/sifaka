"""Tests for the improvement module."""

import unittest
from unittest.mock import MagicMock
from typing import Dict, Any, Optional, List
from sifaka.improvement import Improver, ImprovementResult
from sifaka.validation import ValidationResult
from sifaka.critics.base import BaseCritic, CriticConfig, CriticMetadata
from sifaka.utils.logging import get_logger
from tests.base.test_base import BaseTestCase
import dataclasses

logger = get_logger(__name__)


class MockCritic(BaseCritic):
    """Mock critic for testing."""

    def __init__(self, feedback: Optional[Dict[str, Any] | CriticMetadata] = None):
        """Initialize mock critic."""
        super().__init__(CriticConfig(
            name="mock_critic",
            description="Mock critic for testing"
        ))
        self.feedback = feedback
        logger.debug(f"MockCritic initialized with feedback: {feedback}")

    def validate(self, text: str) -> bool:
        """Mock validate method."""
        return True

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Mock improve method."""
        return text

    def critique(self, text: str) -> Optional[CriticMetadata]:
        """Mock critique method."""
        logger.debug(f"MockCritic.critique called with text: {text}")
        logger.debug(f"Current feedback: {self.feedback}")
        logger.debug(f"Current feedback type: {type(self.feedback)}")

        if self.feedback is None:
            logger.debug("No feedback provided, returning None")
            return None

        if isinstance(self.feedback, CriticMetadata):
            logger.debug("Returning CriticMetadata feedback")
            logger.debug(f"CriticMetadata fields: {dataclasses.asdict(self.feedback)}")
            return self.feedback
        elif isinstance(self.feedback, dict):
            logger.debug("Converting dict feedback to CriticMetadata")
            # Handle both "score" and "confidence" keys
            feedback_dict = self.feedback.copy()
            score = feedback_dict.get("score", feedback_dict.get("confidence", 0.8))
            feedback = feedback_dict.get("feedback", "")
            suggestions = feedback_dict.get("suggestions", [])
            issues = feedback_dict.get("issues", [])
            attempt_number = feedback_dict.get("attempt_number", 1)
            processing_time_ms = feedback_dict.get("processing_time_ms", 0.0)

            metadata = CriticMetadata(
                score=score,
                feedback=feedback,
                suggestions=suggestions,
                issues=issues,
                attempt_number=attempt_number,
                processing_time_ms=processing_time_ms
            )
            logger.debug(f"Created CriticMetadata: {metadata}")
            logger.debug(f"CriticMetadata fields: {dataclasses.asdict(metadata)}")
            return metadata

        logger.debug("No feedback, returning None")
        return None


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
            suggestions=["suggestion1", "suggestion2"],
            issues=[],
            attempt_number=1,
            processing_time_ms=0.0
        )
        logger.debug(f"Created metadata: {metadata}")
        logger.debug(f"Metadata fields: {dataclasses.asdict(metadata)}")
        self.critic = MockCritic(metadata)  # Update instance variable
        logger.debug(f"Created critic with metadata: {self.critic.feedback}")
        self.improver = Improver(self.critic)  # Create new improver with updated critic
        result = self.improver.improve(output, validation_result)
        logger.debug(f"Got result: {result}")
        logger.debug(f"Result critique_details: {result.critique_details}")

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
            "suggestions": ["suggestion1", "suggestion2"],
            "issues": []
        }
        logger.debug(f"Created feedback: {feedback}")
        self.critic = MockCritic(feedback)  # Update instance variable
        logger.debug(f"Created critic with feedback: {self.critic.feedback}")
        self.improver = Improver(self.critic)  # Create new improver with updated critic
        result = self.improver.improve(output, validation_result)
        logger.debug(f"Got result: {result}")
        logger.debug(f"Result critique_details: {result.critique_details}")

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
        critic = MockCritic(None)  # Pass None instead of empty dict
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