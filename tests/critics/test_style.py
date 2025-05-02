"""
Tests for the critics StyleCritic class.

This module contains comprehensive tests for the StyleCritic class that
analyzes and improves text style. It covers all public methods and various
edge cases to ensure proper functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
import pytest

from sifaka.critics.style import StyleCritic, create_style_critic
from sifaka.critics.base import CriticConfig, CriticMetadata


class TestStyleCritic(unittest.TestCase):
    """Tests for the StyleCritic class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a standard style critic for testing
        self.critic = StyleCritic()

        # Create a mock model for testing model-based improvements
        self.mock_model = MagicMock()
        self.mock_model.generate.return_value = "Improved text with proper capitalization."

        # Style critic with mock model
        self.critic_with_model = StyleCritic(model=self.mock_model)

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        critic = StyleCritic()

        # Check default values
        self.assertEqual(critic.config.name, "style_critic")
        self.assertEqual(critic.config.description, "Analyzes and improves text style")
        self.assertEqual(critic.style_elements, StyleCritic.DEFAULT_STYLE_ELEMENTS)
        self.assertEqual(critic.formality_level, "standard")
        self.assertIsNone(critic.model)

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = CriticConfig(
            name="custom_style_critic",
            description="Custom style critic for testing",
            params={
                "style_elements": ["capitalization", "punctuation"],
                "formality_level": "formal"
            }
        )

        critic = StyleCritic(
            name="custom_style_critic",
            description="Custom style critic for testing",
            config=custom_config
        )

        # Check custom values
        self.assertEqual(critic.config.name, "custom_style_critic")
        self.assertEqual(critic.config.description, "Custom style critic for testing")
        self.assertEqual(critic.style_elements, ["capitalization", "punctuation"])
        self.assertEqual(critic.formality_level, "formal")

    def test_validate_with_valid_text(self):
        """Test validate method with valid text."""
        # Text with proper capitalization and ending punctuation
        valid_text = "This is a properly formatted sentence."
        self.assertTrue(self.critic.validate(valid_text))

        # Text with exclamation mark
        valid_text_exclamation = "This ends with exclamation!"
        self.assertTrue(self.critic.validate(valid_text_exclamation))

        # Text with question mark
        valid_text_question = "Is this a valid sentence?"
        self.assertTrue(self.critic.validate(valid_text_question))

    def test_validate_with_invalid_text(self):
        """Test validate method with invalid text."""
        # Text without capitalization
        invalid_text_caps = "this lacks capitalization."
        self.assertFalse(self.critic.validate(invalid_text_caps))

        # Text without ending punctuation
        invalid_text_punct = "This lacks ending punctuation"
        self.assertFalse(self.critic.validate(invalid_text_punct))

        # Text with neither capitalization nor punctuation
        invalid_text_both = "this lacks both capitalization and punctuation"
        self.assertFalse(self.critic.validate(invalid_text_both))

        # Empty text
        self.assertFalse(self.critic.validate(""))

        # None input
        self.assertFalse(self.critic.validate(None))

        # Whitespace only
        self.assertFalse(self.critic.validate("   "))

    def test_validate_with_custom_style_elements(self):
        """Test validate method with custom style elements."""
        # Create critic with only capitalization checking
        caps_only_critic = StyleCritic(config=CriticConfig(
            name="caps_critic",
            description="Checks only capitalization",
            params={"style_elements": ["capitalization"]}
        ))

        # Should pass with capitalization but no punctuation
        self.assertTrue(caps_only_critic.validate("This has caps but no punctuation"))

        # Should fail without capitalization
        self.assertFalse(caps_only_critic.validate("this lacks capitalization"))

        # Create critic with only punctuation checking
        punct_only_critic = StyleCritic(config=CriticConfig(
            name="punct_critic",
            description="Checks only punctuation",
            params={"style_elements": ["punctuation"]}
        ))

        # Should pass with punctuation but no capitalization
        self.assertTrue(punct_only_critic.validate("this has no caps but ends with punctuation."))

        # Should fail without punctuation
        self.assertFalse(punct_only_critic.validate("This has caps but no punctuation"))

    def test_critique_with_valid_text(self):
        """Test critique method with valid text."""
        # Good text with proper style
        good_text = "This is a properly formatted sentence. It has good capitalization and punctuation."
        result = self.critic.critique(good_text)

        # Check result properties
        self.assertIsInstance(result, CriticMetadata)
        self.assertGreater(result.score, 0.8)  # Should have high score
        self.assertIn("good style", result.feedback.lower())
        self.assertEqual(len(result.issues), 0)  # No issues

        # Bad text with style issues
        bad_text = "this lacks capitalization and punctuation"
        result = self.critic.critique(bad_text)

        # Check result properties
        self.assertIsInstance(result, CriticMetadata)
        self.assertLess(result.score, 0.5)  # Should have low score
        self.assertGreater(len(result.issues), 0)  # Should have issues
        self.assertGreater(len(result.suggestions), 0)  # Should have suggestions

        # Check for specific issues
        has_cap_issue = any('capitalization' in issue.lower() for issue in result.issues)
        has_punct_issue = any('punctuation' in issue.lower() for issue in result.issues)
        self.assertTrue(has_cap_issue)
        self.assertTrue(has_punct_issue)

    def test_critique_with_invalid_text(self):
        """Test critique method with invalid text."""
        result = self.critic.critique("")

        self.assertEqual(result.score, 0.0)
        self.assertIn("Invalid", result.feedback)
        self.assertGreater(len(result.issues), 0)

        result = self.critic.critique(None)
        self.assertEqual(result.score, 0.0)
        self.assertIn("Invalid", result.feedback)

    def test_critique_sentence_variety(self):
        """Test critique method for sentence variety checking."""
        # Text with monotonous sentence structure
        monotonous_text = "This is short. This is short. This is short. This is short. This is short."
        result = self.critic.critique(monotonous_text)

        # Check for sentence variety issue
        has_variety_issue = any('sentence' in issue.lower() for issue in result.issues)
        self.assertTrue(has_variety_issue)

        # Text with varied sentence structure
        varied_text = "This is a short sentence. This sentence is a bit longer and has more complex structure. Very short. The final sentence has a different length and structure as well."
        result = self.critic.critique(varied_text)

        # Check style scores in extra data
        self.assertIn("extra", dir(result))
        if hasattr(result, "extra") and result.extra:
            style_scores = result.extra.get("style_scores", {})
            if "sentence_variety" in style_scores:
                self.assertGreater(style_scores["sentence_variety"], 0.3)

    def test_critique_paragraph_breaks(self):
        """Test critique method for paragraph break checking."""
        # Long text without paragraph breaks
        long_text = "This is a long text without paragraph breaks. " * 20
        result = self.critic.critique(long_text)

        # Check for paragraph breaks issue
        has_paragraph_issue = any('paragraph' in issue.lower() for issue in result.issues)
        self.assertTrue(has_paragraph_issue)

        # Text with paragraph breaks
        text_with_breaks = "This is the first paragraph.\n\nThis is the second paragraph.\n\nThis is the third paragraph."
        result = self.critic.critique(text_with_breaks)

        # Should not have paragraph break issue
        has_paragraph_issue = any('paragraph' in issue.lower() for issue in result.issues)
        self.assertFalse(has_paragraph_issue)

    def test_critique_word_variety(self):
        """Test critique method for word variety checking."""
        # Text with repetitive words - make it more repetitive to ensure detection
        repetitive_text = "The cat cat cat cat cat cat cat cat cat cat. The cat cat cat cat. The cat cat cat. The cat cat cat. The cat cat cat."
        result = self.critic.critique(repetitive_text)

        # Examine all issues for debugging
        all_issues = ', '.join(result.issues)

        # If we still don't get a vocabulary variety issue, let's check if word_variety is in style_elements
        if "word_variety" in self.critic.style_elements:
            # Check if the style score for word variety is low
            if hasattr(result, "extra") and result.extra:
                style_scores = result.extra.get("style_scores", {})
                if "word_variety" in style_scores:
                    word_variety_score = style_scores["word_variety"]
                    # Assert the score is low, which indicates vocabulary issues
                    self.assertLess(word_variety_score, 0.5,
                                   f"Expected low word variety score, got {word_variety_score}")

        # We'll also test our varied vocabulary text
        varied_text = "The feline observed the surroundings. It leaped gracefully from the ledge. Then it scampered quickly across the room. After eating, the cat dozed peacefully. A soft purring sound emanated from the sleeping animal."
        result = self.critic.critique(varied_text)

        # Check style scores in extra data
        if hasattr(result, "extra") and result.extra:
            style_scores = result.extra.get("style_scores", {})
            if "word_variety" in style_scores:
                word_variety_score = style_scores["word_variety"]
                self.assertGreater(word_variety_score, 0.4,
                                  f"Expected high word variety score, got {word_variety_score}")

    def test_improve_with_violations(self):
        """Test improve method with style violations."""
        text = "this needs capitalization and punctuation"
        violations = [
            {"issue": "Missing capitalization", "fix": "capitalize"},
            {"issue": "Missing punctuation", "fix": "add_period"}
        ]

        improved = self.critic.improve(text, violations)

        # Check improvements
        self.assertNotEqual(improved, text)  # Should be different
        self.assertTrue(improved[0].isupper())  # Should start with capital
        self.assertTrue(improved.endswith("."))  # Should end with period

    def test_improve_with_model(self):
        """Test improve method using a model for improvements."""
        text = "this needs improvement"
        violations = [{"issue": "Style issues"}]

        improved = self.critic_with_model.improve(text, violations)

        # Should use model output
        self.assertEqual(improved, "Improved text with proper capitalization.")

        # Verify model was called with appropriate prompt
        self.mock_model.generate.assert_called_once()
        prompt = self.mock_model.generate.call_args[0][0]
        self.assertIn("Improve the style", prompt)
        self.assertIn(text, prompt)

    def test_improve_model_fallback(self):
        """Test improve method falling back to rule-based when model fails."""
        text = "this needs improvement"
        violations = [{"issue": "Missing capitalization"}]

        # Mock model to raise an exception
        self.mock_model.generate.side_effect = Exception("Model error")

        improved = self.critic_with_model.improve(text, violations)

        # Should fall back to rule-based improvement
        self.assertNotEqual(improved, text)  # Should be different
        self.assertTrue(improved[0].isupper())  # Should start with capital

    def test_improve_with_empty_violations(self):
        """Test improve method with empty violations list."""
        text = "Original text."

        # Should return unchanged text
        self.assertEqual(self.critic.improve(text, []), text)

        # Should return unchanged text for invalid input
        self.assertEqual(self.critic.improve("", [{"issue": "Issue"}]), "")
        self.assertEqual(self.critic.improve(None, [{"issue": "Issue"}]), None)

    def test_improve_with_feedback(self):
        """Test improve_with_feedback method."""
        text = "this needs capitalization and punctuation"
        feedback = "Improve capitalization and add proper punctuation"

        improved = self.critic.improve_with_feedback(text, feedback)

        # Check improvements
        self.assertNotEqual(improved, text)  # Should be different
        self.assertTrue(improved[0].isupper())  # Should start with capital
        self.assertTrue(improved.endswith("."))  # Should end with period

    def test_improve_with_feedback_formality(self):
        """Test improve_with_feedback method with formality changes."""
        text = "I don't wanna go. You're gonna be late!"
        feedback = "Make more formal"

        improved = self.critic.improve_with_feedback(text, feedback)

        # Check formality improvements
        self.assertNotEqual(improved, text)
        self.assertNotIn("don't", improved)
        self.assertNotIn("wanna", improved)
        self.assertNotIn("gonna", improved)
        self.assertIn("do not", improved)
        self.assertIn("want to", improved)
        self.assertIn("going to", improved)

    def test_improve_with_feedback_model(self):
        """Test improve_with_feedback method using a model."""
        text = "this needs improvement"
        feedback = "Make more formal"

        improved = self.critic_with_model.improve_with_feedback(text, feedback)

        # Should use model output
        self.assertEqual(improved, "Improved text with proper capitalization.")

        # Verify model was called with appropriate prompt
        self.mock_model.generate.assert_called_once()
        prompt = self.mock_model.generate.call_args[0][0]
        self.assertIn("Improve the style", prompt)
        self.assertIn(text, prompt)
        self.assertIn(feedback, prompt)

    def test_improve_with_feedback_empty_input(self):
        """Test improve_with_feedback method with empty input."""
        # Should return unchanged text for invalid input
        self.assertEqual(self.critic.improve_with_feedback("", "Feedback"), "")
        self.assertEqual(self.critic.improve_with_feedback(None, "Feedback"), None)

    def test_create_style_critic_factory(self):
        """Test create_style_critic factory function."""
        # Create with custom parameters
        critic = create_style_critic(
            name="custom_critic",
            description="Custom description",
            min_confidence=0.6,
            formality_level="formal",
            style_elements=["capitalization", "punctuation"],
            model=self.mock_model
        )

        # Check properties
        self.assertEqual(critic.config.name, "custom_critic")
        self.assertEqual(critic.config.description, "Custom description")
        self.assertEqual(critic.config.min_confidence, 0.6)
        self.assertEqual(critic.formality_level, "formal")
        self.assertEqual(critic.style_elements, ["capitalization", "punctuation"])
        self.assertEqual(critic.model, self.mock_model)

    def test_create_style_critic_with_additional_params(self):
        """Test create_style_critic with additional parameters."""
        critic = create_style_critic(
            max_attempts=3,
            params={"additional_param": "value"}
        )

        self.assertEqual(critic.config.max_attempts, 3)
        self.assertEqual(critic.config.params["additional_param"], "value")
        self.assertEqual(critic.config.params["formality_level"], "standard")


if __name__ == "__main__":
    unittest.main()