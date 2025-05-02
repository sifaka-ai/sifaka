import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any, Optional

from sifaka.critics.style import StyleCritic, create_style_critic
from sifaka.critics.base import CriticMetadata, CriticConfig


class MockModelProvider:
    """Mock model provider for testing."""

    def __init__(self, response=None):
        """Initialize with optional custom response."""
        self.response = response or "This is an improved text. It has proper capitalization and punctuation."
        self.prompts = []

    def generate(self, prompt, **kwargs):
        """Mock generate method."""
        self.prompts.append(prompt)
        return self.response


class TestStyleCritic(unittest.TestCase):
    """Tests for StyleCritic class."""

    def setUp(self):
        """Set up test fixtures."""
        self.critic = StyleCritic()
        self.mock_model = MockModelProvider()
        self.critic_with_model = StyleCritic(model=self.mock_model)

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        critic = StyleCritic()
        self.assertEqual(critic._config.name, "style_critic")
        self.assertEqual(critic._config.description, "Analyzes and improves text style")
        self.assertEqual(critic.formality_level, "standard")
        self.assertListEqual(critic.style_elements, StyleCritic.DEFAULT_STYLE_ELEMENTS)
        self.assertIsNone(critic.model)

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = CriticConfig(
            name="custom_style_critic",
            description="Custom style critic",
            params={
                "style_elements": ["capitalization", "punctuation"],
                "formality_level": "formal"
            }
        )

        critic = StyleCritic(
            name="custom_style_critic",
            description="Custom style critic",
            config=config
        )

        self.assertEqual(critic._config.name, "custom_style_critic")
        self.assertEqual(critic._config.description, "Custom style critic")
        self.assertEqual(critic.formality_level, "formal")
        self.assertListEqual(critic.style_elements, ["capitalization", "punctuation"])

    def test_validate_valid_text(self):
        """Test validation with valid text."""
        text = "This is a properly formatted sentence. It has proper capitalization."
        self.assertTrue(self.critic.validate(text))

    def test_validate_invalid_text_missing_capitalization(self):
        """Test validation with missing capitalization."""
        text = "this is missing capitalization. This is correct."
        self.assertFalse(self.critic.validate(text))

    def test_validate_invalid_text_missing_punctuation(self):
        """Test validation with missing punctuation."""
        text = "This is missing ending punctuation"
        self.assertFalse(self.critic.validate(text))

    def test_validate_empty_text(self):
        """Test validation with empty text."""
        self.assertFalse(self.critic.validate(""))
        self.assertFalse(self.critic.validate("   "))

    def test_critique_valid_text(self):
        """Test critique with valid text."""
        text = "This is a properly formatted sentence. It has proper capitalization and punctuation."
        result = self.critic.critique(text)

        self.assertIsInstance(result, CriticMetadata)
        self.assertGreater(result.score, 0.8)  # Good score for valid text
        self.assertEqual(result.feedback, "Text has good style overall")
        self.assertEqual(len(result.issues), 0)  # No issues

    def test_critique_invalid_text(self):
        """Test critique with style issues."""
        text = "this is poorly formatted text it lacks proper capitalization and punctuation"
        result = self.critic.critique(text)

        self.assertIsInstance(result, CriticMetadata)
        self.assertLess(result.score, 0.5)  # Bad score for text with issues
        self.assertGreater(len(result.issues), 0)  # Should have issues
        self.assertGreater(len(result.suggestions), 0)  # Should have suggestions

        # Check for specific issues
        issues_text = ' '.join(result.issues).lower()
        self.assertTrue(
            'capitalization' in issues_text or 'punctuation' in issues_text,
            f"Expected capitalization or punctuation issues, got: {issues_text}"
        )

    def test_critique_empty_text(self):
        """Test critique with empty text."""
        result = self.critic.critique("")

        self.assertIsInstance(result, CriticMetadata)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.feedback, "Invalid or empty text")
        self.assertIn("empty", result.issues[0].lower())

    def test_improve_with_violations(self):
        """Test improve with style violations."""
        text = "this needs improvement it lacks proper style"
        violations = [
            {"issue": "Missing capitalization", "fix": "capitalize"},
            {"issue": "Missing punctuation", "fix": "add_period"}
        ]

        improved = self.critic.improve(text, violations)

        # Check if issues were fixed
        self.assertNotEqual(improved, text)
        self.assertTrue(improved[0].isupper())  # Should capitalize first letter
        self.assertTrue(improved[-1] in ".!?")  # Should add ending punctuation

    def test_improve_with_model(self):
        """Test improve with model provider."""
        text = "this needs improvement"
        violations = [{"issue": "Missing capitalization"}]

        improved = self.critic_with_model.improve(text, violations)

        # Should use model output
        self.assertEqual(improved, self.mock_model.response)

        # Verify prompt contains violation information
        self.assertIn("Missing capitalization", self.mock_model.prompts[0])

    def test_improve_with_feedback(self):
        """Test improve_with_feedback method."""
        text = "this text needs improvement and better style"
        feedback = "Improve capitalization and add proper punctuation"

        improved = self.critic.improve_with_feedback(text, feedback)

        # Check if feedback was applied
        self.assertNotEqual(improved, text)
        self.assertTrue(improved[0].isupper())  # Should capitalize first letter
        self.assertTrue(improved[-1] in ".!?")  # Should add ending punctuation

    def test_improve_with_feedback_and_model(self):
        """Test improve_with_feedback with model provider."""
        text = "this needs improvement"
        feedback = "Make it more formal"

        improved = self.critic_with_model.improve_with_feedback(text, feedback)

        # Should use model output
        self.assertEqual(improved, self.mock_model.response)

        # Verify prompt contains feedback
        self.assertIn(feedback, self.mock_model.prompts[0])

    def test_improve_with_formal_feedback(self):
        """Test improve_with_feedback with formality improvement."""
        text = "I don't wanna go and I can't see how it's gonna help."
        feedback = "Make it more formal"

        improved = self.critic.improve_with_feedback(text, feedback)

        # Check if contractions were expanded
        self.assertNotIn("don't", improved)
        self.assertIn("do not", improved)
        self.assertNotIn("wanna", improved)
        self.assertIn("want to", improved)
        self.assertNotIn("can't", improved)
        self.assertIn("cannot", improved)

    def test_empty_text_handling(self):
        """Test empty text handling in all methods."""
        empty_text = ""

        # validate should return False
        self.assertFalse(self.critic.validate(empty_text))

        # critique should return low score and error message
        result = self.critic.critique(empty_text)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.feedback, "Invalid or empty text")

        # improve should return empty string
        violations = [{"issue": "Empty text"}]
        self.assertEqual(self.critic.improve(empty_text, violations), empty_text)

        # improve_with_feedback should return empty string
        self.assertEqual(self.critic.improve_with_feedback(empty_text, "Fix it"), empty_text)

    def test_style_elements_filtering(self):
        """Test filtering of style elements in critique."""
        # Create critic with only capitalization checks
        config = CriticConfig(
            name="test_critic",
            description="Test critic",
            params={"style_elements": ["capitalization"]}
        )
        critic = StyleCritic(config=config)

        # Text with punctuation issues but correct capitalization
        text = "This has correct capitalization but no ending punctuation"
        result = critic.critique(text)

        # Should only check capitalization, not punctuation
        self.assertGreater(result.score, 0.8)  # Good score
        self.assertEqual(len(result.issues), 0)  # No issues for capitalization

        # Style scores should only include capitalization
        style_scores = result.extra.get("style_scores", {})
        self.assertIn("capitalization", style_scores)
        self.assertNotIn("punctuation", style_scores)


class TestCreateStyleCritic(unittest.TestCase):
    """Tests for create_style_critic factory function."""

    def test_create_with_defaults(self):
        """Test creation with default parameters."""
        critic = create_style_critic()

        self.assertIsInstance(critic, StyleCritic)
        self.assertEqual(critic._config.name, "style_critic")
        self.assertEqual(critic._config.description, "Analyzes and improves text style")
        self.assertEqual(critic.formality_level, "standard")

    def test_create_with_custom_params(self):
        """Test creation with custom parameters."""
        model = MockModelProvider()
        custom_elements = ["capitalization", "punctuation"]

        critic = create_style_critic(
            name="custom_critic",
            description="Custom description",
            min_confidence=0.6,
            formality_level="formal",
            style_elements=custom_elements,
            model=model
        )

        self.assertEqual(critic._config.name, "custom_critic")
        self.assertEqual(critic._config.description, "Custom description")
        self.assertEqual(critic.formality_level, "formal")
        self.assertEqual(critic.style_elements, custom_elements)
        self.assertEqual(critic.model, model)
        self.assertEqual(critic._config.min_confidence, 0.6)

    def test_create_with_additional_params(self):
        """Test creation with additional parameters."""
        critic = create_style_critic(
            params={"additional_param": "value"},
            max_attempts=5
        )

        self.assertEqual(critic._config.max_attempts, 5)
        self.assertEqual(critic._config.params.get("additional_param"), "value")

    def test_style_elements_override(self):
        """Test that style_elements overrides params value."""
        elements = ["capitalization", "sentence_structure"]
        critic = create_style_critic(
            style_elements=elements,
            params={"style_elements": ["should be overridden"]}
        )

        self.assertEqual(critic.style_elements, elements)


if __name__ == "__main__":
    unittest.main()