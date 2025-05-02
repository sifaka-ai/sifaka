"""
Tests for the reflexion critic module.

This module contains isolated tests for the reflexion critic components defined in
sifaka/critics/reflexion.py, focusing on components with low test coverage.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestReflexionCriticConfig(unittest.TestCase):
    """Tests for the ReflexionCriticConfig class."""

    def test_valid_config(self):
        """Test creating a valid config."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCriticConfig

        config = ReflexionCriticConfig(
            name="test_critic",
            description="Test critic",
            system_prompt="Test system prompt",
            temperature=0.5,
            max_tokens=100,
            memory_buffer_size=3,
            reflection_depth=2,
        )
        self.assertEqual(config.name, "test_critic")
        self.assertEqual(config.description, "Test critic")
        self.assertEqual(config.system_prompt, "Test system prompt")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 100)
        self.assertEqual(config.memory_buffer_size, 3)
        self.assertEqual(config.reflection_depth, 2)

    def test_invalid_system_prompt(self):
        """Test creating a config with an invalid system prompt."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCriticConfig

        with self.assertRaises(ValueError):
            ReflexionCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="",  # Empty system prompt
                temperature=0.5,
                max_tokens=100,
            )

    def test_invalid_temperature(self):
        """Test creating a config with an invalid temperature."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCriticConfig

        with self.assertRaises(ValueError):
            ReflexionCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="Test system prompt",
                temperature=1.5,  # Temperature > 1
                max_tokens=100,
            )

    def test_invalid_max_tokens(self):
        """Test creating a config with invalid max_tokens."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCriticConfig

        with self.assertRaises(ValueError):
            ReflexionCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="Test system prompt",
                temperature=0.5,
                max_tokens=0,  # Max tokens must be positive
            )

    def test_invalid_memory_buffer_size(self):
        """Test creating a config with invalid memory_buffer_size."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCriticConfig

        with self.assertRaises(ValueError):
            ReflexionCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="Test system prompt",
                temperature=0.5,
                max_tokens=100,
                memory_buffer_size=-1,  # Negative buffer size
            )

    def test_invalid_reflection_depth(self):
        """Test creating a config with invalid reflection_depth."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCriticConfig

        with self.assertRaises(ValueError):
            ReflexionCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="Test system prompt",
                temperature=0.5,
                max_tokens=100,
                reflection_depth=0,  # Should be positive
            )


class TestReflexionPromptFactory(unittest.TestCase):
    """Tests for the ReflexionPromptFactory class."""

    def setUp(self):
        """Set up test environment."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionPromptFactory
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


class TestReflexionCriticHelperMethods(unittest.TestCase):
    """Tests for the helper methods in ReflexionCritic."""

    def test_violations_to_feedback_empty(self):
        """Test converting empty violations to feedback."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCritic

        # Create a minimal mock instance
        critic = MagicMock(spec=ReflexionCritic)
        critic._violations_to_feedback = ReflexionCritic._violations_to_feedback.__get__(critic)

        result = critic._violations_to_feedback([])
        self.assertEqual(result, "No issues found.")

    def test_violations_to_feedback(self):
        """Test converting violations to feedback."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCritic

        # Create a minimal mock instance
        critic = MagicMock(spec=ReflexionCritic)
        critic._violations_to_feedback = ReflexionCritic._violations_to_feedback.__get__(critic)

        violations = [
            {"rule_name": "Rule1", "message": "Violation 1"},
            {"rule_name": "Rule2", "message": "Violation 2"},
        ]
        result = critic._violations_to_feedback(violations)
        self.assertIn("Rule1: Violation 1", result)
        self.assertIn("Rule2: Violation 2", result)

    def test_parse_critique_response_score(self):
        """Test parsing critique response with score."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCritic

        # Create a minimal mock instance
        critic = MagicMock(spec=ReflexionCritic)
        critic._parse_critique_response = ReflexionCritic._parse_critique_response.__get__(critic)

        response = "SCORE: 0.8\nFEEDBACK: Good text"
        result = critic._parse_critique_response(response)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good text")

    def test_parse_critique_response_issues(self):
        """Test parsing critique response with issues."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCritic

        # Create a minimal mock instance
        critic = MagicMock(spec=ReflexionCritic)
        critic._parse_critique_response = ReflexionCritic._parse_critique_response.__get__(critic)

        response = "SCORE: 0.8\nFEEDBACK: Good text\nISSUES:\n- Issue 1\n- Issue 2"
        result = critic._parse_critique_response(response)
        self.assertEqual(result["issues"], ["Issue 1", "Issue 2"])

    def test_parse_critique_response_suggestions(self):
        """Test parsing critique response with suggestions."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCritic

        # Create a minimal mock instance
        critic = MagicMock(spec=ReflexionCritic)
        critic._parse_critique_response = ReflexionCritic._parse_critique_response.__get__(critic)

        response = "SCORE: 0.8\nFEEDBACK: Good text\nISSUES:\n- Issue 1\nSUGGESTIONS:\n- Suggestion 1\n- Suggestion 2"
        result = critic._parse_critique_response(response)
        self.assertEqual(result["suggestions"], ["Suggestion 1", "Suggestion 2"])

    def test_parse_critique_response_invalid(self):
        """Test parsing invalid critique response."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCritic

        # Create a minimal mock instance
        critic = MagicMock(spec=ReflexionCritic)
        critic._parse_critique_response = ReflexionCritic._parse_critique_response.__get__(critic)

        response = "Invalid response"
        result = critic._parse_critique_response(response)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

    def test_get_relevant_reflections(self):
        """Test getting relevant reflections."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCritic

        # Create a minimal mock instance
        critic = MagicMock(spec=ReflexionCritic)
        critic._get_relevant_reflections = ReflexionCritic._get_relevant_reflections.__get__(critic)

        # Setup memory manager
        critic._memory_manager = MagicMock()
        critic._memory_manager.get_memory.return_value = ["Reflection 1", "Reflection 2"]

        # Call the method
        reflections = critic._get_relevant_reflections()

        # Check the result
        self.assertEqual(reflections, ["Reflection 1", "Reflection 2"])
        critic._memory_manager.get_memory.assert_called_once()


class TestReflexionCritic(unittest.TestCase):
    """Tests for the ReflexionCritic class."""

    def setUp(self):
        """Set up test environment."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCritic, ReflexionCriticConfig

        # Create a mock LLM provider
        self.mock_llm = MagicMock()
        self.mock_llm.model_name = "test_model"
        self.mock_llm.generate.return_value = "IMPROVED_TEXT: Improved text"

        # Create a mock critique service
        self.mock_critique_service = MagicMock()
        self.mock_critique_service.improve.return_value = "Improved text"
        self.mock_critique_service.validate.return_value = True
        self.mock_critique_service.critique.return_value = {
            "score": 0.8,
            "feedback": "Good text",
            "issues": [],
            "suggestions": []
        }

        # Create a config
        self.config = ReflexionCriticConfig(
            name="test_critic",
            description="Test critic",
            system_prompt="Test system prompt",
            temperature=0.5,
            max_tokens=100,
            memory_buffer_size=3,
            reflection_depth=2,
        )

        # Patch the components
        with patch('sifaka.critics.managers.prompt_factories.ReflexionCriticPromptManager'), \
             patch('sifaka.critics.managers.response.ResponseParser'), \
             patch('sifaka.critics.managers.memory.MemoryManager'), \
             patch('sifaka.critics.services.critique.CritiqueService') as mock_service_class:
            mock_service_class.return_value = self.mock_critique_service
            self.critic = ReflexionCritic(
                name="test_critic",
                description="Test critic",
                llm_provider=self.mock_llm,
                config=self.config
            )

    def test_init_without_llm(self):
        """Test initializing without an LLM provider."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCritic
        from pydantic import ValidationError

        # Patch ValidationError
        with patch('sifaka.critics.reflexion.ValidationError') as mock_validation_error:
            mock_validation_error.from_exception_data.return_value = Exception("Field required")

            # Try to create a critic without an LLM
            with self.assertRaises(Exception):
                ReflexionCritic(name="test_critic", description="Test critic")

    def test_init_with_default_config(self):
        """Test initializing with default config."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCritic

        # Patch the components
        with patch('sifaka.critics.managers.prompt_factories.ReflexionCriticPromptManager'), \
             patch('sifaka.critics.managers.response.ResponseParser'), \
             patch('sifaka.critics.managers.memory.MemoryManager'), \
             patch('sifaka.critics.services.critique.CritiqueService'):

            # Create a critic with default config
            critic = ReflexionCritic(
                name="test_critic",
                description="Test critic",
                llm_provider=self.mock_llm,
            )

            # Check that the config was created
            self.assertIsNotNone(critic._config)

    def test_config_property(self):
        """Test the config property."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import ReflexionCriticConfig

        config = self.critic.config
        self.assertIsInstance(config, ReflexionCriticConfig)
        self.assertEqual(config.name, "test_critic")

    def test_validate(self):
        """Test the validate method."""
        result = self.critic.validate("Text to validate")
        self.assertTrue(result)
        self.mock_critique_service.validate.assert_called_once_with("Text to validate")

    def test_validate_empty_text(self):
        """Test the validate method with empty text."""
        with self.assertRaises(ValueError):
            self.critic.validate("")

    def test_improve_with_string_feedback(self):
        """Test the improve method with string feedback."""
        result = self.critic.improve("Text to improve", "Feedback")
        self.assertEqual(result, "Improved text")
        self.mock_critique_service.improve.assert_called_once_with("Text to improve", "Feedback")

    def test_improve_with_violations_feedback(self):
        """Test the improve method with violations feedback."""
        # Mock the violations_to_feedback method
        original_method = self.critic._violations_to_feedback
        self.critic._violations_to_feedback = MagicMock(return_value="Formatted feedback")

        # Call the method with violations
        violations = [{"rule_name": "Rule1", "message": "Violation 1"}]
        result = self.critic.improve("Text to improve", violations)

        # Check the result
        self.assertEqual(result, "Improved text")
        self.critic._violations_to_feedback.assert_called_once_with(violations)
        self.mock_critique_service.improve.assert_called_once_with("Text to improve", "Formatted feedback")

        # Restore the original method
        self.critic._violations_to_feedback = original_method

    def test_improve_without_feedback(self):
        """Test the improve method without feedback."""
        result = self.critic.improve("Text to improve")
        self.assertEqual(result, "Improved text")
        self.mock_critique_service.improve.assert_called_once_with("Text to improve", "Please improve this text.")

    def test_improve_empty_text(self):
        """Test the improve method with empty text."""
        with self.assertRaises(ValueError):
            self.critic.improve("")

    def test_critique(self):
        """Test the critique method."""
        result = self.critic.critique("Text to critique")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good text")
        self.mock_critique_service.critique.assert_called_once_with("Text to critique")

    def test_critique_empty_text(self):
        """Test the critique method with empty text."""
        with self.assertRaises(ValueError):
            self.critic.critique("")

    @patch('asyncio.run')
    def test_async_methods(self, mock_run):
        """Test the async methods."""
        mock_run.return_value = True

        # Setup async methods
        self.mock_critique_service.avalidate.return_value = True
        self.mock_critique_service.acritique.return_value = {
            "score": 0.8,
            "feedback": "Good text",
            "issues": [],
            "suggestions": []
        }
        self.mock_critique_service.aimprove.return_value = "Improved text"

        # Test avalidate
        self.critic.avalidate("Text to validate")
        self.mock_critique_service.avalidate.assert_called_once_with("Text to validate")

        # Test acritique
        self.critic.acritique("Text to critique")
        self.mock_critique_service.acritique.assert_called_once_with("Text to critique")

        # Test aimprove with string feedback
        self.critic.aimprove("Text to improve", "Feedback")
        self.mock_critique_service.aimprove.assert_called_once_with("Text to improve", "Feedback")

        # Reset mock
        self.mock_critique_service.aimprove.reset_mock()

        # Test aimprove with violations feedback
        # Mock the violations_to_feedback method
        original_method = self.critic._violations_to_feedback
        self.critic._violations_to_feedback = MagicMock(return_value="Formatted feedback")

        # Call the method with violations
        violations = [{"rule_name": "Rule1", "message": "Violation 1"}]
        self.critic.aimprove("Text to improve", violations)

        # Check the result
        self.critic._violations_to_feedback.assert_called_once_with(violations)
        self.mock_critique_service.aimprove.assert_called_once_with("Text to improve", "Formatted feedback")

        # Restore the original method
        self.critic._violations_to_feedback = original_method

        # Reset mock
        self.mock_critique_service.aimprove.reset_mock()

        # Test aimprove without feedback
        self.critic.aimprove("Text to improve")
        self.mock_critique_service.aimprove.assert_called_once_with("Text to improve", "Please improve this text.")


class TestCreateReflexionCritic(unittest.TestCase):
    """Tests for the create_reflexion_critic function."""

    def setUp(self):
        """Set up test environment."""
        # Create a mock LLM provider
        self.mock_llm = MagicMock()
        self.mock_llm.model_name = "test_model"

    def test_create_with_defaults(self):
        """Test creating a critic with default parameters."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import create_reflexion_critic, ReflexionCritic, ReflexionCriticConfig

        # Mock the required classes
        with patch('sifaka.critics.reflexion.ReflexionCritic') as mock_critic, \
             patch('sifaka.critics.reflexion.ReflexionCriticConfig') as mock_config:
            mock_config.return_value = MagicMock()
            mock_critic.return_value = MagicMock()

            # Call the function
            critic = create_reflexion_critic(self.mock_llm)

            # Check that the required classes were called
            mock_config.assert_called_once()
            mock_critic.assert_called_once()

    def test_create_with_custom_parameters(self):
        """Test creating a critic with custom parameters."""
        # Import here to use the mock imports
        from sifaka.critics.reflexion import create_reflexion_critic, ReflexionCritic, ReflexionCriticConfig

        # Mock the required classes
        with patch('sifaka.critics.reflexion.ReflexionCritic') as mock_critic, \
             patch('sifaka.critics.reflexion.ReflexionCriticConfig') as mock_config:
            mock_config.return_value = MagicMock()
            mock_critic.return_value = MagicMock()

            # Call the function with custom parameters
            critic = create_reflexion_critic(
                self.mock_llm,
                name="custom_critic",
                description="Custom description",
                system_prompt="Custom system prompt",
                temperature=0.3,
                max_tokens=500,
                min_confidence=0.8,
                memory_buffer_size=10,
                reflection_depth=3
            )

            # Check that the required classes were called with the right parameters
            mock_config.assert_called_once_with(
                name="custom_critic",
                description="Custom description",
                system_prompt="Custom system prompt",
                temperature=0.3,
                max_tokens=500,
                min_confidence=0.8,
                memory_buffer_size=10,
                reflection_depth=3
            )
            mock_critic.assert_called_once()


if __name__ == "__main__":
    unittest.main()