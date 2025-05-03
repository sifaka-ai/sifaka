"""
Tests for critics core module.

This module contains comprehensive tests for the CriticCore class, which is
the main interface for critics delegating to specialized components.
"""

import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from typing import Dict, List, Any

from sifaka.critics.core import CriticCore
from sifaka.critics.models import CriticConfig, CriticMetadata
from sifaka.critics.managers.memory import MemoryManager
from sifaka.critics.managers.prompt import PromptManager
from sifaka.critics.managers.response import ResponseParser


class TestCriticCore(unittest.TestCase):
    """Tests for the CriticCore class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.llm_provider = MagicMock()
        self.prompt_manager = MagicMock(spec=PromptManager)
        self.response_parser = MagicMock(spec=ResponseParser)
        self.memory_manager = MagicMock(spec=MemoryManager)
        self.config = CriticConfig(
            name="test_critic",
            description="Test critic",
            min_confidence=0.7,
            max_attempts=3
        )

        # Mock the CritiqueService
        self.mock_critique_service = MagicMock()
        with patch('sifaka.critics.core.CritiqueService', return_value=self.mock_critique_service):
            self.critic = CriticCore(
                config=self.config,
                llm_provider=self.llm_provider,
                prompt_manager=self.prompt_manager,
                response_parser=self.response_parser,
                memory_manager=self.memory_manager
            )

    def test_initialization(self):
        """Test initialization with all parameters."""
        # Verify the critic was created with the right dependencies
        self.assertEqual(self.critic._config, self.config)
        self.assertEqual(self.critic._model, self.llm_provider)
        self.assertEqual(self.critic._prompt_manager, self.prompt_manager)
        self.assertEqual(self.critic._response_parser, self.response_parser)
        self.assertEqual(self.critic._memory_manager, self.memory_manager)
        self.assertEqual(self.critic._critique_service, self.mock_critique_service)

    def test_initialization_with_defaults(self):
        """Test initialization with default managers."""
        # Mock DefaultPromptManager
        mock_default_prompt_manager = MagicMock()

        # Mock the creator methods
        with patch('sifaka.critics.core.DefaultPromptManager', return_value=mock_default_prompt_manager):
            with patch('sifaka.critics.core.ResponseParser', return_value=MagicMock()):
                with patch('sifaka.critics.core.CritiqueService', return_value=MagicMock()):
                    critic = CriticCore(
                        config=self.config,
                        llm_provider=self.llm_provider
                    )

                    # Verify default managers were created
                    self.assertEqual(critic._prompt_manager, mock_default_prompt_manager)
                    self.assertIsNotNone(critic._response_parser)
                    self.assertIsNone(critic._memory_manager)

    def test_validate(self):
        """Test validate method."""
        # Configure mock
        self.mock_critique_service.validate.return_value = True

        # Call the method
        result = self.critic.validate("test text")

        # Verify the result
        self.assertTrue(result)

        # Verify the method was called with the right parameters
        self.mock_critique_service.validate.assert_called_once_with("test text")

    def test_validate_with_empty_text(self):
        """Test validate method with empty text."""
        # Configure mock to raise ValueError for empty text
        self.mock_critique_service.validate.side_effect = ValueError("text must be a non-empty string")

        # Verify the exception is propagated
        with self.assertRaises(ValueError):
            self.critic.validate("")

    def test_improve(self):
        """Test improve method."""
        # Configure mock
        self.mock_critique_service.improve.return_value = "improved text"

        # Call the method with violations
        violations = [{"rule": "test", "message": "test message"}]
        result = self.critic.improve("test text", violations)

        # Verify the result
        self.assertEqual(result, "improved text")

        # Verify the method was called with the right parameters
        self.mock_critique_service.improve.assert_called_once_with("test text", violations)

    def test_improve_with_empty_text(self):
        """Test improve method with empty text."""
        # Configure mock to raise ValueError for empty text
        self.mock_critique_service.improve.side_effect = ValueError("text must be a non-empty string")

        # Verify the exception is propagated
        with self.assertRaises(ValueError):
            self.critic.improve("", [])

    def test_critique(self):
        """Test critique method."""
        # Configure mock
        critique_result = {
            "score": 0.8,
            "feedback": "Good text",
            "issues": ["minor issue"],
            "suggestions": ["small suggestion"]
        }
        self.mock_critique_service.critique.return_value = critique_result

        # Call the method
        result = self.critic.critique("test text")

        # Verify the result
        self.assertIsInstance(result, CriticMetadata)
        self.assertEqual(result.score, 0.8)
        self.assertEqual(result.feedback, "Good text")
        self.assertEqual(result.issues, ["minor issue"])
        self.assertEqual(result.suggestions, ["small suggestion"])

        # Verify the method was called with the right parameters
        self.mock_critique_service.critique.assert_called_once_with("test text")

    def test_critique_with_empty_values(self):
        """Test critique method with empty or missing values."""
        # Configure mock with missing values
        critique_result = {}
        self.mock_critique_service.critique.return_value = critique_result

        # Call the method
        result = self.critic.critique("test text")

        # Verify the result has default values
        self.assertIsInstance(result, CriticMetadata)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.feedback, "")
        self.assertEqual(result.issues, [])
        self.assertEqual(result.suggestions, [])

    def test_improve_with_feedback(self):
        """Test improve_with_feedback method."""
        # Configure mock
        self.mock_critique_service.improve.return_value = "improved text with feedback"

        # Call the method
        result = self.critic.improve_with_feedback("test text", "test feedback")

        # Verify the result
        self.assertEqual(result, "improved text with feedback")

        # Verify the method was called with the right parameters
        self.mock_critique_service.improve.assert_called_once_with("test text", "test feedback")


class TestCriticCoreAsync:
    """Tests for async methods of CriticCore."""

    @pytest.fixture
    def critic(self):
        """Create a critic for testing."""
        # Create mock objects
        llm_provider = MagicMock()
        prompt_manager = MagicMock(spec=PromptManager)
        response_parser = MagicMock(spec=ResponseParser)
        memory_manager = MagicMock(spec=MemoryManager)
        config = CriticConfig(
            name="test_critic",
            description="Test critic",
            min_confidence=0.7,
            max_attempts=3
        )

        # Mock the CritiqueService with AsyncMock methods
        mock_critique_service = MagicMock()
        mock_critique_service.avalidate = AsyncMock()
        mock_critique_service.acritique = AsyncMock()
        mock_critique_service.aimprove = AsyncMock()

        with patch('sifaka.critics.core.CritiqueService', return_value=mock_critique_service):
            critic = CriticCore(
                config=config,
                llm_provider=llm_provider,
                prompt_manager=prompt_manager,
                response_parser=response_parser,
                memory_manager=memory_manager
            )

        return critic

    @pytest.mark.asyncio
    async def test_avalidate(self, critic):
        """Test async validation."""
        # Configure mock
        critic._critique_service.avalidate.return_value = True

        # Call the method
        result = await critic.avalidate("test text")

        # Verify the result
        assert result is True

        # Verify the method was called with the right parameters
        critic._critique_service.avalidate.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_acritique(self, critic):
        """Test async critique."""
        # Configure mock
        critique_result = {
            "score": 0.9,
            "feedback": "Great text",
            "issues": [],
            "suggestions": ["minor suggestion"]
        }
        critic._critique_service.acritique.return_value = critique_result

        # Call the method
        result = await critic.acritique("test text")

        # Verify the result
        assert isinstance(result, CriticMetadata)
        assert result.score == 0.9
        assert result.feedback == "Great text"
        assert result.issues == []
        assert result.suggestions == ["minor suggestion"]

        # Verify the method was called with the right parameters
        critic._critique_service.acritique.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_aimprove(self, critic):
        """Test async improve."""
        # Configure mock
        critic._critique_service.aimprove.return_value = "improved text async"

        # Call the method
        violations = [{"rule": "test", "message": "test message"}]
        result = await critic.aimprove("test text", violations)

        # Verify the result
        assert result == "improved text async"

        # Verify the method was called with the right parameters
        critic._critique_service.aimprove.assert_called_once_with("test text", violations)

    @pytest.mark.asyncio
    async def test_aimprove_with_feedback(self, critic):
        """Test async improve with feedback."""
        # Configure mock
        critic._critique_service.aimprove.return_value = "improved text with feedback async"

        # Call the method
        result = await critic.aimprove_with_feedback("test text", "test feedback")

        # Verify the result
        assert result == "improved text with feedback async"

        # Verify the method was called with the right parameters
        critic._critique_service.aimprove.assert_called_once_with("test text", "test feedback")


if __name__ == "__main__":
    unittest.main()
