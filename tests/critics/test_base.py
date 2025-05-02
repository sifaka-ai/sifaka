"""Tests for base critic functionality."""

import unittest
from unittest.mock import MagicMock
from typing import Dict, Any, List

from sifaka.critics.base import (
    CriticConfig,
    CriticMetadata,
    CriticOutput,
    CriticResult,
    BaseCritic,
    create_critic,
)


class MockCritic(BaseCritic):
    """Mock critic for testing."""

    def validate(self, text: str) -> bool:
        """Mock implementation of validate."""
        return len(text) > 10

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Mock implementation of improve."""
        return f"Improved: {text}"

    def critique(self, text: str) -> CriticMetadata:
        """Mock implementation of critique."""
        score = 0.9 if len(text) > 10 else 0.5
        return CriticMetadata(
            score=score,
            feedback="Test feedback",
            issues=["Test issue"] if score < 0.7 else [],
            suggestions=["Test suggestion"] if score < 0.7 else [],
            attempt_number=1,
            processing_time_ms=100.0
        )


class TestCriticConfig(unittest.TestCase):
    """Tests for CriticConfig."""

    def test_valid_config(self):
        """Test valid configuration initialization."""
        config = CriticConfig(
            name="test_critic",
            description="Test critic configuration",
            min_confidence=0.7,
            max_attempts=3,
            cache_size=100,
            priority=1,
            cost=1.0
        )
        self.assertEqual(config.name, "test_critic")
        self.assertEqual(config.description, "Test critic configuration")
        self.assertEqual(config.min_confidence, 0.7)
        self.assertEqual(config.max_attempts, 3)
        self.assertEqual(config.cache_size, 100)
        self.assertEqual(config.priority, 1)
        self.assertEqual(config.cost, 1.0)

    def test_invalid_config(self):
        """Test invalid configuration initialization."""
        # Test empty name
        with self.assertRaises(ValueError):
            CriticConfig(name="", description="Test")

        # Test empty description
        with self.assertRaises(ValueError):
            CriticConfig(name="test", description="")

        # Test invalid min_confidence
        with self.assertRaises(ValueError):
            CriticConfig(
                name="test",
                description="Test",
                min_confidence=1.5
            )

        # Test invalid max_attempts
        with self.assertRaises(ValueError):
            CriticConfig(
                name="test",
                description="Test",
                max_attempts=0
            )

        # Test invalid cache_size
        with self.assertRaises(ValueError):
            CriticConfig(
                name="test",
                description="Test",
                cache_size=-1
            )

        # Test invalid priority
        with self.assertRaises(ValueError):
            CriticConfig(
                name="test",
                description="Test",
                priority=-1
            )

        # Test invalid cost
        with self.assertRaises(ValueError):
            CriticConfig(
                name="test",
                description="Test",
                cost=-1
            )


class TestCriticMetadata(unittest.TestCase):
    """Tests for CriticMetadata."""

    def test_valid_metadata(self):
        """Test valid metadata initialization."""
        metadata = CriticMetadata(
            score=0.8,
            feedback="Test feedback",
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            attempt_number=1,
            processing_time_ms=100.0
        )
        self.assertEqual(metadata.score, 0.8)
        self.assertEqual(metadata.feedback, "Test feedback")
        self.assertEqual(metadata.issues, ["Test issue"])
        self.assertEqual(metadata.suggestions, ["Test suggestion"])
        self.assertEqual(metadata.attempt_number, 1)
        self.assertEqual(metadata.processing_time_ms, 100.0)

    def test_invalid_metadata(self):
        """Test invalid metadata initialization."""
        # Test invalid score
        with self.assertRaises(ValueError):
            CriticMetadata(score=1.5, feedback="Test")

        # Test invalid attempt_number
        with self.assertRaises(ValueError):
            CriticMetadata(score=0.8, feedback="Test", attempt_number=0)

        # Test invalid processing_time
        with self.assertRaises(ValueError):
            CriticMetadata(score=0.8, feedback="Test", processing_time_ms=-1)


class TestBaseCritic(unittest.TestCase):
    """Tests for BaseCritic."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CriticConfig(
            name="test_critic",
            description="Test critic configuration",
            min_confidence=0.7,
            max_attempts=3,
            cache_size=100,
            priority=1,
            cost=1.0
        )
        self.critic = MockCritic(self.config)

    def test_critic_initialization(self):
        """Test critic initialization."""
        self.assertEqual(self.critic.config, self.config)

    def test_invalid_critic_initialization(self):
        """Test invalid critic initialization."""
        with self.assertRaises(TypeError):
            BaseCritic(MagicMock())  # type: ignore

    def test_process_valid_text(self):
        """Test processing valid text."""
        text = "This is a long enough text"
        result = self.critic.process(text, [])
        self.assertEqual(result.result, CriticResult.SUCCESS)
        self.assertEqual(result.improved_text, text)
        self.assertEqual(result.metadata.score, 0.9)

    def test_process_text_needing_improvement(self):
        """Test processing text that needs improvement."""
        text = "Too short"
        result = self.critic.process(text, [])
        self.assertEqual(result.result, CriticResult.NEEDS_IMPROVEMENT)
        self.assertEqual(result.improved_text, f"Improved: {text}")
        self.assertEqual(result.metadata.score, 0.5)

    def test_process_invalid_text(self):
        """Test processing invalid text."""
        with self.assertRaises(ValueError):
            self.critic.process("", [])

        with self.assertRaises(ValueError):
            self.critic.process("   ", [])


class TestCreateCritic(unittest.TestCase):
    """Tests for create_critic factory function."""

    def test_create_critic(self):
        """Test creating a critic with factory function."""
        critic = create_critic(
            MockCritic,
            name="test_critic",
            description="Test critic",
            min_confidence=0.8,
            max_attempts=5,
            cache_size=200,
            priority=2,
            cost=2.0
        )
        self.assertIsInstance(critic, MockCritic)
        self.assertEqual(critic.config.name, "test_critic")
        self.assertEqual(critic.config.description, "Test critic")
        self.assertEqual(critic.config.min_confidence, 0.8)
        self.assertEqual(critic.config.max_attempts, 5)
        self.assertEqual(critic.config.cache_size, 200)
        self.assertEqual(critic.config.priority, 2)
        self.assertEqual(critic.config.cost, 2.0)

    def test_create_critic_with_defaults(self):
        """Test creating a critic with default values."""
        critic = create_critic(
            MockCritic,
            name="test_critic",
            description="Test critic"
        )
        self.assertIsInstance(critic, MockCritic)
        self.assertEqual(critic.config.min_confidence, 0.7)  # Default value
        self.assertEqual(critic.config.max_attempts, 3)  # Default value
        self.assertEqual(critic.config.cache_size, 100)  # Default value
        self.assertEqual(critic.config.priority, 1)  # Default value
        self.assertEqual(critic.config.cost, 1.0)  # Default value


if __name__ == "__main__":
    unittest.main()