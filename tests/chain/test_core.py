"""
Tests for the core chain implementation.
"""

import unittest
from unittest.mock import Mock

from sifaka.chain.core import ChainCore
from sifaka.chain.factories import create_simple_chain
from sifaka.chain.formatters.result import ResultFormatter
from sifaka.chain.managers.prompt import PromptManager
from sifaka.chain.managers.validation import ValidationManager
from sifaka.chain.result import ChainResult
from sifaka.chain.strategies.retry import SimpleRetryStrategy
from sifaka.validation import RuleResult


class MockModelProvider:
    """Mock model provider for testing."""

    def setUp(self):
        """Set up mock model provider."""
        self.generate = Mock(return_value="Generated text")
        self.model_name = "mock-model"

    def generate(self, prompt: str) -> str:
        """Mock implementation of generate."""
        return "Generated text"

    def count_tokens(self, text: str) -> int:
        """Mock implementation of count_tokens."""
        return 10


class MockRule:
    """Mock rule for testing."""

    def __init__(self):
        """Initialize mock rule."""
        self._should_pass = True
        self.name = ""

    def setUp(self, name: str, should_pass: bool = True):
        """Set up mock rule."""
        self.name = name
        self._should_pass = should_pass

    def validate(self, text: str) -> RuleResult:
        """Mock implementation of validate."""
        return RuleResult(
            passed=self._should_pass,
            message="Mock message",
        )


class MockCritic:
    """Mock critic for testing."""

    def setUp(self):
        """Set up mock critic."""
        self.critique = Mock(
            return_value={
                "score": 0.8,
                "feedback": "Good text",
                "issues": ["Minor issue"],
                "suggestions": ["Minor suggestion"],
            }
        )
        self.improve = Mock(return_value="Improved text")

    def critique(self, text: str):
        """Mock implementation of critique."""
        return {
            "score": 0.8,
            "feedback": "Good text",
            "issues": ["Minor issue"],
            "suggestions": ["Minor suggestion"],
        }

    def improve(self, text: str, violations):
        """Mock implementation of improve."""
        return "Improved text"


class TestChainCore(unittest.TestCase):
    """Tests for the ChainCore class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModelProvider()
        self.model.setUp()
        self.passing_rule = MockRule()
        self.passing_rule.setUp("passing_rule", should_pass=True)
        self.failing_rule = MockRule()
        self.failing_rule.setUp("failing_rule", should_pass=False)
        self.validation_manager = ValidationManager[str]([self.passing_rule])
        self.prompt_manager = PromptManager()
        self.retry_strategy = SimpleRetryStrategy[str](max_attempts=3)
        self.result_formatter = ResultFormatter[str]()
        self.critic = MockCritic()
        self.critic.setUp()

        self.chain = ChainCore[str](
            model=self.model,
            validation_manager=self.validation_manager,
            prompt_manager=self.prompt_manager,
            retry_strategy=self.retry_strategy,
            result_formatter=self.result_formatter,
            critic=self.critic,
        )

    def test_initialization(self):
        """Test that the chain initializes correctly."""
        self.assertEqual(self.chain.model, self.model)
        self.assertEqual(self.chain.validation_manager, self.validation_manager)
        self.assertEqual(self.chain.prompt_manager, self.prompt_manager)
        self.assertEqual(self.chain.retry_strategy, self.retry_strategy)
        self.assertEqual(self.chain.result_formatter, self.result_formatter)
        self.assertEqual(self.chain.critic, self.critic)

    def test_run_with_passing_rule(self):
        """Test that run works correctly with a passing rule."""
        # Mock the retry_strategy.run method
        self.retry_strategy.run = Mock(
            return_value=ChainResult(
                output="Generated text",
                rule_results=[
                    RuleResult(
                        passed=True,
                        message="Mock message",
                    )
                ],
            )
        )

        result = self.chain.run("Test prompt")

        self.assertEqual(result.output, "Generated text")
        self.assertTrue(result.rule_results[0].passed)

    def test_run_with_failing_rule(self):
        """Test that run works correctly with a failing rule."""
        # Create a new chain with failing rule
        validation_manager = ValidationManager[str]([self.failing_rule])
        chain = ChainCore[str](
            model=self.model,
            validation_manager=validation_manager,
            prompt_manager=self.prompt_manager,
            retry_strategy=self.retry_strategy,
            result_formatter=self.result_formatter,
            critic=self.critic,
        )

        # Mock the retry_strategy.run method to raise ValueError
        self.retry_strategy.run = Mock(side_effect=ValueError("Validation failed"))

        with self.assertRaises(ValueError):
            chain.run("Test prompt")


class TestFactories(unittest.TestCase):
    """Tests for the factory functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModelProvider()
        self.passing_rule = MockRule()
        self.critic = MockCritic()

    def test_create_simple_chain(self):
        """Test that create_simple_chain works correctly."""
        chain = create_simple_chain(
            model=self.model,
            rules=[self.passing_rule],
            critic=self.critic,
            max_attempts=3,
        )

        self.assertIsInstance(chain, ChainCore)
        self.assertEqual(chain.model, self.model)
        self.assertEqual(chain.critic, self.critic)
        self.assertIsInstance(chain.validation_manager, ValidationManager)
        self.assertIsInstance(chain.prompt_manager, PromptManager)
        self.assertIsInstance(chain.retry_strategy, SimpleRetryStrategy)
        self.assertIsInstance(chain.result_formatter, ResultFormatter)
        self.assertEqual(chain.retry_strategy.max_attempts, 3)


if __name__ == "__main__":
    unittest.main()
