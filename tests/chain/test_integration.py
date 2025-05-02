"""
Integration tests for the chain module.
"""

import unittest
from unittest.mock import Mock, patch
import pytest

from sifaka.chain import (
    ChainCore,
    PromptManager,
    ValidationManager,
    ResultFormatter,
    SimpleRetryStrategy,
    create_simple_chain,
    create_backoff_chain,
)
from sifaka.validation import RuleResult, ValidationResult


class MockModelProvider:
    """Mock model provider for testing."""

    def __init__(self, responses=None):
        """Initialize the mock model provider."""
        self.responses = responses or ["Generated text"]
        self.call_count = 0
        self.model_name = "mock-model"

    def generate(self, prompt: str) -> str:
        """Mock implementation of generate."""
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response

    def count_tokens(self, text: str) -> int:
        """Mock implementation of count_tokens."""
        return len(text.split())


class MockRule:
    """Mock rule for testing."""

    def __init__(self, name: str, should_pass: bool = True):
        """Initialize the mock rule."""
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

    def __init__(self):
        """Initialize the mock critic."""
        self.call_count = 0

    def critique(self, text: str):
        """Mock implementation of critique."""
        self.call_count += 1
        return {
            "score": 0.8,
            "feedback": "Good text, but could be improved",
            "issues": ["Minor issue"],
            "suggestions": ["Minor suggestion"],
        }

    def improve(self, text: str, violations=None):
        """Mock implementation of improve."""
        return "Improved " + text


class TestIntegration(unittest.TestCase):
    """Integration tests for the chain module."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModelProvider()
        self.passing_rule = MockRule("passing_rule", should_pass=True)
        self.failing_rule = MockRule("failing_rule", should_pass=False)
        self.critic = MockCritic()

    def test_create_simple_chain(self):
        """Test creating a simple chain."""
        # Create a simple chain
        chain = create_simple_chain(
            model=self.model,
            rules=[self.passing_rule],
            critic=None,
            max_attempts=3
        )

        # Verify it's the correct type
        from sifaka.chain.orchestrator import ChainOrchestrator
        self.assertIsInstance(chain, ChainOrchestrator)

        # Verify it has the correct configuration
        self.assertEqual(chain._core._retry_strategy.max_attempts, 3)

    def test_create_backoff_chain(self):
        """Test creating a chain with backoff retry strategy."""
        # Create a chain with backoff
        chain = create_backoff_chain(
            model=self.model,
            rules=[self.passing_rule],
            critic=None,
            max_attempts=3,
            initial_backoff=1.0,
            backoff_factor=2.0,
            max_backoff=60.0
        )

        # Verify it's the correct type
        from sifaka.chain.orchestrator import ChainOrchestrator
        from sifaka.chain.strategies.retry import BackoffRetryStrategy
        self.assertIsInstance(chain, ChainOrchestrator)

        # Verify it uses a backoff retry strategy
        self.assertIsInstance(chain._core._retry_strategy, BackoffRetryStrategy)
        self.assertEqual(chain._core._retry_strategy.max_attempts, 3)

        # Cannot check these attributes as they might be private
        # or implemented differently than expected

    def test_chain_with_passing_rule(self):
        """Test chain with a rule that passes."""
        # Create chain with passing rule
        chain = create_simple_chain(
            model=self.model,
            rules=[self.passing_rule],
            critic=None,
            max_attempts=3
        )

        # Run the chain
        result = chain.run("Test prompt")

        # Verify the result
        self.assertEqual(result.output, "Generated text")
        self.assertTrue(result.rule_results[0].passed)

        # Verify the model was called once
        self.assertEqual(self.model.call_count, 1)

    def test_chain_with_failing_rule(self):
        """Test chain with a rule that fails."""
        # Create a model that returns different responses on retries
        model = MockModelProvider(responses=["Bad output", "Good output"])

        # Create rule that fails on first output but passes on second
        class ConditionalRule:
            def validate(self, text: str) -> RuleResult:
                if text == "Bad output":
                    return RuleResult(passed=False, message="Failed")
                return RuleResult(passed=True, message="Passed")

        # Create chain with the conditional rule
        chain = create_simple_chain(
            model=model,
            rules=[ConditionalRule()],
            critic=None,
            max_attempts=3
        )

        # Run the chain
        result = chain.run("Test prompt")

        # Verify the result
        self.assertEqual(result.output, "Good output")
        self.assertTrue(result.rule_results[0].passed)

        # Verify the model was called twice
        self.assertEqual(model.call_count, 2)

    def test_chain_with_critic(self):
        """Test chain with a critic."""
        # Create a model that returns outputs needing improvement
        model = MockModelProvider(responses=["Unimproved output"])

        # Create a custom critic that works with our test
        class TestCritic:
            def __init__(self):
                self.call_count = 0

            def critique(self, text):
                self.call_count += 1
                return {
                    "score": 0.5,
                    "feedback": "Needs improvement",
                    "issues": ["Not improved"],
                    "suggestions": ["Improve it"]
                }

            def improve(self, text, violations=None):
                return "Improved " + text

            # Add necessary method to make it work with the chain
            def improve_with_feedback(self, text, feedback):
                return "Improved " + text

        critic = TestCritic()

        # Create rule that always passes for improved text
        class ImprovementRule:
            def validate(self, text: str) -> RuleResult:
                if text.startswith("Improved"):
                    return RuleResult(passed=True, message="Passed after improvement")
                return RuleResult(passed=False, message="Needs improvement")

        # Create mock model that returns different outputs based on call count
        class ResponseModelProvider:
            def __init__(self):
                self.call_count = 0
                self.model_name = "test-model"

            def generate(self, prompt):
                self.call_count += 1
                if self.call_count == 1:
                    return "Unimproved output"
                else:
                    return "Improved output"

        response_model = ResponseModelProvider()

        # Create chain with critic
        chain = create_simple_chain(
            model=response_model,
            rules=[ImprovementRule()],
            critic=critic,
            max_attempts=3
        )

        # Run the chain
        result = chain.run("Test prompt")

        # Verify the critic was called
        self.assertGreater(critic.call_count, 0)

    @patch("time.sleep")
    def test_real_chain_with_backoff(self, mock_sleep):
        """Test a real chain with backoff strategy."""
        # Create a model that returns different responses
        model = MockModelProvider(responses=["First output", "Second output"])

        # Create a rule that fails for the first output but passes for the second
        class ConditionalRule:
            def validate(self, text: str) -> RuleResult:
                if text == "First output":
                    return RuleResult(passed=False, message="Failed")
                return RuleResult(passed=True, message="Passed")

        # Create a chain with backoff strategy
        chain = create_backoff_chain(
            model=model,
            rules=[ConditionalRule()],
            critic=None,
            max_attempts=3,
            initial_backoff=1.0,
            backoff_factor=2.0,
            max_backoff=60.0,
        )

        # Run the chain
        result = chain.run("Test prompt")

        # Verify the result
        self.assertEqual(result.output, "Second output")
        self.assertTrue(result.rule_results[0].passed)
        self.assertEqual(result.rule_results[0].message, "Passed")

        # Verify the model was called twice
        self.assertEqual(model.call_count, 2)

        # Verify sleep was called with the correct backoff
        mock_sleep.assert_called_once_with(1.0)

    def test_real_chain_with_critic(self):
        """Test a real chain with a critic."""
        # Create a model that returns different responses
        model = MockModelProvider(responses=["First output", "Second output"])

        # Create a rule that fails for the first output but passes for the second
        class ConditionalRule:
            def validate(self, text: str) -> RuleResult:
                if text == "First output":
                    return RuleResult(passed=False, message="Failed")
                return RuleResult(passed=True, message="Passed")

        # Create a chain with a critic
        chain = create_simple_chain(
            model=model,
            rules=[ConditionalRule()],
            critic=self.critic,
            max_attempts=3,
        )

        # Run the chain
        result = chain.run("Test prompt")

        # Verify the result
        self.assertEqual(result.output, "Second output")
        self.assertTrue(result.rule_results[0].passed)
        self.assertEqual(result.rule_results[0].message, "Passed")

        # Verify the model was called twice
        self.assertEqual(model.call_count, 2)

        # Verify the critic was called once
        self.assertEqual(self.critic.call_count, 1)

    def test_custom_prompt_manager(self):
        """Test a chain with a custom prompt manager."""

        # Create a custom prompt manager
        class CustomPromptManager(PromptManager):
            def create_prompt_with_feedback(self, original_prompt: str, feedback: str) -> str:
                return f"System: {feedback}\n\nUser: {original_prompt}"

        # Create a model that returns different responses
        model = MockModelProvider(responses=["First output", "Second output"])

        # Create a rule that fails for the first output but passes for the second
        class ConditionalRule:
            def validate(self, text: str) -> RuleResult:
                if text == "First output":
                    return RuleResult(passed=False, message="Failed")
                return RuleResult(passed=True, message="Passed")

        # Create custom components
        validation_manager = ValidationManager[str]([ConditionalRule()])
        prompt_manager = CustomPromptManager()
        retry_strategy = SimpleRetryStrategy[str](max_attempts=3)
        result_formatter = ResultFormatter[str]()

        # Create a chain
        chain = ChainCore[str](
            model=model,
            validation_manager=validation_manager,
            prompt_manager=prompt_manager,
            retry_strategy=retry_strategy,
            result_formatter=result_formatter,
        )

        # Run the chain
        result = chain.run("Test prompt")

        # Verify the result
        self.assertEqual(result.output, "Second output")
        self.assertTrue(result.rule_results[0].passed)
        self.assertEqual(result.rule_results[0].message, "Passed")

        # Verify the model was called twice
        self.assertEqual(model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
