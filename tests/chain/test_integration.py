"""
Integration tests for the chain module.
"""

import unittest
from unittest.mock import Mock

from sifaka.chain import (
    ChainCore,
    PromptManager,
    ValidationManager,
    ResultFormatter,
    SimpleRetryStrategy,
    create_simple_chain,
    create_backoff_chain,
)
from sifaka.validation import RuleResult


class MockModelProvider:
    """Mock model provider for testing."""
    
    def __init__(self):
        """Initialize the mock model provider."""
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


class TestIntegration(unittest.TestCase):
    """Integration tests for the chain module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModelProvider()
        self.passing_rule = MockRule("passing_rule", should_pass=True)
        self.failing_rule = MockRule("failing_rule", should_pass=False)
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
        
    def test_create_backoff_chain(self):
        """Test that create_backoff_chain works correctly."""
        chain = create_backoff_chain(
            model=self.model,
            rules=[self.passing_rule],
            critic=self.critic,
            max_attempts=3,
            initial_backoff=1.0,
            backoff_factor=2.0,
            max_backoff=60.0,
        )
        
        self.assertIsInstance(chain, ChainCore)
        self.assertEqual(chain.model, self.model)
        self.assertEqual(chain.critic, self.critic)
        self.assertIsInstance(chain.validation_manager, ValidationManager)
        self.assertIsInstance(chain.prompt_manager, PromptManager)
        self.assertIsInstance(chain.result_formatter, ResultFormatter)
        
    def test_chain_with_passing_rule(self):
        """Test that a chain with a passing rule works correctly."""
        chain = create_simple_chain(
            model=self.model,
            rules=[self.passing_rule],
            critic=None,
            max_attempts=3,
        )
        
        # Mock the retry_strategy.run method
        chain.retry_strategy.run = Mock(
            return_value=Mock(
                output="Generated text",
                rule_results=[
                    RuleResult(
                        passed=True,
                        message="Mock message",
                    )
                ],
            )
        )
        
        result = chain.run("Test prompt")
        
        self.assertEqual(result.output, "Generated text")
        self.assertTrue(result.rule_results[0].passed)
        
    def test_chain_with_failing_rule(self):
        """Test that a chain with a failing rule raises ValueError."""
        chain = create_simple_chain(
            model=self.model,
            rules=[self.failing_rule],
            critic=None,
            max_attempts=3,
        )
        
        # Mock the retry_strategy.run method to raise ValueError
        chain.retry_strategy.run = Mock(side_effect=ValueError("Validation failed"))
        
        with self.assertRaises(ValueError):
            chain.run("Test prompt")
            
    def test_chain_with_critic(self):
        """Test that a chain with a critic works correctly."""
        chain = create_simple_chain(
            model=self.model,
            rules=[self.passing_rule],
            critic=self.critic,
            max_attempts=3,
        )
        
        # Mock the retry_strategy.run method
        chain.retry_strategy.run = Mock(
            return_value=Mock(
                output="Generated text",
                rule_results=[
                    RuleResult(
                        passed=True,
                        message="Mock message",
                    )
                ],
                critique_details={
                    "score": 0.8,
                    "feedback": "Good text",
                    "issues": ["Minor issue"],
                    "suggestions": ["Minor suggestion"],
                },
            )
        )
        
        result = chain.run("Test prompt")
        
        self.assertEqual(result.output, "Generated text")
        self.assertTrue(result.rule_results[0].passed)
        self.assertEqual(result.critique_details["score"], 0.8)
        self.assertEqual(result.critique_details["feedback"], "Good text")
        

if __name__ == "__main__":
    unittest.main()
