"""
Tests for the compatibility layer.
"""

import unittest
from unittest.mock import Mock

from sifaka.chain import Chain
from sifaka.chain.result import ChainResult
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


class TestCompat(unittest.TestCase):
    """Tests for the compatibility layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModelProvider()
        self.passing_rule = MockRule("passing_rule", should_pass=True)
        self.failing_rule = MockRule("failing_rule", should_pass=False)
        self.critic = MockCritic()
        
        self.chain = Chain(
            model=self.model,
            rules=[self.passing_rule],
            critic=self.critic,
            max_attempts=3,
        )
        
    def test_initialization(self):
        """Test that the chain initializes correctly."""
        self.assertEqual(self.chain.model, self.model)
        self.assertEqual(self.chain.rules, [self.passing_rule])
        self.assertEqual(self.chain.critic, self.critic)
        self.assertEqual(self.chain.max_attempts, 3)
        
    def test_run(self):
        """Test that run works correctly."""
        # Mock the _chain.run method
        self.chain._chain.run = Mock(
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


if __name__ == "__main__":
    unittest.main()
