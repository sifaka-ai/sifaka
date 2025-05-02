"""
Tests for the ValidationManager class.
"""

import unittest
from unittest.mock import Mock, patch

from sifaka.chain.managers.validation import ValidationManager
from sifaka.validation import ValidationResult, RuleResult


class MockRule:
    """Mock rule for testing."""
    
    def __init__(self, name: str, should_pass: bool = True, message: str = "Mock message"):
        """Initialize the mock rule."""
        self.name = name
        self._should_pass = should_pass
        self._message = message
        
    def validate(self, text: str) -> RuleResult:
        """Mock implementation of validate."""
        return RuleResult(
            passed=self._should_pass,
            message=self._message,
        )


class TestValidationManager(unittest.TestCase):
    """Tests for the ValidationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.passing_rule = MockRule("passing_rule", should_pass=True)
        self.failing_rule = MockRule("failing_rule", should_pass=False)
        
    def test_initialization(self):
        """Test initialization."""
        # Test with a single rule
        manager = ValidationManager[str]([self.passing_rule])
        self.assertEqual(manager.rules, [self.passing_rule])
        
        # Test with multiple rules
        manager = ValidationManager[str]([self.passing_rule, self.failing_rule])
        self.assertEqual(manager.rules, [self.passing_rule, self.failing_rule])
        
        # Test with no rules
        manager = ValidationManager[str]([])
        self.assertEqual(manager.rules, [])
        
    def test_validate_with_passing_rule(self):
        """Test validate with a passing rule."""
        manager = ValidationManager[str]([self.passing_rule])
        result = manager.validate("Test text")
        
        self.assertTrue(result.rule_results[0].passed)
        self.assertEqual(result.rule_results[0].message, "Mock message")
        
    def test_validate_with_failing_rule(self):
        """Test validate with a failing rule."""
        manager = ValidationManager[str]([self.failing_rule])
        result = manager.validate("Test text")
        
        self.assertFalse(result.rule_results[0].passed)
        self.assertEqual(result.rule_results[0].message, "Mock message")
        
    def test_validate_with_multiple_rules(self):
        """Test validate with multiple rules."""
        manager = ValidationManager[str]([self.passing_rule, self.failing_rule])
        result = manager.validate("Test text")
        
        self.assertTrue(result.rule_results[0].passed)
        self.assertFalse(result.rule_results[1].passed)
        
    def test_get_error_messages(self):
        """Test get_error_messages."""
        manager = ValidationManager[str]([self.passing_rule, self.failing_rule])
        result = manager.validate("Test text")
        
        error_messages = manager.get_error_messages(result)
        self.assertEqual(len(error_messages), 1)
        self.assertEqual(error_messages[0], "Mock message")
        
    def test_get_error_messages_with_no_errors(self):
        """Test get_error_messages with no errors."""
        manager = ValidationManager[str]([self.passing_rule])
        result = manager.validate("Test text")
        
        error_messages = manager.get_error_messages(result)
        self.assertEqual(len(error_messages), 0)
        
    def test_get_error_messages_with_custom_message(self):
        """Test get_error_messages with a custom message."""
        failing_rule = MockRule("failing_rule", should_pass=False, message="Custom error message")
        manager = ValidationManager[str]([failing_rule])
        result = manager.validate("Test text")
        
        error_messages = manager.get_error_messages(result)
        self.assertEqual(len(error_messages), 1)
        self.assertEqual(error_messages[0], "Custom error message")


if __name__ == "__main__":
    unittest.main()
