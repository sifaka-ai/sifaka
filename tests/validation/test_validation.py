"""Tests for the validation module."""

import unittest
from unittest.mock import MagicMock, patch
from typing import List, Optional
from sifaka.validation import Validator, ValidationResult
from sifaka.rules import Rule, RuleResult, RuleConfig
from sifaka.rules.base import BaseValidator
from tests.base.test_base import BaseTestCase


class MockValidator(BaseValidator[str]):
    """Mock validator for testing."""

    def __init__(self, passes: bool, message: str = ""):
        """Initialize mock validator."""
        self._passes = passes
        self._message = message

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Mock validate method."""
        return RuleResult(
            passed=self._passes,
            message=self._message or ("Validation passed" if self._passes else "Validation failed")
        )

    @property
    def validation_type(self) -> type[str]:
        """Get validation type."""
        return str


class MockRule(Rule[str, RuleResult, MockValidator, None]):
    """Mock rule for testing."""

    def __init__(self, name: str, passes: bool, message: str = ""):
        """Initialize mock rule."""
        super().__init__(
            name=name,
            description=f"Mock rule: {name}",
            config=RuleConfig(),
            validator=MockValidator(passes, message)
        )

    def _create_default_validator(self) -> MockValidator:
        """Create default validator."""
        return MockValidator(True)


class TestValidator(BaseTestCase):
    """Tests for Validator class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.rules = [
            MockRule("rule1", True),
            MockRule("rule2", True),
            MockRule("rule3", True)
        ]
        self.validator = Validator(self.rules)

    def test_initialization(self):
        """Test validator initialization."""
        self.assertEqual(len(self.validator.rules), 3)
        self.assertIsInstance(self.validator.rules[0], MockRule)

    def test_validate_all_passed(self):
        """Test validation when all rules pass."""
        output = "test output"
        result = self.validator.validate(output)

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.output, output)
        self.assertEqual(len(result.rule_results), 3)
        self.assertTrue(result.all_passed)
        self.assertTrue(all(r.passed for r in result.rule_results))

    def test_validate_some_failed(self):
        """Test validation when some rules fail."""
        rules = [
            MockRule("rule1", True),
            MockRule("rule2", False, "Rule 2 failed"),
            MockRule("rule3", True)
        ]
        validator = Validator(rules)
        output = "test output"
        result = validator.validate(output)

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.output, output)
        self.assertEqual(len(result.rule_results), 3)
        self.assertFalse(result.all_passed)
        self.assertEqual(result.rule_results[1].message, "Rule 2 failed")

    def test_validate_all_failed(self):
        """Test validation when all rules fail."""
        rules = [
            MockRule("rule1", False, "Rule 1 failed"),
            MockRule("rule2", False, "Rule 2 failed"),
            MockRule("rule3", False, "Rule 3 failed")
        ]
        validator = Validator(rules)
        output = "test output"
        result = validator.validate(output)

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.output, output)
        self.assertEqual(len(result.rule_results), 3)
        self.assertFalse(result.all_passed)
        self.assertFalse(any(r.passed for r in result.rule_results))

    def test_get_error_messages(self):
        """Test getting error messages from validation results."""
        rules = [
            MockRule("rule1", True),
            MockRule("rule2", False, "Rule 2 failed"),
            MockRule("rule3", False, "Rule 3 failed")
        ]
        validator = Validator(rules)
        output = "test output"
        result = validator.validate(output)

        error_messages = validator.get_error_messages(result)
        self.assertEqual(len(error_messages), 2)
        self.assertIn("Rule 2 failed", error_messages)
        self.assertIn("Rule 3 failed", error_messages)
        self.assertNotIn("Rule 1 failed", error_messages)

    def test_get_error_messages_no_errors(self):
        """Test getting error messages when there are no errors."""
        output = "test output"
        result = self.validator.validate(output)
        error_messages = self.validator.get_error_messages(result)
        self.assertEqual(len(error_messages), 0)

    def test_validation_with_empty_rules(self):
        """Test validation with empty rules list."""
        validator = Validator([])
        output = "test output"
        result = validator.validate(output)

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.output, output)
        self.assertEqual(len(result.rule_results), 0)
        self.assertTrue(result.all_passed)

    def test_validation_with_complex_output(self):
        """Test validation with complex output type."""
        class ComplexOutput:
            def __init__(self, value: str):
                self.value = value

            def __str__(self):
                return self.value

        class ComplexValidator(BaseValidator[ComplexOutput]):
            """Validator for complex output."""

            def validate(self, output: ComplexOutput, **kwargs) -> RuleResult:
                """Mock validate method."""
                return RuleResult(
                    passed=True,
                    message="Complex validation passed"
                )

            @property
            def validation_type(self) -> type[ComplexOutput]:
                """Get validation type."""
                return ComplexOutput

        class ComplexRule(Rule[ComplexOutput, RuleResult, ComplexValidator, None]):
            """Rule for complex output."""

            def __init__(self, name: str, passes: bool, message: str = ""):
                """Initialize complex rule."""
                super().__init__(
                    name=name,
                    description=f"Complex rule: {name}",
                    config=RuleConfig(),
                    validator=ComplexValidator()
                )

            def _create_default_validator(self) -> ComplexValidator:
                """Create default validator."""
                return ComplexValidator()

        rules = [
            ComplexRule("rule1", True),
            ComplexRule("rule2", True)
        ]
        validator = Validator(rules)
        output = ComplexOutput("test value")
        result = validator.validate(output)

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.output, output)
        self.assertEqual(len(result.rule_results), 2)
        self.assertTrue(result.all_passed)
        self.assertTrue(all(r.passed for r in result.rule_results))


if __name__ == "__main__":
    unittest.main()