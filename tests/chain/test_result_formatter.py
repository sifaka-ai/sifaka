"""
Tests for the ResultFormatter class.
"""

import unittest
from unittest.mock import Mock

from sifaka.chain.formatters.result import ResultFormatter
from sifaka.chain.result import ChainResult
from sifaka.validation import ValidationResult, RuleResult


class TestResultFormatter(unittest.TestCase):
    """Tests for the ResultFormatter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ResultFormatter[str]()
        
    def test_format_result(self):
        """Test format_result method."""
        # Create a validation result
        rule_result = RuleResult(passed=True, message="Passed")
        validation_result = ValidationResult(
            output="Test output",
            rule_results=[rule_result],
            all_passed=True,
        )
        
        # Format the result
        result = self.formatter.format_result(
            output="Test output",
            validation_result=validation_result,
        )
        
        # Verify the result
        self.assertIsInstance(result, ChainResult)
        self.assertEqual(result.output, "Test output")
        self.assertEqual(len(result.rule_results), 1)
        self.assertTrue(result.rule_results[0].passed)
        self.assertEqual(result.rule_results[0].message, "Passed")
        self.assertIsNone(result.critique_details)
        
    def test_format_result_with_critique_details(self):
        """Test format_result method with critique details."""
        # Create a validation result
        rule_result = RuleResult(passed=True, message="Passed")
        validation_result = ValidationResult(
            output="Test output",
            rule_results=[rule_result],
            all_passed=True,
        )
        
        # Create critique details
        critique_details = {
            "score": 0.8,
            "feedback": "Good text",
            "issues": ["Minor issue"],
            "suggestions": ["Minor suggestion"],
        }
        
        # Format the result
        result = self.formatter.format_result(
            output="Test output",
            validation_result=validation_result,
            critique_details=critique_details,
        )
        
        # Verify the result
        self.assertIsInstance(result, ChainResult)
        self.assertEqual(result.output, "Test output")
        self.assertEqual(len(result.rule_results), 1)
        self.assertTrue(result.rule_results[0].passed)
        self.assertEqual(result.rule_results[0].message, "Passed")
        self.assertEqual(result.critique_details, critique_details)
        
    def test_format_feedback_from_validation(self):
        """Test format_feedback_from_validation method."""
        # Create a validation result with a failing rule
        rule_result = RuleResult(passed=False, message="Failed")
        validation_result = ValidationResult(
            output="Test output",
            rule_results=[rule_result],
            all_passed=False,
        )
        
        # Format the feedback
        feedback = self.formatter.format_feedback_from_validation(validation_result)
        
        # Verify the feedback
        self.assertEqual(feedback, "The following issues were found:\n- Failed\n")
        
    def test_format_feedback_from_validation_with_multiple_failures(self):
        """Test format_feedback_from_validation method with multiple failures."""
        # Create a validation result with multiple failing rules
        rule_results = [
            RuleResult(passed=False, message="Failed 1"),
            RuleResult(passed=False, message="Failed 2"),
            RuleResult(passed=True, message="Passed"),
        ]
        validation_result = ValidationResult(
            output="Test output",
            rule_results=rule_results,
            all_passed=False,
        )
        
        # Format the feedback
        feedback = self.formatter.format_feedback_from_validation(validation_result)
        
        # Verify the feedback
        self.assertEqual(
            feedback,
            "The following issues were found:\n- Failed 1\n- Failed 2\n"
        )
        
    def test_format_feedback_from_critique(self):
        """Test format_feedback_from_critique method."""
        # Create critique details with feedback
        critique_details = {
            "feedback": "Needs improvement",
        }
        
        # Format the feedback
        feedback = self.formatter.format_feedback_from_critique(critique_details)
        
        # Verify the feedback
        self.assertEqual(feedback, "Needs improvement")
        
    def test_format_feedback_from_critique_with_issues_and_suggestions(self):
        """Test format_feedback_from_critique method with issues and suggestions."""
        # Create critique details with issues and suggestions
        critique_details = {
            "issues": ["Issue 1", "Issue 2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"],
        }
        
        # Format the feedback
        feedback = self.formatter.format_feedback_from_critique(critique_details)
        
        # Verify the feedback
        expected_feedback = (
            "The following issues were found:\n"
            "- Issue 1\n"
            "- Issue 2\n"
            "\n"
            "Suggestions for improvement:\n"
            "- Suggestion 1\n"
            "- Suggestion 2\n"
        )
        self.assertEqual(feedback, expected_feedback)
        
    def test_format_feedback_from_critique_with_empty_details(self):
        """Test format_feedback_from_critique method with empty details."""
        # Create empty critique details
        critique_details = {}
        
        # Format the feedback
        feedback = self.formatter.format_feedback_from_critique(critique_details)
        
        # Verify the feedback
        self.assertEqual(feedback, "The following issues were found:\n")


if __name__ == "__main__":
    unittest.main()
