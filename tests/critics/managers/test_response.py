"""
Tests for the ResponseParser class.

This module contains tests for the ResponseParser class, which is responsible
for parsing responses from language models.
"""

import unittest
from typing import Dict, Any

from sifaka.critics.managers.response import ResponseParser


class TestResponseParser(unittest.TestCase):
    """Tests for the ResponseParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = ResponseParser()

    # Tests for parse_validation_response
    def test_parse_validation_response_dict_true(self):
        """Test parsing a validation response dict with true value."""
        response = {"valid": True}
        result = self.parser.parse_validation_response(response)
        self.assertTrue(result)

    def test_parse_validation_response_dict_false(self):
        """Test parsing a validation response dict with false value."""
        response = {"valid": False}
        result = self.parser.parse_validation_response(response)
        self.assertFalse(result)

    def test_parse_validation_response_dict_missing_valid(self):
        """Test parsing a validation response dict with missing valid key."""
        response = {"other_field": True}
        result = self.parser.parse_validation_response(response)
        self.assertFalse(result)

    def test_parse_validation_response_str_true(self):
        """Test parsing a validation response string with true value."""
        response = "VALID: true\nREASON: Good content"
        result = self.parser.parse_validation_response(response)
        self.assertTrue(result)

    def test_parse_validation_response_str_true_no_space(self):
        """Test parsing a validation response string with true value (no space)."""
        response = "VALID:true\nREASON: Good content"
        result = self.parser.parse_validation_response(response)
        self.assertTrue(result)

    def test_parse_validation_response_str_true_case_insensitive(self):
        """Test parsing a validation response string with true value (case insensitive)."""
        response = "valid: TRUE\nreason: Good content"
        result = self.parser.parse_validation_response(response)
        self.assertTrue(result)

    def test_parse_validation_response_str_false(self):
        """Test parsing a validation response string with false value."""
        response = "VALID: false\nREASON: Bad content"
        result = self.parser.parse_validation_response(response)
        self.assertFalse(result)

    def test_parse_validation_response_str_false_no_space(self):
        """Test parsing a validation response string with false value (no space)."""
        response = "VALID:false\nREASON: Bad content"
        result = self.parser.parse_validation_response(response)
        self.assertFalse(result)

    def test_parse_validation_response_str_false_case_insensitive(self):
        """Test parsing a validation response string with false value (case insensitive)."""
        response = "valid: FALSE\nreason: Bad content"
        result = self.parser.parse_validation_response(response)
        self.assertFalse(result)

    def test_parse_validation_response_malformed_string(self):
        """Test parsing a malformed validation response string."""
        response = "This is not a valid response"
        result = self.parser.parse_validation_response(response)
        self.assertFalse(result)

    def test_parse_validation_response_non_dict_non_str(self):
        """Test parsing a non-dict, non-str response."""
        response = 42  # type: ignore
        result = self.parser.parse_validation_response(response)
        self.assertFalse(result)

    # Tests for parse_critique_response
    def test_parse_critique_response_dict(self):
        """Test parsing a critique response dict."""
        response = {
            "score": 0.8,
            "feedback": "Good text",
            "issues": ["Minor grammar issue"],
            "suggestions": ["Fix grammar"]
        }
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good text")
        self.assertEqual(result["issues"], ["Minor grammar issue"])
        self.assertEqual(result["suggestions"], ["Fix grammar"])

    def test_parse_critique_response_dict_missing_fields(self):
        """Test parsing a critique response dict with missing fields."""
        response = {"score": 0.7}
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.7)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

    def test_parse_critique_response_dict_incorrect_types(self):
        """Test parsing a critique response dict with incorrect types."""
        response = {
            "score": "0.8",  # String instead of float
            "feedback": 42,  # Int instead of string
            "issues": "Issue",  # String instead of list
            "suggestions": "Suggestion"  # String instead of list - use a string which is iterable
        }
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.8)  # Should convert to float
        self.assertEqual(result["feedback"], "42")  # Should convert to string
        self.assertEqual(result["issues"], list("Issue"))  # ['I', 's', 's', 'u', 'e']
        self.assertEqual(result["suggestions"], list("Suggestion"))  # Each character becomes an item

    def test_parse_critique_response_string_full(self):
        """Test parsing a full critique response string."""
        response = """
        SCORE: 0.8
        FEEDBACK: This is good feedback.
        ISSUES:
        - Issue 1
        - Issue 2
        SUGGESTIONS:
        - Suggestion 1
        - Suggestion 2
        """
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "This is good feedback.")
        self.assertEqual(result["issues"], ["Issue 1", "Issue 2"])
        self.assertEqual(result["suggestions"], ["Suggestion 1", "Suggestion 2"])

    def test_parse_critique_response_string_score_only(self):
        """Test parsing a critique response string with score only."""
        response = "SCORE: 0.7"
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.7)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

    def test_parse_critique_response_string_score_and_feedback(self):
        """Test parsing a critique response string with score and feedback."""
        response = """
        SCORE: 0.6
        FEEDBACK: Some feedback text
        """
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.6)
        self.assertEqual(result["feedback"], "Some feedback text")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

    def test_parse_critique_response_string_score_out_of_range(self):
        """Test parsing a critique response string with score out of range."""
        # Score above 1.0
        response = "SCORE: 1.5\nFEEDBACK: Some feedback"
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 1.0)  # Should be clamped to 1.0

        # Score below 0.0
        response = "SCORE: -0.5\nFEEDBACK: Some feedback"
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.0)  # Should be clamped to 0.0

    def test_parse_critique_response_string_invalid_score(self):
        """Test parsing a critique response string with invalid score."""
        response = "SCORE: not_a_number\nFEEDBACK: Some feedback"
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.0)  # Should default to 0.0

    def test_parse_critique_response_string_issues_only(self):
        """Test parsing a critique response string with issues only."""
        response = """
        ISSUES:
        - Issue 1
        - Issue 2
        """
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["issues"], ["Issue 1", "Issue 2"])
        self.assertEqual(result["suggestions"], [])

    def test_parse_critique_response_string_suggestions_only(self):
        """Test parsing a critique response string with suggestions only."""
        response = """
        SUGGESTIONS:
        - Suggestion 1
        - Suggestion 2
        """
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], ["Suggestion 1", "Suggestion 2"])

    def test_parse_critique_response_string_malformed_lists(self):
        """Test parsing a critique response string with malformed lists."""
        response = """
        SCORE: 0.7
        FEEDBACK: Some feedback
        ISSUES:
        Issue 1 (missing dash)
        - Issue 2
        SUGGESTIONS:
        Suggestion 1 (missing dash)
        - Suggestion 2
        """
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.7)
        self.assertEqual(result["feedback"], "Some feedback")
        self.assertEqual(result["issues"], ["Issue 2"])  # Only the one with dash
        self.assertEqual(result["suggestions"], ["Suggestion 2"])  # Only the one with dash

    def test_parse_critique_response_non_dict_non_str(self):
        """Test parsing a non-dict, non-str critique response."""
        response = 42  # type: ignore
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Failed to critique text", result["feedback"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(len(result["suggestions"]), 1)

    # Tests for parse_improvement_response
    def test_parse_improvement_response_dict(self):
        """Test parsing an improvement response dict."""
        response = {"improved_text": "This is improved text"}
        result = self.parser.parse_improvement_response(response)
        self.assertEqual(result, "This is improved text")

    def test_parse_improvement_response_dict_missing_field(self):
        """Test parsing an improvement response dict with missing field."""
        response = {"other_field": "Some value"}
        result = self.parser.parse_improvement_response(response)
        self.assertEqual(result, "Failed to improve text: Invalid response format")

    def test_parse_improvement_response_string_with_marker(self):
        """Test parsing an improvement response string with marker."""
        response = "Some preamble\nIMPROVED_TEXT: This is the improved text"
        result = self.parser.parse_improvement_response(response)
        self.assertEqual(result, "This is the improved text")

    def test_parse_improvement_response_string_without_marker(self):
        """Test parsing an improvement response string without marker."""
        response = "This is the improved text"
        result = self.parser.parse_improvement_response(response)
        self.assertEqual(result, "This is the improved text")

    def test_parse_improvement_response_string_with_whitespace(self):
        """Test parsing an improvement response string with whitespace."""
        response = "   This is the improved text   "
        result = self.parser.parse_improvement_response(response)
        self.assertEqual(result, "This is the improved text")

    def test_parse_improvement_response_non_dict_non_str(self):
        """Test parsing a non-dict, non-str improvement response."""
        response = 42  # type: ignore
        result = self.parser.parse_improvement_response(response)
        self.assertEqual(result, "Failed to improve text: Invalid response format")

    # Tests for parse_reflection_response
    def test_parse_reflection_response_dict(self):
        """Test parsing a reflection response dict."""
        response = {"reflection": "This is a reflection"}
        result = self.parser.parse_reflection_response(response)
        self.assertEqual(result, "This is a reflection")

    def test_parse_reflection_response_dict_missing_field(self):
        """Test parsing a reflection response dict with missing field."""
        response = {"other_field": "Some value"}
        result = self.parser.parse_reflection_response(response)
        self.assertIsNone(result)

    def test_parse_reflection_response_string_with_marker(self):
        """Test parsing a reflection response string with marker."""
        response = "Some preamble\nREFLECTION: This is the reflection"
        result = self.parser.parse_reflection_response(response)
        self.assertEqual(result, "This is the reflection")

    def test_parse_reflection_response_string_without_marker(self):
        """Test parsing a reflection response string without marker."""
        response = "This is the reflection"
        result = self.parser.parse_reflection_response(response)
        self.assertEqual(result, "This is the reflection")

    def test_parse_reflection_response_string_with_whitespace(self):
        """Test parsing a reflection response string with whitespace."""
        response = "   This is the reflection   "
        result = self.parser.parse_reflection_response(response)
        self.assertEqual(result, "This is the reflection")

    def test_parse_reflection_response_non_dict_non_str(self):
        """Test parsing a non-dict, non-str reflection response."""
        response = 42  # type: ignore
        result = self.parser.parse_reflection_response(response)
        self.assertIsNone(result)

    # Tests for _parse_critique_string
    def test_parse_critique_string_complete(self):
        """Test parsing a complete critique string."""
        response = """
        SCORE: 0.75
        FEEDBACK: This is detailed feedback.
        ISSUES:
        - Issue 1
        - Issue 2
        SUGGESTIONS:
        - Suggestion 1
        - Suggestion 2
        """
        result = self.parser._parse_critique_string(response)
        self.assertEqual(result["score"], 0.75)
        self.assertEqual(result["feedback"], "This is detailed feedback.")
        self.assertEqual(result["issues"], ["Issue 1", "Issue 2"])
        self.assertEqual(result["suggestions"], ["Suggestion 1", "Suggestion 2"])

    def test_parse_critique_string_empty(self):
        """Test parsing an empty critique string."""
        response = ""
        result = self.parser._parse_critique_string(response)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

    def test_parse_critique_string_mixed_case(self):
        """Test parsing a critique string with mixed case markers."""
        response = """
        score: 0.65
        feedback: Mixed case feedback.
        issues:
        - Issue 1
        suggestions:
        - Suggestion 1
        """
        # This should not parse correctly since the function looks for exact marker strings
        result = self.parser._parse_critique_string(response)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])


if __name__ == "__main__":
    unittest.main()