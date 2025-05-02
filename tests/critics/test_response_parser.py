"""
Tests for the ResponseParser class in the critics module.
"""

import unittest
from typing import Dict, Any

from sifaka.critics.managers.response import ResponseParser


class TestResponseParser(unittest.TestCase):
    """Tests for the ResponseParser class."""

    def setUp(self):
        """Set up test case."""
        self.parser = ResponseParser()

    def test_parse_validation_response_dict(self):
        """Test parsing a validation response dict."""
        # Test with valid=True
        response = {"valid": True, "reason": "Good content"}
        result = self.parser.parse_validation_response(response)
        self.assertTrue(result)

        # Test with valid=False
        response = {"valid": False, "reason": "Bad content"}
        result = self.parser.parse_validation_response(response)
        self.assertFalse(result)

    def test_parse_validation_response_string(self):
        """Test parsing a validation response string."""
        # Test with VALID: true (must be uppercase in the string)
        response = "VALID: true\nREASON: Good content"
        result = self.parser.parse_validation_response(response)
        self.assertTrue(result)

        # Test with VALID: false (must be uppercase in the string)
        response = "VALID: false\nREASON: Bad content"
        result = self.parser.parse_validation_response(response)
        self.assertFalse(result)

        # Test with invalid format
        response = "Not a valid response format"
        result = self.parser.parse_validation_response(response)
        self.assertFalse(result)

    def test_parse_critique_response_dict(self):
        """Test parsing a critique response dict."""
        response = {
            "score": 0.8,
            "feedback": "Good content",
            "issues": ["Issue 1", "Issue 2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"]
        }
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good content")
        self.assertEqual(result["issues"], ["Issue 1", "Issue 2"])
        self.assertEqual(result["suggestions"], ["Suggestion 1", "Suggestion 2"])

    def test_parse_critique_response_string(self):
        """Test parsing a critique response string."""
        response = """SCORE: 0.8
FEEDBACK: Good content
ISSUES:
- Issue 1
- Issue 2
SUGGESTIONS:
- Suggestion 1
- Suggestion 2"""
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good content")
        self.assertIn("Issue 1", result["issues"])
        self.assertIn("Issue 2", result["issues"])
        self.assertIn("Suggestion 1", result["suggestions"])
        self.assertIn("Suggestion 2", result["suggestions"])

    def test_parse_critique_response_string_incomplete(self):
        """Test parsing an incomplete critique response string."""
        # Test with only score and feedback
        response = """SCORE: 0.5
FEEDBACK: Incomplete feedback"""
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["feedback"], "Incomplete feedback")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

        # Test with only issues
        response = """ISSUES:
- Only issues here"""
        result = self.parser.parse_critique_response(response)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["feedback"], "")
        self.assertIn("Only issues here", result["issues"])
        self.assertEqual(result["suggestions"], [])

    def test_parse_critique_response_invalid(self):
        """Test parsing an invalid critique response."""
        # Test with invalid type
        result = self.parser.parse_critique_response(None)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Failed to critique", result["feedback"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(len(result["suggestions"]), 1)

    def test_parse_improvement_response_dict(self):
        """Test parsing an improvement response dict."""
        response = {"improved_text": "This is improved text"}
        result = self.parser.parse_improvement_response(response)
        self.assertEqual(result, "This is improved text")

    def test_parse_improvement_response_string(self):
        """Test parsing an improvement response string."""
        # Test with IMPROVED_TEXT: format
        response = "IMPROVED_TEXT: This is improved text"
        result = self.parser.parse_improvement_response(response)
        self.assertEqual(result, "This is improved text")

        # Test without explicit format marker
        response = "This is just plain improved text"
        result = self.parser.parse_improvement_response(response)
        self.assertEqual(result, "This is just plain improved text")

    def test_parse_improvement_response_invalid(self):
        """Test parsing an invalid improvement response."""
        result = self.parser.parse_improvement_response(None)
        self.assertIn("Failed to improve", result)

    def test_parse_reflection_response_dict(self):
        """Test parsing a reflection response dict."""
        response = {"reflection": "This is a reflection"}
        result = self.parser.parse_reflection_response(response)
        self.assertEqual(result, "This is a reflection")

    def test_parse_reflection_response_string(self):
        """Test parsing a reflection response string."""
        # Test with REFLECTION: format
        response = "REFLECTION: This is a reflection"
        result = self.parser.parse_reflection_response(response)
        self.assertEqual(result, "This is a reflection")

        # Test without explicit format marker
        response = "This is just a plain reflection"
        result = self.parser.parse_reflection_response(response)
        self.assertEqual(result, "This is just a plain reflection")

    def test_parse_reflection_response_invalid(self):
        """Test parsing an invalid reflection response."""
        result = self.parser.parse_reflection_response(None)
        self.assertIsNone(result)

    def test_parse_critique_string_special_cases(self):
        """Test parsing critique string with special cases."""
        # Test with invalid score value
        response = """SCORE: not_a_number
FEEDBACK: Good content"""
        result = self.parser._parse_critique_string(response)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["feedback"], "Good content")

        # Test with score out of range (should be clamped)
        response = """SCORE: 1.5
FEEDBACK: Good content"""
        result = self.parser._parse_critique_string(response)
        self.assertEqual(result["score"], 1.0)  # Clamped to 1.0

        response = """SCORE: -0.5
FEEDBACK: Bad content"""
        result = self.parser._parse_critique_string(response)
        self.assertEqual(result["score"], 0.0)  # Clamped to 0.0