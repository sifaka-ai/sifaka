"""
Tests for the critics ResponseParser class.

This module contains comprehensive tests for the ResponseParser class that
is used to parse responses from language models.

Tests cover all public methods including parse_validation_response, parse_critique_response,
parse_improvement_response, and parse_reflection_response, as well as the _parse_critique_string
private method. Both normal and edge cases are tested.
"""

import unittest
from unittest.mock import patch
import pytest

from sifaka.critics.managers.response import ResponseParser


class TestResponseParser(unittest.TestCase):
    """Tests for the ResponseParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = ResponseParser()

    def test_parse_validation_response_with_dict(self):
        """Test parsing validation response in dictionary format."""
        # Valid response
        self.assertTrue(self.parser.parse_validation_response({"valid": True}))
        self.assertTrue(self.parser.parse_validation_response({"valid": 1}))

        # Invalid response
        self.assertFalse(self.parser.parse_validation_response({"valid": False}))
        self.assertFalse(self.parser.parse_validation_response({"valid": 0}))
        self.assertFalse(self.parser.parse_validation_response({"valid": None}))

        # Missing "valid" key
        self.assertFalse(self.parser.parse_validation_response({"something_else": True}))

    def test_parse_validation_response_with_string(self):
        """Test parsing validation response in string format."""
        # Valid response - different formats
        self.assertTrue(self.parser.parse_validation_response("VALID: true"))
        self.assertTrue(self.parser.parse_validation_response("Valid: True"))
        self.assertTrue(self.parser.parse_validation_response("valid: true"))
        self.assertTrue(self.parser.parse_validation_response("VALID:true"))
        self.assertTrue(self.parser.parse_validation_response("Some other text\nVALID: true\nMore text"))

        # Invalid response - different formats
        self.assertFalse(self.parser.parse_validation_response("VALID: false"))
        self.assertFalse(self.parser.parse_validation_response("Valid: False"))
        self.assertFalse(self.parser.parse_validation_response("valid: false"))
        self.assertFalse(self.parser.parse_validation_response("VALID:false"))
        self.assertFalse(self.parser.parse_validation_response("Some other text\nVALID: false\nMore text"))

        # Malformed response
        self.assertFalse(self.parser.parse_validation_response("This is not a valid response"))
        self.assertFalse(self.parser.parse_validation_response(""))
        self.assertFalse(self.parser.parse_validation_response("VALID: maybe"))

    def test_parse_critique_response_with_dict(self):
        """Test parsing critique response in dictionary format."""
        # Complete valid response
        response_dict = {
            "score": 0.8,
            "feedback": "Good quality content",
            "issues": ["Minor grammar issue", "Could use more examples"],
            "suggestions": ["Fix grammar", "Add examples"]
        }
        result = self.parser.parse_critique_response(response_dict)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good quality content")
        self.assertEqual(result["issues"], ["Minor grammar issue", "Could use more examples"])
        self.assertEqual(result["suggestions"], ["Fix grammar", "Add examples"])

        # Missing keys (should provide defaults)
        result = self.parser.parse_critique_response({"score": 0.9})
        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

        # Type conversion
        result = self.parser.parse_critique_response({
            "score": "0.7",  # string score
            "feedback": 123,  # int feedback
            "issues": {"issue1": "desc"},  # dict issues
            "suggestions": "Fix it"  # string suggestions
        })
        self.assertEqual(result["score"], 0.7)
        self.assertEqual(result["feedback"], "123")
        # The actual behavior converts dict to a list containing the keys, not the whole dict
        self.assertEqual(result["issues"], ["issue1"])
        # String is split into characters
        self.assertEqual(result["suggestions"], list("Fix it"))

    def test_parse_critique_response_with_string(self):
        """Test parsing critique response in string format."""
        # Complete response
        complete_response = """
        SCORE: 0.85
        FEEDBACK: This is good quality content with some minor issues.
        ISSUES:
        - Grammar errors in second paragraph
        - Missing conclusion
        SUGGESTIONS:
        - Fix grammar issues
        - Add a strong conclusion
        """
        result = self.parser.parse_critique_response(complete_response)
        self.assertEqual(result["score"], 0.85)
        self.assertEqual(result["feedback"], "This is good quality content with some minor issues.")
        self.assertEqual(result["issues"], ["Grammar errors in second paragraph", "Missing conclusion"])
        self.assertEqual(result["suggestions"], ["Fix grammar issues", "Add a strong conclusion"])

        # Partial response
        partial_response = """
        SCORE: 0.6
        FEEDBACK: Needs improvement.
        """
        result = self.parser.parse_critique_response(partial_response)
        self.assertEqual(result["score"], 0.6)
        self.assertEqual(result["feedback"], "Needs improvement.")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

        # Invalid score value
        invalid_score = """
        SCORE: invalid
        FEEDBACK: Some feedback
        """
        result = self.parser.parse_critique_response(invalid_score)
        self.assertEqual(result["score"], 0.0)  # Default to 0.0 for invalid score
        self.assertEqual(result["feedback"], "Some feedback")

        # Malformed response
        malformed = "This is not a properly formatted critique response"
        result = self.parser.parse_critique_response(malformed)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

    def test_parse_critique_response_with_invalid_type(self):
        """Test parsing critique response with invalid type."""
        result = self.parser.parse_critique_response(123)  # Not a string or dict
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Failed to critique text", result["feedback"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(len(result["suggestions"]), 1)

    def test_parse_critique_string_with_score_clamping(self):
        """Test that _parse_critique_string clamps scores to [0, 1]."""
        # Score too high
        high_score = "SCORE: 1.5"
        result = self.parser._parse_critique_string(high_score)
        self.assertEqual(result["score"], 1.0)

        # Score too low
        low_score = "SCORE: -0.5"
        result = self.parser._parse_critique_string(low_score)
        self.assertEqual(result["score"], 0.0)

    def test_parse_improvement_response_with_dict(self):
        """Test parsing improvement response in dictionary format."""
        # Valid response
        self.assertEqual(
            self.parser.parse_improvement_response({"improved_text": "This is improved"}),
            "This is improved"
        )

        # Dictionary without improved_text key
        self.assertEqual(
            self.parser.parse_improvement_response({"some_key": "This is not improved"}),
            "Failed to improve text: Invalid response format"
        )

    def test_parse_improvement_response_with_string(self):
        """Test parsing improvement response in string format."""
        # String with IMPROVED_TEXT marker
        self.assertEqual(
            self.parser.parse_improvement_response("IMPROVED_TEXT: This is the improved version."),
            "This is the improved version."
        )

        # String with IMPROVED_TEXT marker and other content - it keeps the full content after the marker
        self.assertEqual(
            self.parser.parse_improvement_response("Some explanation\nIMPROVED_TEXT: Better version\nMore notes"),
            "Better version\nMore notes"
        )

        # Simple string without marker
        self.assertEqual(
            self.parser.parse_improvement_response("This is just a plain response"),
            "This is just a plain response"
        )

        # Empty string
        self.assertEqual(self.parser.parse_improvement_response(""), "")

        # String with whitespace
        self.assertEqual(self.parser.parse_improvement_response("  text with spaces  "), "text with spaces")

    def test_parse_improvement_response_with_invalid_type(self):
        """Test parsing improvement response with invalid type."""
        self.assertEqual(
            self.parser.parse_improvement_response(123),  # Not a string or dict
            "Failed to improve text: Invalid response format"
        )
        self.assertEqual(
            self.parser.parse_improvement_response(None),  # None
            "Failed to improve text: Invalid response format"
        )

    def test_parse_reflection_response_with_dict(self):
        """Test parsing reflection response in dictionary format."""
        # Valid response
        self.assertEqual(
            self.parser.parse_reflection_response({"reflection": "This is a reflection"}),
            "This is a reflection"
        )

        # Dictionary without reflection key
        self.assertIsNone(self.parser.parse_reflection_response({"some_key": "This is not a reflection"}))

    def test_parse_reflection_response_with_string(self):
        """Test parsing reflection response in string format."""
        # String with REFLECTION marker
        self.assertEqual(
            self.parser.parse_reflection_response("REFLECTION: I learned to improve clarity."),
            "I learned to improve clarity."
        )

        # String with REFLECTION marker and other content - it keeps the full content after the marker
        self.assertEqual(
            self.parser.parse_reflection_response("Some explanation\nREFLECTION: Be more concise\nMore notes"),
            "Be more concise\nMore notes"
        )

        # Simple string without marker
        self.assertEqual(
            self.parser.parse_reflection_response("This is just a plain reflection"),
            "This is just a plain reflection"
        )

        # Empty string
        self.assertEqual(self.parser.parse_reflection_response(""), "")

        # String with whitespace
        self.assertEqual(
            self.parser.parse_reflection_response("  reflection with spaces  "),
            "reflection with spaces"
        )

    def test_parse_reflection_response_with_invalid_type(self):
        """Test parsing reflection response with invalid type."""
        self.assertIsNone(self.parser.parse_reflection_response(123))  # Not a string or dict
        self.assertIsNone(self.parser.parse_reflection_response(None))  # None


if __name__ == "__main__":
    unittest.main()