"""
Response parsing for critics.

This module provides utilities for parsing responses from language models
in the context of critic operations.
"""

import re
import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from sifaka.utils.logging import get_logger
from sifaka.utils.errors import CriticError
from sifaka.utils.patterns import compile_pattern, match_pattern, find_patterns

# Configure logger
logger = get_logger(__name__)


class ResponseParser(BaseModel):
    """
    Parser for critic responses.

    This class is responsible for parsing responses from language models
    in the context of critic operations.
    """

    def parse_validation_response(self, response: str) -> bool:
        """
        Parse a validation response.

        Args:
            response: The response from the language model

        Returns:
            True if the text is valid, False otherwise
        """
        # Look for validation patterns
        valid_patterns = [
            r"(?i)valid",
            r"(?i)passes",
            r"(?i)acceptable",
            r"(?i)meets.*criteria",
            r"(?i)meets.*standards",
        ]
        invalid_patterns = [
            r"(?i)invalid",
            r"(?i)fails",
            r"(?i)unacceptable",
            r"(?i)does not meet.*criteria",
            r"(?i)does not meet.*standards",
        ]

        # Check for valid patterns
        for pattern in valid_patterns:
            if re.search(pattern, response):
                return True

        # Check for invalid patterns
        for pattern in invalid_patterns:
            if re.search(pattern, response):
                return False

        # Default to valid if no patterns match
        return True

    def parse_critique_response(self, response: str) -> Dict[str, Any]:
        """
        Parse a critique response.

        Args:
            response: The response from the language model

        Returns:
            A dictionary containing the critique details
        """
        # Try to parse as JSON
        try:
            # Look for JSON block
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result
        except Exception:
            pass

        # Extract score
        score_match = re.search(r"(?i)score:?\s*(\d+(\.\d+)?)", response)
        score = float(score_match.group(1)) if score_match else 0.5

        # Extract feedback
        feedback = response
        if "feedback:" in response.lower():
            feedback_match = re.search(r"(?i)feedback:?\s*(.*?)(\n\n|$)", response, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()

        # Extract issues
        issues = []
        issues_section = re.search(r"(?i)issues:?\s*(.*?)(\n\n|$)", response, re.DOTALL)
        if issues_section:
            issues_text = issues_section.group(1)
            issues = [issue.strip() for issue in re.findall(r"[-*]\s*(.*?)(?=\n[-*]|\n\n|$)", issues_text, re.DOTALL)]

        # Extract suggestions
        suggestions = []
        suggestions_section = re.search(r"(?i)suggestions:?\s*(.*?)(\n\n|$)", response, re.DOTALL)
        if suggestions_section:
            suggestions_text = suggestions_section.group(1)
            suggestions = [
                suggestion.strip()
                for suggestion in re.findall(r"[-*]\s*(.*?)(?=\n[-*]|\n\n|$)", suggestions_text, re.DOTALL)
            ]

        return {
            "score": score,
            "feedback": feedback,
            "issues": issues,
            "suggestions": suggestions,
        }

    def parse_improvement_response(self, response: str) -> str:
        """
        Parse an improvement response.

        Args:
            response: The response from the language model

        Returns:
            The improved text
        """
        # Look for improved text section
        improved_section = re.search(r"(?i)improved text:?\s*(.*?)(\n\n|$)", response, re.DOTALL)
        if improved_section:
            return improved_section.group(1).strip()

        # Look for code block
        code_block = re.search(r"```(?:.*?)\n(.*?)```", response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Return the entire response if no patterns match
        return response.strip()

    def parse_reflection_response(self, response: str) -> str:
        """
        Parse a reflection response.

        Args:
            response: The response from the language model

        Returns:
            The reflection text
        """
        # Look for reflection section
        reflection_section = re.search(r"(?i)reflection:?\s*(.*?)(\n\n|$)", response, re.DOTALL)
        if reflection_section:
            return reflection_section.group(1).strip()

        # Return the entire response if no patterns match
        return response.strip()
