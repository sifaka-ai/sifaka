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
from sifaka.utils.errors.component import CriticError
from sifaka.utils.patterns import compile_pattern, match_pattern, find_patterns

logger = get_logger(__name__)


class ResponseParser(BaseModel):
    """
    Parser for critic responses.

    This class is responsible for parsing responses from language models
    in the context of critic operations.
    """

    def parse_validation_response(self, response: str) -> Any:
        """
        Parse a validation response.

        Args:
            response: The response from the language model

        Returns:
            True if the text is valid, False otherwise
        """
        valid_patterns = [
            "(?i)valid",
            "(?i)passes",
            "(?i)acceptable",
            "(?i)meets.*criteria",
            "(?i)meets.*standards",
        ]
        invalid_patterns = [
            "(?i)invalid",
            "(?i)fails",
            "(?i)unacceptable",
            "(?i)does not meet.*criteria",
            "(?i)does not meet.*standards",
        ]
        for pattern in valid_patterns:
            if re.search(pattern, response):
                return True
        for pattern in invalid_patterns:
            if re.search(pattern, response):
                return False
        return True

    def parse_critique_response(self, response: str) -> Any:
        """
        Parse a critique response.

        Args:
            response: The response from the language model

        Returns:
            A dictionary containing the critique details
        """
        try:
            json_match = re.search("```json\\s*(.*?)\\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result
        except Exception:
            pass
        score_match = re.search("(?i)score:?\\s*(\\d+(\\.\\d+)?)", response)
        score = float(score_match.group(1) if score_match else 0.5)
        feedback = response
        if "feedback:" in response.lower():
            feedback_match = re.search("(?i)feedback:?\\s*(.*?)(\\n\\n|$)", response, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
        issues = []
        issues_section = re.search("(?i)issues:?\\s*(.*?)(\\n\\n|$)", response, re.DOTALL)
        if issues_section:
            issues_text = issues_section.group(1)
            issues = [
                issue.strip()
                for issue in re.findall("[-*]\\s*(.*?)(?=\\n[-*]|\\n\\n|$)", issues_text, re.DOTALL)
            ]
        suggestions = []
        suggestions_section = re.search("(?i)suggestions:?\\s*(.*?)(\\n\\n|$)", response, re.DOTALL)
        if suggestions_section:
            suggestions_text = suggestions_section.group(1)
            suggestions = [
                suggestion.strip()
                for suggestion in re.findall(
                    "[-*]\\s*(.*?)(?=\\n[-*]|\\n\\n|$)", suggestions_text, re.DOTALL
                )
            ]
        return {"score": score, "feedback": feedback, "issues": issues, "suggestions": suggestions}

    def parse_improvement_response(self, response: str) -> Any:
        """
        Parse an improvement response.

        Args:
            response: The response from the language model

        Returns:
            The improved text
        """
        improved_section = re.search("(?i)improved text:?\\s*(.*?)(\\n\\n|$)", response, re.DOTALL)
        if improved_section:
            return improved_section.group(1).strip()
        code_block = re.search("```(?:.*?)\\n(.*?)```", response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        return response.strip()

    def parse_reflection_response(self, response: str) -> Any:
        """
        Parse a reflection response.

        Args:
            response: The response from the language model

        Returns:
            The reflection text
        """
        reflection_section = re.search("(?i)reflection:?\\s*(.*?)(\\n\\n|$)", response, re.DOTALL)
        if reflection_section:
            return reflection_section.group(1).strip()
        return response.strip()
