"""
Response parser for Sifaka components.

This module provides the ResponseParser class which is responsible for
parsing and validating responses from language models. It handles different
types of responses including validation, critique, improvement, and reflection.

## Component Overview

1. **Core Components**
   - `ResponseParser`: Main class for parsing model responses
   - Response format validation
   - Structured data extraction
   - Error handling

2. **Response Types**
   - Validation responses (boolean)
   - Critique responses (structured feedback)
   - Improvement responses (text)
   - Reflection responses (text)

## Component Lifecycle

1. **Initialization**
   - Setup logging
   - Configure error handling
   - Initialize state

2. **Operation**
   - Parse different response types
   - Validate response formats
   - Extract structured data
   - Handle errors

3. **Cleanup**
   - Reset state
   - Log final status

## Error Handling

1. **Format Errors**
   - Invalid response structure
   - Missing required fields
   - Type mismatches
   - Recovery: Return default values

2. **Parsing Errors**
   - Invalid string formats
   - Missing markers
   - Recovery: Use fallback parsing

3. **Validation Errors**
   - Invalid score ranges
   - Empty feedback
   - Recovery: Return safe defaults

## Examples

```python
from sifaka.core.managers.response import ResponseParser

# Create a response parser
parser = ResponseParser()

# Parse a validation response
validation_response = "VALID: true"
is_valid = parser.parse_validation_response(validation_response)
print(f"Is valid: {is_valid}")

# Parse a critique response
critique_response = (
    "SCORE: 0.8\n"
    "FEEDBACK: Good text quality\n"
    "ISSUES:\n"
    "- Could use more detail\n"
    "- Some grammar issues\n"
    "SUGGESTIONS:\n"
    "- Add specific examples\n"
    "- Fix grammar errors"
)
critique = parser.parse_critique_response(critique_response)
print(f"Score: {critique['score']}")
print(f"Feedback: {critique['feedback']}")

# Parse an improvement response
improvement_response = "IMPROVED_TEXT: This is the improved version."
improved_text = parser.parse_improvement_response(improvement_response)
print(f"Improved text: {improved_text}")

# Parse a reflection response
reflection_response = "REFLECTION: Learned to focus on clarity."
reflection = parser.parse_reflection_response(reflection_response)
print(f"Reflection: {reflection}")
```
"""

from typing import Any, Dict, Optional, Union
import time

from pydantic import PrivateAttr

from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ResponseParser:
    """
    Parses responses from language models.

    This class is responsible for parsing and validating responses from
    language models, converting them into structured data that can be
    used by various components in the Sifaka framework.

    ## Lifecycle Management

    1. **Initialization**
       - Setup logging
       - Configure error handling
       - Initialize state

    2. **Operation**
       - Parse different response types
       - Validate response formats
       - Extract structured data
       - Handle errors

    3. **Cleanup**
       - Reset state
       - Log final status

    ## Error Handling

    1. **Format Errors**
       - Invalid response structure
       - Missing required fields
       - Type mismatches
       - Recovery: Return default values

    2. **Parsing Errors**
       - Invalid string formats
       - Missing markers
       - Recovery: Use fallback parsing

    3. **Validation Errors**
       - Invalid score ranges
       - Empty feedback
       - Recovery: Return safe defaults

    Examples:
        ```python
        from sifaka.core.managers.response import ResponseParser

        # Create a response parser
        parser = ResponseParser()

        # Parse a validation response
        validation_response = "VALID: true"
        is_valid = parser.parse_validation_response(validation_response)
        print(f"Is valid: {is_valid}")

        # Parse a critique response
        critique_response = (
            "SCORE: 0.8\n"
            "FEEDBACK: Good text quality\n"
            "ISSUES:\n"
            "- Could use more detail\n"
            "- Some grammar issues\n"
            "SUGGESTIONS:\n"
            "- Add specific examples\n"
            "- Fix grammar errors"
        )
        critique = parser.parse_critique_response(critique_response)
        print(f"Score: {critique['score']}")
        print(f"Feedback: {critique['feedback']}")
        ```
    """

    # State management
    _state_manager = PrivateAttr(default_factory=StateManager)

    def __init__(self):
        """
        Initialize a ResponseParser instance.

        This method sets up the response parser with state management
        and initializes tracking metrics.
        """
        # Initialize state
        self._state_manager.update("initialized", True)

        # Initialize metadata
        self._state_manager.set_metadata("component_type", "response_parser")
        self._state_manager.set_metadata("creation_time", time.time())
        self._state_manager.set_metadata("parse_count", 0)
        self._state_manager.set_metadata("error_count", 0)

    def parse_validation_response(self, response: Union[str, Dict[str, Any]]) -> bool:
        """
        Parse a validation response.

        This method parses a response from a language model that indicates
        whether a text is valid or not. It handles both string and dictionary
        formats.

        Lifecycle:
        1. Input validation
        2. Format detection
        3. Value extraction
        4. Error handling

        Args:
            response: The response to parse, either a string or dictionary

        Returns:
            True if the text is valid, False otherwise

        Examples:
            ```python
            # Parse a dictionary response
            response = {"valid": True}
            is_valid = parser.parse_validation_response(response)
            print(f"Is valid: {is_valid}")  # True

            # Parse a string response
            response = "VALID: true"
            is_valid = parser.parse_validation_response(response)
            print(f"Is valid: {is_valid}")  # True
            ```
        """
        # Track parse count
        parse_count = self._state_manager.get_metadata("parse_count", 0)
        self._state_manager.set_metadata("parse_count", parse_count + 1)

        try:
            if isinstance(response, dict) and "valid" in response:
                return bool(response["valid"])
            elif isinstance(response, str):
                # Try to parse the response - using proper case insensitive comparison
                response_lower = response.lower()
                if "valid: true" in response_lower or "valid:true" in response_lower:
                    return True
                elif "valid: false" in response_lower or "valid:false" in response_lower:
                    return False
            return False
        except Exception as e:
            # Track error
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            logger.error(f"Error parsing validation response: {e}")
            return False

    def parse_critique_response(self, response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse a critique response.

        This method parses a response from a language model that provides
        feedback on a text. It handles both string and dictionary formats,
        extracting score, feedback, issues, and suggestions.

        Lifecycle:
        1. Input validation
        2. Format detection
        3. Value extraction
        4. Error handling

        Args:
            response: The response to parse, either a string or dictionary

        Returns:
            A dictionary containing:
            - score: float between 0 and 1
            - feedback: string with general feedback
            - issues: list of identified issues
            - suggestions: list of improvement suggestions

        Examples:
            ```python
            # Parse a dictionary response
            response = {
                "score": 0.8,
                "feedback": "Good text quality",
                "issues": ["Could use more detail"],
                "suggestions": ["Add specific examples"]
            }
            critique = parser.parse_critique_response(response)
            print(f"Score: {critique['score']}")  # 0.8

            # Parse a string response
            response = (
                "SCORE: 0.8\n"
                "FEEDBACK: Good text quality\n"
                "ISSUES:\n"
                "- Could use more detail\n"
                "SUGGESTIONS:\n"
                "- Add specific examples"
            )
            critique = parser.parse_critique_response(response)
            print(f"Score: {critique['score']}")  # 0.8
            ```
        """
        # Track parse count
        parse_count = self._state_manager.get_metadata("parse_count", 0)
        self._state_manager.set_metadata("parse_count", parse_count + 1)

        try:
            if isinstance(response, dict):
                # If response is already a dict, use it directly
                return {
                    "score": float(response.get("score", 0.0)),
                    "feedback": str(response.get("feedback", "")),
                    "issues": list(response.get("issues", [])),
                    "suggestions": list(response.get("suggestions", [])),
                }
            elif isinstance(response, str):
                # Parse string response
                return self._parse_critique_string(response)
            else:
                return {
                    "score": 0.0,
                    "feedback": "Failed to critique text: Invalid response format",
                    "issues": ["Invalid response format"],
                    "suggestions": ["Try again with clearer text"],
                }
        except Exception as e:
            # Track error
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            logger.error(f"Error parsing critique response: {e}")
            return {
                "score": 0.0,
                "feedback": f"Error parsing critique: {str(e)}",
                "issues": ["Error parsing response"],
                "suggestions": ["Try again with clearer text"],
            }

    def parse_improvement_response(self, response: Union[str, Dict[str, Any]]) -> str:
        """
        Parse an improvement response.

        This method parses a response from a language model that provides
        an improved version of a text. It handles both string and dictionary
        formats.

        Lifecycle:
        1. Input validation
        2. Format detection
        3. Value extraction
        4. Error handling

        Args:
            response: The response to parse, either a string or dictionary

        Returns:
            The improved text as a string

        Examples:
            ```python
            # Parse a dictionary response
            response = {"improved_text": "This is the improved version."}
            improved = parser.parse_improvement_response(response)
            print(improved)  # "This is the improved version."

            # Parse a string response
            response = "IMPROVED_TEXT: This is the improved version."
            improved = parser.parse_improvement_response(response)
            print(improved)  # "This is the improved version."
            ```
        """
        # Track parse count
        parse_count = self._state_manager.get_metadata("parse_count", 0)
        self._state_manager.set_metadata("parse_count", parse_count + 1)

        try:
            if isinstance(response, dict) and "improved_text" in response:
                return response["improved_text"]
            elif isinstance(response, str):
                # Try to extract improved text from string response
                if "IMPROVED_TEXT:" in response:
                    parts = response.split("IMPROVED_TEXT:")
                    if len(parts) > 1:
                        return parts[1].strip()
                return response.strip()
            else:
                return "Failed to improve text: Invalid response format"
        except Exception as e:
            # Track error
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            logger.error(f"Error parsing improvement response: {e}")
            return f"Failed to improve text: {str(e)}"

    def parse_reflection_response(self, response: Union[str, Dict[str, Any]]) -> Optional[str]:
        """
        Parse a reflection response.

        This method parses a response from a language model that provides
        a reflection on the improvement process. It handles both string
        and dictionary formats.

        Lifecycle:
        1. Input validation
        2. Format detection
        3. Value extraction
        4. Error handling

        Args:
            response: The response to parse, either a string or dictionary

        Returns:
            The reflection text as a string, or None if parsing fails

        Examples:
            ```python
            # Parse a dictionary response
            response = {"reflection": "Learned to focus on clarity."}
            reflection = parser.parse_reflection_response(response)
            print(reflection)  # "Learned to focus on clarity."

            # Parse a string response
            response = "REFLECTION: Learned to focus on clarity."
            reflection = parser.parse_reflection_response(response)
            print(reflection)  # "Learned to focus on clarity."
            ```
        """
        # Track parse count
        parse_count = self._state_manager.get_metadata("parse_count", 0)
        self._state_manager.set_metadata("parse_count", parse_count + 1)

        try:
            if isinstance(response, dict) and "reflection" in response:
                return response["reflection"]
            elif isinstance(response, str):
                if "REFLECTION:" in response:
                    parts = response.split("REFLECTION:")
                    if len(parts) > 1:
                        return parts[1].strip()
                return response.strip()
            return None
        except Exception as e:
            # Track error
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            logger.error(f"Error parsing reflection response: {e}")
            return None

    def _parse_critique_string(self, response: str) -> Dict[str, Any]:
        """
        Parse a critique response string.

        This internal method parses a string response containing critique
        information, extracting score, feedback, issues, and suggestions.

        Lifecycle:
        1. Input validation
        2. Marker detection
        3. Value extraction
        4. Error handling

        Args:
            response: The string response to parse

        Returns:
            A dictionary containing:
            - score: float between 0 and 1
            - feedback: string with general feedback
            - issues: list of identified issues
            - suggestions: list of improvement suggestions

        Examples:
            ```python
            response = (
                "SCORE: 0.8\n"
                "FEEDBACK: Good text quality\n"
                "ISSUES:\n"
                "- Could use more detail\n"
                "SUGGESTIONS:\n"
                "- Add specific examples"
            )
            critique = parser._parse_critique_string(response)
            print(f"Score: {critique['score']}")  # 0.8
            print(f"Feedback: {critique['feedback']}")  # "Good text quality"
            ```
        """
        result = {
            "score": 0.0,
            "feedback": "",
            "issues": [],
            "suggestions": [],
        }

        # Extract score
        if "SCORE:" in response:
            score_line = response.split("SCORE:")[1].split("\n")[0].strip()
            try:
                score = float(score_line)
                result["score"] = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except (ValueError, TypeError):
                pass

        # Extract feedback
        if "FEEDBACK:" in response:
            feedback_parts = response.split("FEEDBACK:")[1].split("ISSUES:")[0].strip()
            result["feedback"] = feedback_parts

        # Extract issues
        if "ISSUES:" in response:
            issues_part = response.split("ISSUES:")[1]
            if "SUGGESTIONS:" in issues_part:
                issues_part = issues_part.split("SUGGESTIONS:")[0]

            issues = []
            for line in issues_part.strip().split("\n"):
                line = line.strip()
                if line.startswith("-"):
                    issues.append(line[1:].strip())
            result["issues"] = issues

        # Extract suggestions
        if "SUGGESTIONS:" in response:
            suggestions_part = response.split("SUGGESTIONS:")[1].strip()
            suggestions = []
            for line in suggestions_part.split("\n"):
                line = line.strip()
                if line.startswith("-"):
                    suggestions.append(line[1:].strip())
            result["suggestions"] = suggestions

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about response parsing.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "parse_count": self._state_manager.get_metadata("parse_count", 0),
            "error_count": self._state_manager.get_metadata("error_count", 0),
            "uptime": time.time() - self._state_manager.get_metadata("creation_time", time.time()),
        }


def create_response_parser() -> ResponseParser:
    """
    Create a response parser.

    This factory function creates a ResponseParser instance with default configuration.

    Returns:
        Configured ResponseParser instance

    Examples:
        ```python
        from sifaka.core.managers.response import create_response_parser

        # Create a response parser
        parser = create_response_parser()

        # Parse a validation response
        validation_response = "VALID: true"
        is_valid = parser.parse_validation_response(validation_response)
        print(f"Is valid: {is_valid}")
        ```
    """
    return ResponseParser()
