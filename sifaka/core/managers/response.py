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
is_valid = parser.parse_validation_response(validation_response) if parser else ""
print(f"Is valid: {is_valid}")

# Parse a critique response
critique_response = (
    "SCORE: 0.8
"
    "FEEDBACK: Good text quality
"
    "ISSUES:
"
    "- Could use more detail
"
    "- Some grammar issues
"
    "SUGGESTIONS:
"
    "- Add specific examples
"
    "- Fix grammar errors"
)
critique = parser.parse_critique_response(critique_response) if parser else ""
print(f"Score: {critique['score']}")
print(f"Feedback: {critique['feedback']}")

# Parse an improvement response
improvement_response = "IMPROVED_TEXT: This is the improved version."
improved_text = parser.parse_improvement_response(improvement_response) if parser else ""
print(f"Improved text: {improved_text}")

# Parse a reflection response
reflection_response = "REFLECTION: Learned to focus on clarity."
reflection = parser.parse_reflection_response(reflection_response) if parser else ""
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
            is_valid = parser.parse_validation_response(validation_response) if parser else ""
            print(f"Is valid: {is_valid}")

            # Parse a critique response
            critique_response = (
                "SCORE: 0.8
    "
                "FEEDBACK: Good text quality
    "
                "ISSUES:
    "
                "- Could use more detail
    "
                "- Some grammar issues
    "
                "SUGGESTIONS:
    "
                "- Add specific examples
    "
                "- Fix grammar errors"
            )
            critique = parser.parse_critique_response(critique_response) if parser else ""
            print(f"Score: {critique['score']}")
            print(f"Feedback: {critique['feedback']}")
            ```
    """

    _state_manager = PrivateAttr(default_factory=StateManager)

    def __init__(self) -> None:
        """
        Initialize a ResponseParser instance.

        This method sets up the response parser with state management
        and initializes tracking metrics.
        """
        self._state_manager = StateManager()
        if self._state_manager:
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("component_type", "response_parser")
            self._state_manager.set_metadata("creation_time", time.time())
            self._state_manager.set_metadata("parse_count", 0)
            self._state_manager.set_metadata("error_count", 0)

    def parse_validation_response(self, response: Union[str, Dict[str, Any]]) -> Any:
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
            is_valid = parser.parse_validation_response(response) if parser else ""
            print(f"Is valid: {is_valid}")  # True

            # Parse a string response
            response = "VALID: true"
            is_valid = parser.parse_validation_response(response) if parser else ""
            print(f"Is valid: {is_valid}")  # True
            ```
        """
        if self._state_manager:
            parse_count = self._state_manager.get_metadata("parse_count", 0)
            self._state_manager.set_metadata("parse_count", parse_count + 1)
        try:
            if isinstance(response, dict) and "valid" in response:
                return bool(response["valid"])
            elif isinstance(response, str):
                response_lower = response.lower() if response else ""
                if "valid: true" in response_lower or "valid:true" in response_lower:
                    return True
                elif "valid: false" in response_lower or "valid:false" in response_lower:
                    return False
            return False
        except Exception as e:
            if self._state_manager:
                error_count = self._state_manager.get_metadata("error_count", 0)
                self._state_manager.set_metadata("error_count", error_count + 1)
            if logger:
                logger.error(f"Error parsing validation response: {e}")
            return False

    def parse_critique_response(self, response: Union[str, Dict[str, Any]]) -> Any:
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
                    critique = parser.parse_critique_response(response) if parser else ""
                    print(f"Score: {critique['score']}")  # 0.8

                    # Parse a string response
                    response = (
                        "SCORE: 0.8
        "
                        "FEEDBACK: Good text quality
        "
                        "ISSUES:
        "
                        "- Could use more detail
        "
                        "SUGGESTIONS:
        "
                        "- Add specific examples"
                    )
                    critique = parser.parse_critique_response(response) if parser else ""
                    print(f"Score: {critique['score']}")  # 0.8
                    ```
        """
        if self._state_manager:
            parse_count = self._state_manager.get_metadata("parse_count", 0)
            self._state_manager.set_metadata("parse_count", parse_count + 1)
        try:
            if isinstance(response, dict):
                return {
                    "score": float(response.get("score", 0.0)) if response else 0.0,
                    "feedback": str(response.get("feedback", "")) if response else "",
                    "issues": list(response.get("issues", [])) if response else [],
                    "suggestions": list(response.get("suggestions", [])) if response else [],
                }
            elif isinstance(response, str):
                return (
                    self._parse_critique_string(response)
                    if self
                    else {"score": 0.0, "feedback": "", "issues": [], "suggestions": []}
                )
            else:
                return {
                    "score": 0.0,
                    "feedback": "Failed to critique text: Invalid response format",
                    "issues": ["Invalid response format"],
                    "suggestions": ["Try again with clearer text"],
                }
        except Exception as e:
            if self._state_manager:
                error_count = self._state_manager.get_metadata("error_count", 0)
                self._state_manager.set_metadata("error_count", error_count + 1)
            if logger:
                logger.error(f"Error parsing critique response: {e}")
            return {
                "score": 0.0,
                "feedback": f"Error parsing critique: {str(e)}",
                "issues": ["Error parsing response"],
                "suggestions": ["Try again with clearer text"],
            }

    def parse_improvement_response(self, response: Union[str, Dict[str, Any]]) -> Any:
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
            improved = parser.parse_improvement_response(response) if parser else ""
            print(improved)  # "This is the improved version."

            # Parse a string response
            response = "IMPROVED_TEXT: This is the improved version."
            improved = parser.parse_improvement_response(response) if parser else ""
            print(improved)  # "This is the improved version."
            ```
        """
        if self._state_manager:
            parse_count = self._state_manager.get_metadata("parse_count", 0)
            self._state_manager.set_metadata("parse_count", parse_count + 1)
        try:
            if isinstance(response, dict) and "improved_text" in response:
                return response["improved_text"]
            elif isinstance(response, str):
                if "IMPROVED_TEXT:" in response:
                    parts = response.split("IMPROVED_TEXT:") if response else []
                    if len(parts) > 1:
                        return parts[1].strip()
                return response.strip() if response else ""
            else:
                return "Failed to improve text: Invalid response format"
        except Exception as e:
            if self._state_manager:
                error_count = self._state_manager.get_metadata("error_count", 0)
                self._state_manager.set_metadata("error_count", error_count + 1)
            if logger:
                logger.error(f"Error parsing improvement response: {e}")
            return f"Failed to improve text: {str(e)}"

    def parse_reflection_response(self, response: Union[str, Dict[str, Any]]) -> Any:
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
            reflection = parser.parse_reflection_response(response) if parser else ""
            print(reflection)  # "Learned to focus on clarity."

            # Parse a string response
            response = "REFLECTION: Learned to focus on clarity."
            reflection = parser.parse_reflection_response(response) if parser else ""
            print(reflection)  # "Learned to focus on clarity."
            ```
        """
        if self._state_manager:
            parse_count = self._state_manager.get_metadata("parse_count", 0)
            self._state_manager.set_metadata("parse_count", parse_count + 1)
        try:
            if isinstance(response, dict) and "reflection" in response:
                return response["reflection"]
            elif isinstance(response, str):
                if "REFLECTION:" in response:
                    parts = response.split("REFLECTION:") if response else []
                    if len(parts) > 1:
                        return parts[1].strip()
                return response.strip() if response else ""
            return None
        except Exception as e:
            if self._state_manager:
                error_count = self._state_manager.get_metadata("error_count", 0)
                self._state_manager.set_metadata("error_count", error_count + 1)
            if logger:
                logger.error(f"Error parsing reflection response: {e}")
            return None

    def _parse_critique_string(self, response: str) -> Any:
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
                        "SCORE: 0.8
        "
                        "FEEDBACK: Good text quality
        "
                        "ISSUES:
        "
                        "- Could use more detail
        "
                        "SUGGESTIONS:
        "
                        "- Add specific examples"
                    )
                    critique = parser._parse_critique_string(response) if parser else ""
                    print(f"Score: {critique['score']}")  # 0.8
                    print(f"Feedback: {critique['feedback']}")  # "Good text quality"
                    ```
        """
        result = {"score": 0.0, "feedback": "", "issues": [], "suggestions": []}
        if "SCORE:" in response:
            try:
                score_line = response.split("SCORE:")[1].split("\n")[0].strip() if response else ""
                score = float(score_line)
                result["score"] = max(0.0, min(1.0, score))
            except (ValueError, TypeError, IndexError):
                pass
        if "FEEDBACK:" in response:
            try:
                feedback_parts = response.split("FEEDBACK:")
                if len(feedback_parts) > 1:
                    feedback_text = feedback_parts[1]
                    if "ISSUES:" in feedback_text:
                        feedback_text = feedback_text.split("ISSUES:")[0]
                    result["feedback"] = feedback_text.strip()
            except (IndexError, AttributeError):
                pass
        if "ISSUES:" in response:
            try:
                issues_parts = response.split("ISSUES:")
                if len(issues_parts) > 1:
                    issues_text = issues_parts[1]
                    if "SUGGESTIONS:" in issues_text:
                        issues_text = issues_text.split("SUGGESTIONS:")[0]

                    issues = []
                    for line in issues_text.strip().split("\n"):
                        line = line.strip() if line else ""
                        if line and line.startswith("-"):
                            issues.append(line[1:].strip())
                    result["issues"] = issues
            except (IndexError, AttributeError):
                pass
        if "SUGGESTIONS:" in response:
            try:
                suggestions_parts = response.split("SUGGESTIONS:")
                if len(suggestions_parts) > 1:
                    suggestions_text = suggestions_parts[1]

                    suggestions = []
                    for line in suggestions_text.strip().split("\n"):
                        line = line.strip() if line else ""
                        if line and line.startswith("-"):
                            suggestions.append(line[1:].strip())
                    result["suggestions"] = suggestions
            except (IndexError, AttributeError):
                pass
        return result

    def get_statistics(self) -> Any:
        """
        Get statistics about response parsing.

        Returns:
            Dictionary with usage statistics
        """
        parse_count = (
            self._state_manager.get_metadata("parse_count", 0) if self._state_manager else 0
        )
        error_count = (
            self._state_manager.get_metadata("error_count", 0) if self._state_manager else 0
        )
        creation_time = (
            self._state_manager.get_metadata("creation_time", time.time())
            if self._state_manager
            else 0
        )
        current_time = time.time()

        return {
            "parse_count": parse_count,
            "error_count": error_count,
            "uptime": current_time - creation_time,
        }


def create_response_parser() -> Any:
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
        is_valid = parser.parse_validation_response(validation_response) if parser else False
        print(f"Is valid: {is_valid}")
        ```
    """
    return ResponseParser()
