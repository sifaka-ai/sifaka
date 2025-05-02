"""
Response parser for critics.

This module provides the ResponseParser class which is responsible for
parsing responses from language models.
"""

from typing import Any, Dict, List, Optional, Union

from ...utils.logging import get_logger

logger = get_logger(__name__)


class ResponseParser:
    """
    Parses responses from language models.
    
    This class is responsible for parsing responses from language models
    into structured data.
    """
    
    def parse_validation_response(self, response: Union[str, Dict[str, Any]]) -> bool:
        """
        Parse a validation response.
        
        Args:
            response: The response from the language model
            
        Returns:
            True if the text is valid, False otherwise
        """
        if isinstance(response, dict) and "valid" in response:
            return bool(response["valid"])
        elif isinstance(response, str):
            # Try to parse the response
            if "VALID: true" in response.lower():
                return True
            elif "VALID: false" in response.lower():
                return False
        return False
        
    def parse_critique_response(self, response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse a critique response.
        
        Args:
            response: The response from the language model
            
        Returns:
            A dictionary containing the critique details
        """
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
            
    def parse_improvement_response(self, response: Union[str, Dict[str, Any]]) -> str:
        """
        Parse an improvement response.
        
        Args:
            response: The response from the language model
            
        Returns:
            The improved text
        """
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
            
    def parse_reflection_response(self, response: Union[str, Dict[str, Any]]) -> Optional[str]:
        """
        Parse a reflection response.
        
        Args:
            response: The response from the language model
            
        Returns:
            The reflection, or None if parsing failed
        """
        if isinstance(response, dict) and "reflection" in response:
            return response["reflection"]
        elif isinstance(response, str):
            if "REFLECTION:" in response:
                parts = response.split("REFLECTION:")
                if len(parts) > 1:
                    return parts[1].strip()
            return response.strip()
        return None
        
    def _parse_critique_string(self, response: str) -> Dict[str, Any]:
        """
        Parse a critique response string.
        
        Args:
            response: The response string from the language model
            
        Returns:
            A dictionary containing the critique details
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
