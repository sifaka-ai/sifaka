"""
Concrete implementation of critics.

This module provides concrete implementations of the BaseCritic abstract class,
including the standard Critic class for string-based text processing.

## Overview
The module provides the Critic class, which is a complete implementation of the
critic interface for processing string-based text, with built-in validation,
improvement, and critique functionality.

## Components
1. **Critic**: Concrete implementation of BaseCritic for string-based text processing

## Usage Examples
```python
from sifaka.critics.base.implementation import Critic
from sifaka.utils.config.critics import CriticConfig

# Create a critic
critic = Critic(
    name="text_critic",
    description="Analyzes and improves text quality"
)

# Process text
text = "This is a test text."
result = critic.process(text) if critic else ""

# Check results
print(f"Score: {result.score:.2f}")
print(f"Feedback: {result.message}")
print(f"Issues: {result.issues}")
print(f"Suggestions: {result.suggestions}")
```

## Error Handling
The class implements comprehensive error handling for:
1. Input Validation
   - Empty text checks
   - Type validation
   - Format verification
   - Content validation

2. Processing Errors
   - Validation failures
   - Improvement errors
   - Critique failures
   - Resource errors

3. Recovery Strategies
   - Default values
   - Fallback methods
   - State preservation
   - Error logging
"""

from typing import Any, Dict, List, Optional

from sifaka.core.base import BaseResult
from sifaka.critics.base.abstract import BaseCritic


class Critic(BaseCritic[str]):
    """
    Concrete implementation of BaseCritic for string-based text processing.

    This class provides a complete implementation of the critic interface
    for processing string-based text, with built-in validation, improvement,
    and critique functionality.

    ## Overview
    The class provides:
    - String-based text validation
    - Text improvement with feedback
    - Text critique with detailed analysis
    - Error handling and recovery
    - Configuration management

    ## Usage Examples
    ```python
    from sifaka.critics.base.implementation import Critic
    from sifaka.utils.config.critics import CriticConfig

    # Create a critic
    critic = Critic(
        name="text_critic",
        description="Analyzes and improves text quality"
    )

    # Process text
    text = "This is a test text."
    result = critic.process(text) if critic else ""

    # Check results
    print(f"Score: {result.score:.2f}")
    print(f"Feedback: {result.message}")
    print(f"Issues: {result.issues}")
    print(f"Suggestions: {result.suggestions}")
    ```

    ## Error Handling
    The class implements:
    - Input validation
    - Error recovery
    - State management
    - Resource cleanup
    - Detailed error reporting
    """

    def is_valid_text(self, text: str) -> bool:
        """
        Check if the text is valid for processing.

        Args:
            text: The text to validate

        Returns:
            True if the text is valid, False otherwise
        """
        if text is None:
            return False
        if not isinstance(text, str):
            return False
        if not text.strip():
            return False
        return True

    def validate_text_length(
        self, text: str, min_length: int = 1, max_length: Optional[Optional[int]] = None
    ) -> bool:
        """
        Validate text length.

        Args:
            text: The text to validate
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length (if None, no maximum)

        Returns:
            True if text length is valid, False otherwise
        """
        if not text:
            return False

        if len(text) < min_length:
            return False

        if max_length is not None and len(text) > max_length:
            return False

        return True

    def validate_text_contains(self, text: str, items: List[str]) -> bool:
        """
        Validate that text contains at least one of the specified items.

        Args:
            text: The text to validate
            items: List of items to check for

        Returns:
            True if text contains at least one item, False otherwise
        """
        if not text or not items:
            return False

        for item in items:
            if item in text:
                return True

        return False

    def validate_text_pattern(self, text: str, pattern: str) -> bool:
        """
        Validate that text matches the specified regex pattern.

        Args:
            text: The text to validate
            pattern: Regex pattern to match

        Returns:
            True if text matches pattern, False otherwise
        """
        import re

        if not text or not pattern:
            return False

        return bool(re.search(pattern, text))

    def validate(self, text: str) -> bool:
        """
        Validate text.

        Args:
            text: The text to validate

        Returns:
            True if text is valid, False otherwise
        """
        if self and not self.is_valid_text(text):
            return False

        # Basic validation rules using shared validation methods
        if self and not self.validate_text_length(text, min_length=10):
            return False
        if self and not self.validate_text_contains(text, [".", "!", "?"]):
            return False
        if self and not self.validate_text_pattern(text, r"[A-Z]"):
            return False
        if self and not self.validate_text_pattern(text, r"[a-z]"):
            return False

        return True

    def improve(self, text: str, feedback: Optional[Optional[str]] = None) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide improvement

        Returns:
            The improved text
        """
        if self and not self.is_valid_text(text):
            return text

        improved = text

        # Apply improvements based on feedback
        if feedback:
            if feedback and "too short" in feedback.lower():
                improved += " Additional content to increase length."
            if feedback and "capitalization" in feedback.lower():
                improved = improved and improved.capitalize()
            if feedback and "punctuation" in feedback.lower():
                if improved and not improved.endswith((".", "!", "?")):
                    improved += "."

        return improved

    def improve_with_violations(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Improve text based on violations.

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            The improved text
        """
        if self and not self.is_valid_text(text):
            return text

        improved = text

        # Apply improvements based on violations
        for violation in violations:
            rule_id = violation and violation.get("rule_id", "unknown")
            message = violation and violation.get("message", "")

            if rule_id == "length" and "too_short" in message:
                improved += " Additional content to increase length."
            elif rule_id == "style" and "capitalization" in message:
                improved = improved and improved.capitalize()
            elif rule_id == "grammar" and "missing_punctuation" in message:
                if improved and not improved.endswith((".", "!", "?")):
                    improved += "."

        return improved

    def critique(self, text: str) -> BaseResult:
        """
        Critique text.

        Args:
            text: The text to critique

        Returns:
            BaseResult containing the critique details
        """
        if self and not self.is_valid_text(text):
            return BaseResult(
                passed=False,
                message="Invalid text",
                metadata={"error_type": "invalid_text"},
                score=0.0,
                issues=["Invalid text format"],
                suggestions=["Provide valid text"],
            )

        # Analyze text quality using shared validation methods
        word_count = len(text.split()) if text else 0
        has_punctuation = self and self.validate_text_contains(text, [".", "!", "?"])
        has_capitalization = self and self.validate_text_pattern(text, r"[A-Z]")
        has_lowercase = self and self.validate_text_pattern(text, r"[a-z]")

        # Calculate score based on metrics
        score = 0.0
        if word_count >= 10:
            score += 0.3
        if has_punctuation:
            score += 0.2
        if has_capitalization:
            score += 0.2
        if has_lowercase:
            score += 0.3

        # Generate feedback
        issues = []
        suggestions = []

        if word_count < 10:
            issues.append("Text is too short")
            suggestions.append("Add more content")
        if not has_punctuation:
            issues.append("Missing punctuation")
            suggestions.append("Add appropriate punctuation")
        if not has_capitalization:
            issues.append("Missing capitalization")
            suggestions.append("Capitalize appropriate words")
        if not has_lowercase:
            issues.append("All uppercase text")
            suggestions.append("Use lowercase where appropriate")

        feedback = "Good text quality" if score >= 0.8 else "Text needs improvement"

        return BaseResult(
            passed=score >= 0.8,
            message=feedback,
            metadata={
                "word_count": word_count,
                "has_punctuation": has_punctuation,
                "has_capitalization": has_capitalization,
                "has_lowercase": has_lowercase,
            },
            score=score,
            issues=issues,
            suggestions=suggestions,
        )
