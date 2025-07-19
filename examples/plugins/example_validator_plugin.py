"""Example validator plugin for Sifaka.

This is a complete, working example of how to create a validator plugin.
Copy this file and modify it to create your own validator plugin.
"""

import logging
import re
from typing import Any, Dict

from sifaka.core.models import SifakaResult, ValidationResult
from sifaka.core.plugin_interfaces import PluginMetadata, PluginType, ValidatorPlugin

logger = logging.getLogger(__name__)


class ExampleValidatorPlugin(ValidatorPlugin):
    """Example validator plugin that checks text quality.

    This validator checks for:
    - Minimum word count
    - No profanity
    - Basic grammar (capitalization, punctuation)
    - Email/URL detection
    """

    def __init__(self) -> None:
        """Initialize the example validator plugin."""
        super().__init__()
        self._min_words = 10
        self._profanity_words = {"damn", "hell", "shit", "fuck"}  # Simple example
        self._email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        self._url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="example_validator",
            version="1.0.0",
            author="Sifaka Team",
            description="Example validator plugin for text quality checks",
            plugin_type=PluginType.VALIDATOR,
            dependencies=[],
            sifaka_version=">=0.1.0",
            python_version=">=3.10",
            license="MIT",
            keywords=["example", "validation", "quality", "sifaka"],
            default_config={
                "min_words": 10,
                "check_profanity": True,
                "check_grammar": True,
                "allow_emails": True,
                "allow_urls": True,
            },
        )

    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate text quality.

        Args:
            text: The text to validate
            result: The complete SifakaResult with history

        Returns:
            ValidationResult with validation status
        """
        try:
            issues = []
            score = 1.0

            # Check minimum word count
            words = text.split()
            if len(words) < self._min_words:
                issues.append(
                    f"Text has only {len(words)} words (minimum: {self._min_words})"
                )
                score -= 0.3

            # Check for profanity
            if self.validation_config.get("check_profanity", True):
                profanity_found = [
                    word for word in words if word.lower() in self._profanity_words
                ]
                if profanity_found:
                    issues.append(
                        f"Contains inappropriate language: {', '.join(profanity_found)}"
                    )
                    score -= 0.4

            # Check basic grammar
            if self.validation_config.get("check_grammar", True):
                # Check if text starts with capital letter
                if text and not text[0].isupper():
                    issues.append("Text should start with a capital letter")
                    score -= 0.1

                # Check if text ends with punctuation
                if text and text[-1] not in ".!?":
                    issues.append("Text should end with proper punctuation")
                    score -= 0.1

            # Check for emails/URLs if not allowed
            if not self.validation_config.get("allow_emails", True):
                if self._email_pattern.search(text):
                    issues.append("Email addresses are not allowed")
                    score -= 0.2

            if not self.validation_config.get("allow_urls", True):
                if self._url_pattern.search(text):
                    issues.append("URLs are not allowed")
                    score -= 0.2

            # Ensure score doesn't go below 0
            score = max(0.0, score)

            # Text passes if score is above 0.5
            passed = score > 0.5

            # Create details
            if issues:
                details = f"Validation issues found: {'; '.join(issues)}"
            else:
                details = "Text passes all validation checks."

            return ValidationResult(
                validator=self.name,
                passed=passed,
                score=score,
                details=details,
                metadata={
                    "word_count": len(words),
                    "issues_count": len(issues),
                    "checks_performed": [
                        "word_count",
                        "profanity"
                        if self.validation_config.get("check_profanity", True)
                        else None,
                        "grammar"
                        if self.validation_config.get("check_grammar", True)
                        else None,
                        "emails"
                        if not self.validation_config.get("allow_emails", True)
                        else None,
                        "urls"
                        if not self.validation_config.get("allow_urls", True)
                        else None,
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Error in example validator: {e}")
            return ValidationResult(
                validator=self.name,
                passed=False,
                score=0.0,
                details=f"Error during validation: {str(e)}",
                metadata={"error": str(e)},
            )

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        if "min_words" in config:
            if not isinstance(config["min_words"], int) or config["min_words"] < 1:
                raise ValueError("min_words must be a positive integer")

        # Validate boolean flags
        bool_flags = ["check_profanity", "check_grammar", "allow_emails", "allow_urls"]
        for flag in bool_flags:
            if flag in config and not isinstance(config[flag], bool):
                raise ValueError(f"{flag} must be a boolean")

        return True

    def _on_initialize(self) -> None:
        """Initialize plugin-specific settings."""
        self._min_words = self.validation_config.get("min_words", 10)

        # Update profanity list if provided
        if "profanity_words" in self.validation_config:
            custom_words = self.validation_config["profanity_words"]
            if isinstance(custom_words, list):
                self._profanity_words.update(custom_words)

        logger.info(f"ExampleValidatorPlugin initialized: min_words={self._min_words}")


# Example usage (for testing - remove in production)
if __name__ == "__main__":
    import asyncio
    from datetime import datetime

    async def test_plugin():
        """Test the example validator plugin."""
        plugin = ExampleValidatorPlugin()
        plugin.initialize()
        plugin.activate()

        # Test with good text
        good_text = (
            "This is a well-written text that should pass all validation checks."
        )
        result = SifakaResult(
            id="test",
            original_text=good_text,
            final_text=good_text,
            iteration=1,
            processing_time=0.1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            generations=[],
            critiques=[],
            validations=[],
        )

        validation = await plugin.validate(good_text, result)
        print(f"Good text - Passed: {validation.passed}, Score: {validation.score}")
        print(f"Details: {validation.details}")

        # Test with bad text
        bad_text = "short text"
        validation = await plugin.validate(bad_text, result)
        print(f"Bad text - Passed: {validation.passed}, Score: {validation.score}")
        print(f"Details: {validation.details}")

    asyncio.run(test_plugin())
