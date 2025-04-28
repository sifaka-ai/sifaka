"""
Repetition pattern validation.

This module provides validator classes for detecting and validating repetitive patterns in text,
including exact repetitions, alternating patterns, and custom patterns.
"""

from typing import List, Tuple

from sifaka.rules.base import BaseValidator, RuleResult
from .config import RepetitionConfig


class RepetitionValidator(BaseValidator[str]):
    """Validator for repetitive patterns in text."""

    def __init__(self, config: RepetitionConfig) -> None:
        """Initialize with repetition configuration."""
        self._config = config

    @property
    def config(self) -> RepetitionConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for repetitive patterns."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Process text according to configuration
        if not self.config.case_sensitive:
            text = text.lower()

        matches = []
        match_count = 0

        if self.config.pattern_type == "custom" and self.config.custom_pattern:
            # Use custom pattern
            pattern = self.config.custom_pattern
            matches = self._find_pattern_matches(text, pattern)
            match_count = len(matches)
        else:
            # Find repeating or alternating patterns
            for i in range(len(text) - self.config.pattern_length + 1):
                if not self.config.allow_overlap and any(i < end for _, end in matches):
                    continue

                pattern = text[i : i + self.config.pattern_length]
                if self._is_valid_pattern(text, pattern, i):
                    matches.append((i, i + len(pattern)))
                    match_count += 1

        return RuleResult(
            passed=True,  # Pattern detection always passes, just reports findings
            message=f"Found {match_count} pattern matches",
            metadata={
                "match_count": match_count,
                "matches_found": [
                    text[start:end] for start, end in matches[:10]
                ],  # Limit to first 10
                "pattern_type": self.config.pattern_type,
                "pattern_length": self.config.pattern_length,
            },
        )

    def _is_valid_pattern(self, text: str, pattern: str, start_pos: int) -> bool:
        """Check if a pattern is valid according to configuration."""
        if self.config.pattern_type == "repeat":
            # Look for exact repetitions
            return text.count(pattern) > 1
        elif self.config.pattern_type == "alternate":
            # Look for alternating patterns
            if len(text) < start_pos + len(pattern) * 2:
                return False
            next_chunk = text[start_pos + len(pattern) : start_pos + len(pattern) * 2]
            return pattern != next_chunk and text.count(pattern + next_chunk) > 1
        return False

    def _find_pattern_matches(self, text: str, pattern: str) -> List[Tuple[int, int]]:
        """Find all occurrences of a pattern in text."""
        matches = []
        start = 0
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            if not self.config.allow_overlap:
                matches.append((pos, pos + len(pattern)))
                start = pos + len(pattern)
            else:
                matches.append((pos, pos + len(pattern)))
                start = pos + 1
        return matches
