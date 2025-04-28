"""
Symmetry pattern validation.

This module provides validator classes for detecting and validating symmetry patterns in text,
including horizontal and vertical mirror symmetry.
"""

from sifaka.rules.base import BaseValidator, RuleResult
from .config import SymmetryConfig


class SymmetryValidator(BaseValidator[str]):
    """Validator for text symmetry patterns."""

    def __init__(self, config: SymmetryConfig) -> None:
        """Initialize with symmetry configuration."""
        self._config = config

    @property
    def config(self) -> SymmetryConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text symmetry."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Process text according to configuration
        processed_text = text
        if not self.config.preserve_case:
            processed_text = processed_text.lower()
        if self.config.ignore_punctuation:
            processed_text = "".join(c for c in processed_text if c.isalnum() or c.isspace())

        lines = processed_text.split("\n")
        symmetry_score = 0.0

        if self.config.mirror_mode in {"horizontal", "both"}:
            # Check horizontal symmetry
            for line in lines:
                if not self.config.preserve_whitespace:
                    line = line.strip()
                reversed_line = line[::-1]
                similarity = self._calculate_similarity(line, reversed_line)
                symmetry_score += similarity
            symmetry_score /= len(lines) if lines else 1

        if self.config.mirror_mode in {"vertical", "both"}:
            # Check vertical symmetry
            reversed_lines = lines[::-1]
            vertical_score = (
                sum(self._calculate_similarity(l1, l2) for l1, l2 in zip(lines, reversed_lines))
                / len(lines)
                if lines
                else 0
            )

            if self.config.mirror_mode == "both":
                symmetry_score = (symmetry_score + vertical_score) / 2
            else:
                symmetry_score = vertical_score

        is_symmetric = symmetry_score >= self.config.symmetry_threshold

        return RuleResult(
            passed=is_symmetric,
            message=(
                "Text meets symmetry requirements"
                if is_symmetric
                else "Text does not meet symmetry requirements"
            ),
            metadata={
                "symmetry_score": symmetry_score,
                "mirror_mode": self.config.mirror_mode,
                "original_text": text,
                "processed_text": processed_text,
            },
        )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two strings."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        return matches / max(len(text1), len(text2))
