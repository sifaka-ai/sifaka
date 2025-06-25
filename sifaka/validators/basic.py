"""Basic validators for common text quality checks."""

import re
from typing import List, Optional, Tuple

from ..core.models import SifakaResult
from .base import BaseValidator, ValidatorConfig


class LengthValidator(BaseValidator):
    """Validates text length requirements."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        config: Optional[ValidatorConfig] = None,
    ):
        super().__init__(config)
        self.min_length = min_length
        self.max_length = max_length

        if min_length is not None and min_length < 0:
            raise ValueError("min_length must be non-negative")
        if max_length is not None and max_length < 0:
            raise ValueError("max_length must be non-negative")
        if (
            min_length is not None
            and max_length is not None
            and min_length > max_length
        ):
            raise ValueError("min_length cannot be greater than max_length")

    @property
    def name(self) -> str:
        return "length"

    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> Tuple[bool, float, str]:
        """Check if text meets length requirements."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        length = len(text)
        passed = True
        details_parts = [f"Text length: {length} characters"]

        # Calculate score based on how well length matches requirements
        score = 1.0

        if self.min_length is not None and length < self.min_length:
            passed = False
            details_parts.append(f"(minimum required: {self.min_length})")
            score = length / self.min_length

        if self.max_length is not None and length > self.max_length:
            passed = False
            details_parts.append(f"(maximum allowed: {self.max_length})")
            # Don't reduce score if we're already over min length
            if self.min_length is None or length >= self.min_length:
                score = self.max_length / length

        # Perfect score if within range
        if passed and self.min_length is not None and self.max_length is not None:
            # Calculate how centered the length is within the range
            range_size = self.max_length - self.min_length
            if range_size > 0:
                center = (self.min_length + self.max_length) / 2
                deviation = abs(length - center) / (range_size / 2)
                score = 1.0 - (deviation * 0.1)  # Small penalty for being off-center

        details = " ".join(details_parts)
        return passed, score, details


class ContentValidator(BaseValidator):
    """Validates required and forbidden content."""

    def __init__(
        self,
        required_terms: Optional[List[str]] = None,
        forbidden_terms: Optional[List[str]] = None,
        case_sensitive: bool = False,
        config: Optional[ValidatorConfig] = None,
    ):
        super().__init__(config)
        self.required_terms = required_terms or []
        self.forbidden_terms = forbidden_terms or []
        self.case_sensitive = case_sensitive

        # Validate input terms
        for term in self.required_terms:
            if not isinstance(term, str):
                raise TypeError(
                    f"All required_terms must be strings, got {type(term).__name__}"
                )

        for term in self.forbidden_terms:
            if not isinstance(term, str):
                raise TypeError(
                    f"All forbidden_terms must be strings, got {type(term).__name__}"
                )

    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> Tuple[bool, float, str]:
        """Check if text contains required terms and avoids forbidden ones."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        check_text = text if self.case_sensitive else text.lower()
        passed = True
        details = []

        # Check required terms
        missing_terms = []
        # Pre-process terms once if case-insensitive
        required_check_terms = (
            self.required_terms
            if self.case_sensitive
            else [term.lower() for term in self.required_terms]
        )

        for i, check_term in enumerate(required_check_terms):
            if check_term not in check_text:
                missing_terms.append(self.required_terms[i])
                passed = False

        if missing_terms:
            details.append(f"Missing required terms: {missing_terms}")

        # Check forbidden terms
        found_forbidden = []
        # Pre-process terms once if case-insensitive
        forbidden_check_terms = (
            self.forbidden_terms
            if self.case_sensitive
            else [term.lower() for term in self.forbidden_terms]
        )

        for i, check_term in enumerate(forbidden_check_terms):
            if check_term in check_text:
                found_forbidden.append(self.forbidden_terms[i])
                passed = False

        if found_forbidden:
            details.append(f"Contains forbidden terms: {found_forbidden}")

        if not details:
            details.append("All content requirements met")

        # Calculate score based on requirements met
        total_checks = len(self.required_terms) + len(self.forbidden_terms)
        if total_checks == 0:
            score = 1.0
        else:
            failed_checks = len(missing_terms) + len(found_forbidden)
            score = max(0.0, (total_checks - failed_checks) / total_checks)

        return passed, score, "; ".join(details)

    @property
    def name(self) -> str:
        return "content"


class FormatValidator(BaseValidator):
    """Validates text format and structure."""

    def __init__(
        self,
        required_sections: Optional[List[str]] = None,
        min_paragraphs: Optional[int] = None,
        max_paragraphs: Optional[int] = None,
        config: Optional[ValidatorConfig] = None,
    ):
        super().__init__(config)
        self.required_sections = required_sections or []
        self.min_paragraphs = min_paragraphs
        self.max_paragraphs = max_paragraphs

    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> Tuple[bool, float, str]:
        """Check text format and structure."""
        passed = True
        details = []

        # Count paragraphs (split by double newlines or single newlines)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
        para_count = len(paragraphs)

        details.append(f"Paragraphs: {para_count}")

        # Check paragraph count
        if self.min_paragraphs is not None and para_count < self.min_paragraphs:
            passed = False
            details.append(f"Need at least {self.min_paragraphs} paragraphs")

        if self.max_paragraphs is not None and para_count > self.max_paragraphs:
            passed = False
            details.append(f"Too many paragraphs (max: {self.max_paragraphs})")

        # Check required sections (case-insensitive search)
        missing_sections = []
        lower_text = text.lower()
        for section in self.required_sections:
            if section.lower() not in lower_text:
                missing_sections.append(section)
                passed = False

        if missing_sections:
            details.append(f"Missing sections: {missing_sections}")

        # Calculate score
        total_checks = (
            (1 if self.min_paragraphs is not None else 0)
            + (1 if self.max_paragraphs is not None else 0)
            + len(self.required_sections)
        )

        if total_checks == 0:
            score = 1.0
        else:
            failed_checks = 0
            if self.min_paragraphs is not None and para_count < self.min_paragraphs:
                failed_checks += 1
            if self.max_paragraphs is not None and para_count > self.max_paragraphs:
                failed_checks += 1
            failed_checks += len(missing_sections)

            score = max(0.0, (total_checks - failed_checks) / total_checks)

        return passed, score, "; ".join(details)

    @property
    def name(self) -> str:
        return "format"
