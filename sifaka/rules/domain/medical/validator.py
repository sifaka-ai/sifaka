"""
Medical domain validation.

This module provides validation rules for medical content, including
terminology validation, definition requirements, and term limits.
"""

from typing import Dict, List, Optional, Set, Tuple
import re

from sifaka.rules.base import RuleResult
from ..base import BaseDomainValidator
from .config import MedicalConfig


class MedicalValidator(BaseDomainValidator):
    """Validator for medical content."""

    def __init__(self, config: Optional[MedicalConfig] = None) -> None:
        """Initialize validator with configuration."""
        super().__init__(config or MedicalConfig())
        self._definition_pattern = re.compile(
            r"\b(\w+)\s*(?:\(|\[|\{)[^)}\]]*(?:means?|refers? to|defined as)[^)}\]]*(?:\)|\]|\})"
        )

    def validate(self, content: str, **kwargs) -> RuleResult:
        """Validate medical content."""
        if not self.can_validate(content):
            return RuleResult(
                passed=False,
                message="Invalid content",
                metadata={"error": "Content must be a non-empty string"},
            )

        # Extract and validate medical terms
        medical_terms = self._extract_medical_terms(content)
        term_count = len(medical_terms)

        # Check required terms
        missing_required = self.config.required_terms - medical_terms
        if missing_required:
            return RuleResult(
                passed=False,
                message=f"Missing required medical terms: {', '.join(missing_required)}",
                metadata={
                    "missing_terms": list(missing_required),
                    "found_terms": list(medical_terms),
                },
            )

        # Check prohibited terms
        found_prohibited = medical_terms & self.config.prohibited_terms
        if found_prohibited:
            return RuleResult(
                passed=False,
                message=f"Found prohibited medical terms: {', '.join(found_prohibited)}",
                metadata={
                    "prohibited_terms": list(found_prohibited),
                    "found_terms": list(medical_terms),
                },
            )

        # Check term limits
        min_terms = self.config.min_medical_terms
        max_terms = self.config.max_medical_terms
        if min_terms > 0 and term_count < min_terms:
            return RuleResult(
                passed=False,
                message=f"Found {term_count} medical terms, minimum required is {min_terms}",
                metadata={"found_terms": list(medical_terms)},
            )
        if max_terms > 0 and term_count > max_terms:
            return RuleResult(
                passed=False,
                message=f"Found {term_count} medical terms, maximum allowed is {max_terms}",
                metadata={"found_terms": list(medical_terms)},
            )

        # Check definitions if required
        if self.config.require_definitions:
            undefined_terms = self._find_undefined_terms(content, medical_terms)
            if undefined_terms:
                return RuleResult(
                    passed=False,
                    message=f"Medical terms without definitions: {', '.join(undefined_terms)}",
                    metadata={
                        "undefined_terms": list(undefined_terms),
                        "found_terms": list(medical_terms),
                    },
                )

        return RuleResult(
            passed=True,
            message=f"Found {term_count} valid medical terms",
            metadata={"found_terms": list(medical_terms)},
        )

    def get_validation_errors(self, content: str) -> List[str]:
        """Get list of validation errors."""
        result = self.validate(content)
        if result.passed:
            return []
        return [result.message]

    def _extract_medical_terms(self, content: str) -> Set[str]:
        """Extract medical terms from content."""
        terms = set()
        content_lower = content.lower() if not self.config.case_sensitive else content

        # Check for terms with medical affixes
        words = re.findall(r"\b\w+\b", content_lower)
        for word in words:
            # Check prefixes
            for prefix in self.config.medical_prefixes:
                if word.startswith(prefix):
                    terms.add(word)
                    break

            # Check suffixes
            for suffix in self.config.medical_suffixes:
                if word.endswith(suffix):
                    terms.add(word)
                    break

        # Add explicitly required terms found in content
        for term in self.config.required_terms:
            term_to_check = term.lower() if not self.config.case_sensitive else term
            if term_to_check in content_lower:
                terms.add(term)

        return terms

    def _find_undefined_terms(self, content: str, terms: Set[str]) -> Set[str]:
        """Find medical terms without definitions."""
        undefined = set()
        definitions = set(
            match.group(1).lower() for match in self._definition_pattern.finditer(content)
        )

        for term in terms:
            if term.lower() not in definitions:
                undefined.add(term)

        return undefined
