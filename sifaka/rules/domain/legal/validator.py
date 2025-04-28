"""
Legal domain validation.

This module provides validation rules for legal content, including
citation validation, legal terminology requirements, and disclaimer checks.
"""

from typing import Dict, List, Optional, Set, Tuple
import re

from sifaka.rules.base import RuleResult
from ..base import BaseDomainValidator
from .config import LegalConfig


class LegalValidator(BaseDomainValidator):
    """Validator for legal content."""

    def __init__(self, config: Optional[LegalConfig] = None) -> None:
        """Initialize validator with configuration."""
        super().__init__(config or LegalConfig())

    def validate(self, content: str, **kwargs) -> RuleResult:
        """Validate legal content."""
        if not self.can_validate(content):
            return RuleResult(
                passed=False,
                message="Invalid content",
                metadata={"error": "Content must be a non-empty string"},
            )

        # Check citations
        citations = self._extract_citations(content)
        citation_count = len(citations)

        if self.config.min_citations > 0 and citation_count < self.config.min_citations:
            return RuleResult(
                passed=False,
                message=f"Found {citation_count} citations, minimum required is {self.config.min_citations}",
                metadata={"citations": citations},
            )

        if self.config.max_citations > 0 and citation_count > self.config.max_citations:
            return RuleResult(
                passed=False,
                message=f"Found {citation_count} citations, maximum allowed is {self.config.max_citations}",
                metadata={"citations": citations},
            )

        # Check legal terms
        term_results = self._validate_legal_terms(content)
        if not term_results["valid"]:
            return RuleResult(
                passed=False,
                message=term_results["message"],
                metadata=term_results["details"],
            )

        # Check disclaimer if required
        if self.config.disclaimer_required:
            disclaimer_found = self._check_disclaimer(content)
            if not disclaimer_found:
                return RuleResult(
                    passed=False,
                    message="Required legal disclaimer not found",
                    metadata={
                        "required_patterns": list(self.config.disclaimer_patterns),
                    },
                )

        return RuleResult(
            passed=True,
            message="Legal content validation passed",
            metadata={
                "citations": citations,
                "term_analysis": term_results["details"],
            },
        )

    def get_validation_errors(self, content: str) -> List[str]:
        """Get list of validation errors."""
        result = self.validate(content)
        if result.passed:
            return []
        return [result.message]

    def _extract_citations(self, content: str) -> List[str]:
        """Extract legal citations from content."""
        citations = []
        for pattern_name, pattern in self.config.citation_patterns.items():
            matches = re.finditer(pattern, content)
            citations.extend(match.group(0) for match in matches)
        return citations

    def _validate_legal_terms(self, content: str) -> Dict:
        """Validate legal terms in content."""
        content_lower = content.lower()
        results = {
            "valid": True,
            "message": "",
            "details": {
                "required": [],
                "prohibited": [],
                "warning": [],
            },
        }

        # Check required terms
        required_terms = self.config.required_terms["required"]
        found_required = {term for term in required_terms if term.lower() in content_lower}
        missing_required = required_terms - found_required

        if len(found_required) < self.config.min_required_terms:
            results.update(
                {
                    "valid": False,
                    "message": f"Found {len(found_required)} required terms, minimum is {self.config.min_required_terms}",
                    "details": {"missing_required": list(missing_required)},
                }
            )
            return results

        # Check prohibited terms
        prohibited_terms = self.config.required_terms["prohibited"]
        found_prohibited = {term for term in prohibited_terms if term.lower() in content_lower}

        if found_prohibited:
            if (
                self.config.max_prohibited_terms == 0
                or len(found_prohibited) > self.config.max_prohibited_terms
            ):
                results.update(
                    {
                        "valid": False,
                        "message": f"Found prohibited terms: {', '.join(found_prohibited)}",
                        "details": {"prohibited_terms": list(found_prohibited)},
                    }
                )
                return results

        # Check warning terms
        warning_terms = self.config.required_terms["warning"]
        found_warning = {term for term in warning_terms if term.lower() in content_lower}

        # Update details with all findings
        results["details"].update(
            {
                "found_required": list(found_required),
                "missing_required": list(missing_required),
                "found_prohibited": list(found_prohibited),
                "found_warning": list(found_warning),
            }
        )

        return results

    def _check_disclaimer(self, content: str) -> bool:
        """Check if content contains required legal disclaimer."""
        content_lower = content.lower()
        return any(
            re.search(pattern, content_lower, re.IGNORECASE)
            for pattern in self.config.disclaimer_patterns
        )
