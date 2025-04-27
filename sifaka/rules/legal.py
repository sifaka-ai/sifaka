"""
Legal-specific rules for Sifaka.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field
from sifaka.rules.base import Rule, RuleResult


class LegalCitationRule(Rule):
    """Rule that validates legal citations in the output."""

    citation_patterns: List[str] = Field(
        default=[], description="List of regex patterns for legal citations"
    )
    compiled_patterns: List[re.Pattern] = []

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        config = config or {}
        self.citation_patterns = config.get(
            "citation_patterns",
            [
                r"\d+\s+U\.S\.\s+\d+",  # US Reports citations
                r"\d+\s+S\.\s*Ct\.\s+\d+",  # Supreme Court Reporter
                r"\d+\s+F\.\d+d\s+\d+",  # Federal Reporter citations
                r"\d+\s+F\.\s*Supp\.\s+\d+",  # Federal Supplement
            ],
        )

        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.citation_patterns]

    def _validate_impl(self, output: str) -> RuleResult:
        """
        Validate that legal citations in the output are properly formatted.

        Args:
            output: The text to validate

        Returns:
            RuleResult with citation validation results
        """
        try:
            # Find all citations
            found_citations = []
            for pattern in self.compiled_patterns:
                found_citations.extend(pattern.findall(output))

            # Check if citations are properly formatted
            invalid_citations = []
            for citation in found_citations:
                # Add specific validation logic here if needed
                # For now, we just ensure they match the patterns
                if not any(pattern.match(citation) for pattern in self.compiled_patterns):
                    invalid_citations.append(citation)

            passed = len(invalid_citations) == 0

            return RuleResult(
                passed=passed,
                message=(
                    f"Found {len(invalid_citations)} invalid citations"
                    if not passed
                    else f"All {len(found_citations)} citations are valid"
                ),
                metadata={
                    "found_citations": found_citations,
                    "invalid_citations": invalid_citations,
                    "total_citations": len(found_citations),
                },
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during citation validation: {str(e)}",
                metadata={"error": str(e)},
            )


# Convenience function for use in the API
def legal_citation_check(output: str, **kwargs) -> RuleResult:
    """
    Check that all legal citations in the output are properly formatted.

    Args:
        output (str): The LLM output to validate
        **kwargs: Additional context for validation

    Returns:
        RuleResult: The result of the validation
    """
    rule = LegalCitationRule(
        name="Legal Citation Check", description="Check legal citations in the output"
    )
    return rule._validate_impl(output)


class LegalRule(Rule):
    """Rule that validates if text contains legal terms."""

    def __init__(self, legal_terms: Optional[List[str]] = None, cache_size: int = 0):
        """
        Initialize the legal rule.

        Args:
            legal_terms: List of legal terms to check for. If None, uses default terms.
            cache_size: Size of the validation cache. Defaults to 0 (no caching).
        """
        super().__init__(cache_size=cache_size)
        self.legal_terms = (
            legal_terms
            if legal_terms is not None
            else ["confidential", "proprietary", "classified", "restricted", "private", "sensitive"]
        )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate if the output contains legal terms.

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult indicating if the text is safe
        """
        found_terms = []
        lower_output = output.lower()

        for term in self.legal_terms:
            if term.lower() in lower_output:
                found_terms.append(term)

        is_safe = len(found_terms) == 0
        return RuleResult(
            is_safe=is_safe,
            message=(
                f"Found legal terms: {', '.join(found_terms)}"
                if found_terms
                else "No legal terms found"
            ),
            metadata={"found_terms": found_terms},
        )
