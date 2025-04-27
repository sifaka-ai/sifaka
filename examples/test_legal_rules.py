"""Test script for consolidated legal rules."""

# Test both import paths for backward compatibility
print("Testing domain imports:")
from sifaka.rules.domain import (
    LegalRule,
    LegalCitationRule,
    LegalTermsRule,
    create_legal_rule,
    create_legal_citation_rule,
    create_legal_terms_rule,
)

print("✅ Domain imports successful")

print("\nTesting backward compatibility imports:")
from sifaka.rules.legal import (
    LegalCitationRule as LegacyLegalCitationRule,
    LegalTermsRule as LegacyLegalTermsRule,
    create_legal_citation_rule as legacy_create_legal_citation_rule,
    create_legal_terms_rule as legacy_create_legal_terms_rule,
)

print("✅ Legacy imports successful (with expected deprecation warning)")

print("\nTesting top-level imports:")
from sifaka.rules import (
    LegalRule,
    LegalCitationRule,
    LegalTermsRule,
    create_legal_rule,
    create_legal_citation_rule,
    create_legal_terms_rule,
)

print("✅ Top-level imports successful")

# Test creating and using rules
print("\nTesting rule creation and usage:")

# Test sample text
legal_text = """
According to the Supreme Court's decision in 410 U.S. 113, and subsequent rulings
including 505 U.S. 833, the right to privacy is protected under the Constitution.

This document is for informational purposes only and does not constitute legal advice.
Please consult a qualified attorney for legal counsel.

The terms of this agreement are confidential and proprietary.
"""

# Create and test rules
legal_rule = create_legal_rule(
    name="general_legal",
    description="Validates general legal content",
    config={
        "disclaimer_required": True,
    },
)
legal_citation_rule = create_legal_citation_rule(
    name="citations",
    description="Validates legal citations",
    config={
        "min_citations": 1,
    },
)
legal_terms_rule = create_legal_terms_rule(
    name="terminology",
    description="Validates legal terminology",
    config={
        "required_terms": {"confidential"},
        "prohibited_terms": {"unauthorized"},
    },
)

# Test the rules
legal_result = legal_rule._validate_impl(legal_text)
citation_result = legal_citation_rule._validate_impl(legal_text)
terms_result = legal_terms_rule._validate_impl(legal_text)

print(f"Legal rule passed: {legal_result.passed}")
print(f"Citation rule passed: {citation_result.passed}")
print(f"Terms rule passed: {terms_result.passed}")

print("\nTest completed successfully!")
