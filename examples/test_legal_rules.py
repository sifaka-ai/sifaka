#!/usr/bin/env python3
"""
Legal Rules Example for Sifaka.

This example demonstrates:
1. Creating and using different types of legal rules
2. Validating legal text with citations
3. Checking for required legal terms and disclaimers

Usage:
    python legal_rules_example.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
"""

import os
import sys

# Add parent directory to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sifaka.rules import (
    create_legal_citation_rule,
    create_legal_rule,
    create_legal_terms_rule,
)
from sifaka.utils.logging import get_logger

# Initialize logger from Sifaka
logger = get_logger(__name__)

def main():
    """Run the legal rules example."""
    logger.info("Starting legal rules example...")

    # Sample legal text
    legal_text = """
    According to the Supreme Court's decision in 410 U.S. 113, and subsequent rulings
    including 505 U.S. 833, the right to privacy is protected under the Constitution.

    This document is for informational purposes only and does not constitute legal advice.
    Please consult a qualified attorney for legal counsel.

    The terms of this agreement are confidential and proprietary.
    """

    logger.info("Sample legal text:")
    logger.info(f"'{legal_text.strip()}'")

    # Create legal rules
    logger.info("\nCreating legal rules...")

    # General legal rule (requires disclaimer)
    legal_rule = create_legal_rule(
        name="general_legal",
        description="Validates general legal content",
        config={"disclaimer_required": True},
    )

    # Citation rule (requires at least one citation)
    citation_rule = create_legal_citation_rule(
        name="legal_citations",
        description="Validates legal citations",
        config={"min_citations": 1},
    )

    # Terminology rule (checks for required and prohibited terms)
    terms_rule = create_legal_terms_rule(
        name="legal_terminology",
        description="Validates legal terminology",
        config={
            "required_terms": {"confidential"},
            "prohibited_terms": {"unauthorized"},
        },
    )

    # Validate the text with each rule
    logger.info("\nValidating legal text...")

    # General legal validation
    legal_result = legal_rule._validate_impl(legal_text)
    logger.info(f"\nGeneral Legal Rule - Passed: {legal_result.passed}")
    logger.info(f"Message: {legal_result.message}")
    if legal_result.metadata:
        for key, value in legal_result.metadata.items():
            logger.info(f"{key}: {value}")

    # Citation validation
    citation_result = citation_rule._validate_impl(legal_text)
    logger.info(f"\nCitation Rule - Passed: {citation_result.passed}")
    logger.info(f"Message: {citation_result.message}")
    if "citations" in citation_result.metadata:
        logger.info("Citations found:")
        for citation in citation_result.metadata["citations"]:
            logger.info(f"- {citation}")

    # Terminology validation
    terms_result = terms_rule._validate_impl(legal_text)
    logger.info(f"\nTerminology Rule - Passed: {terms_result.passed}")
    logger.info(f"Message: {terms_result.message}")
    if "found_terms" in terms_result.metadata:
        logger.info("Required terms found:")
        for term in terms_result.metadata["found_terms"]:
            logger.info(f"- {term}")

    # Test with invalid text
    invalid_text = "This is a regular text without any legal citations or disclaimers."

    logger.info("\n\nTesting with invalid text:")
    logger.info(f"'{invalid_text}'")

    invalid_legal = legal_rule._validate_impl(invalid_text)
    invalid_citation = citation_rule._validate_impl(invalid_text)
    invalid_terms = terms_rule._validate_impl(invalid_text)

    logger.info(f"\nGeneral Legal Rule - Passed: {invalid_legal.passed}")
    logger.info(f"Citation Rule - Passed: {invalid_citation.passed}")
    logger.info(f"Terminology Rule - Passed: {invalid_terms.passed}")

    logger.info("\nLegal rules example completed.")

if __name__ == "__main__":
    main()
