"""Tests for legal rules."""

import pytest
from typing import Dict, Any, List, Set, Protocol, runtime_checkable, Final
from dataclasses import dataclass, field
import re

from sifaka.rules.legal import (
    LegalCitationRule,
    LegalTermsRule,
    LegalCitationConfig,
    LegalTermsConfig,
    LegalCitationValidator,
    LegalTermsValidator,
    DefaultLegalCitationValidator,
    DefaultLegalTermsValidator,
    create_legal_citation_rule,
    create_legal_terms_rule,
)
from sifaka.rules.base import RuleResult


@pytest.fixture
def citation_config() -> LegalCitationConfig:
    """Create a test citation configuration."""
    return LegalCitationConfig(
        citation_patterns=[
            r"\d+\s+U\.S\.\s+\d+",  # US Reports
            r"\d+\s+F\.\d+d\s+\d+",  # Federal Reporter
        ],
        require_citations=True,
        min_citations=1,
        max_citations=5,
        cache_size=10,
        priority=2,
        cost=1.5,
    )


@pytest.fixture
def terms_config() -> LegalTermsConfig:
    """Create a test terms configuration."""
    return LegalTermsConfig(
        legal_terms={"confidential", "proprietary", "classified"},
        warning_terms={"warning", "caution", "notice"},
        required_terms={"disclaimer"},
        prohibited_terms={"secret", "private"},
        case_sensitive=False,
        cache_size=10,
        priority=2,
        cost=1.5,
    )


@pytest.fixture
def citation_validator(citation_config: LegalCitationConfig) -> LegalCitationValidator:
    """Create a test citation validator."""
    return DefaultLegalCitationValidator(citation_config)


@pytest.fixture
def terms_validator(terms_config: LegalTermsConfig) -> LegalTermsValidator:
    """Create a test terms validator."""
    return DefaultLegalTermsValidator(terms_config)


@pytest.fixture
def citation_rule(
    citation_validator: LegalCitationValidator,
) -> LegalCitationRule:
    """Create a test citation rule."""
    return LegalCitationRule(
        name="Test Citation Rule",
        description="Test legal citation validation",
        validator=citation_validator,
    )


@pytest.fixture
def terms_rule(
    terms_validator: LegalTermsValidator,
) -> LegalTermsRule:
    """Create a test terms rule."""
    return LegalTermsRule(
        name="Test Terms Rule",
        description="Test legal terms validation",
        validator=terms_validator,
    )


def test_citation_config_validation():
    """Test citation configuration validation."""
    # Test valid configuration
    config = LegalCitationConfig(
        citation_patterns=[r"\d+\s+U\.S\.\s+\d+"],
        min_citations=1,
        max_citations=5,
    )
    assert config.citation_patterns == [r"\d+\s+U\.S\.\s+\d+"]
    assert config.min_citations == 1
    assert config.max_citations == 5

    # Test invalid configurations
    with pytest.raises(ValueError, match="citation_patterns must be a list"):
        LegalCitationConfig(citation_patterns="invalid")  # type: ignore

    with pytest.raises(ValueError, match="min_citations must be non-negative"):
        LegalCitationConfig(min_citations=-1)

    with pytest.raises(
        ValueError, match="max_citations must be greater than or equal to min_citations"
    ):
        LegalCitationConfig(min_citations=5, max_citations=1)


def test_terms_config_validation():
    """Test terms configuration validation."""
    # Test valid configuration
    config = LegalTermsConfig(
        legal_terms={"confidential"},
        warning_terms={"warning"},
        required_terms={"disclaimer"},
        prohibited_terms={"secret"},
    )
    assert "confidential" in config.legal_terms
    assert "warning" in config.warning_terms
    assert "disclaimer" in config.required_terms
    assert "secret" in config.prohibited_terms

    # Test invalid configurations
    with pytest.raises(ValueError, match="legal_terms must be a set"):
        LegalTermsConfig(legal_terms=["invalid"])  # type: ignore

    with pytest.raises(ValueError, match="warning_terms must be a set"):
        LegalTermsConfig(warning_terms=["invalid"])  # type: ignore


def test_citation_validation(citation_rule: LegalCitationRule):
    """Test citation validation."""
    # Test valid citations
    text = "According to 123 U.S. 456 and 789 F.3d 123..."
    result = citation_rule.validate(text)
    assert result.passed
    assert len(result.metadata["found_citations"]) == 2

    # Test missing citations
    text = "This text has no legal citations."
    result = citation_rule.validate(text)
    assert not result.passed
    assert "No citations found" in result.message

    # Test too many citations
    text = "123 U.S. 456, 789 F.3d 123, 234 U.S. 567, 345 F.3d 789, 456 U.S. 890, 567 F.3d 901"
    result = citation_rule.validate(text)
    assert not result.passed
    assert "maximum allowed" in result.message


def test_terms_validation(terms_rule: LegalTermsRule):
    """Test terms validation."""
    # Test valid terms
    text = "This document is confidential and contains a disclaimer."
    result = terms_rule.validate(text)
    assert result.passed
    assert "confidential" in result.metadata["found_legal_terms"]

    # Test missing required terms
    text = "This document is confidential but missing required terms."
    result = terms_rule.validate(text)
    assert not result.passed
    assert "Missing required legal terms" in result.message

    # Test prohibited terms
    text = "This document is private and confidential."
    result = terms_rule.validate(text)
    assert not result.passed
    assert "Found prohibited legal terms" in result.message


def test_factory_functions():
    """Test factory functions for creating rules."""
    # Test citation rule creation
    citation_rule = create_legal_citation_rule(
        name="Test Citation Rule",
        description="Test citation validation",
        config=LegalCitationConfig(min_citations=2),
    )
    assert citation_rule.name == "Test Citation Rule"
    assert citation_rule.validator.config.min_citations == 2

    # Test terms rule creation
    terms_rule = create_legal_terms_rule(
        name="Test Terms Rule",
        description="Test terms validation",
        config=LegalTermsConfig(case_sensitive=True),
    )
    assert terms_rule.name == "Test Terms Rule"
    assert terms_rule.validator.config.case_sensitive


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test empty text
    rule = create_legal_citation_rule(
        name="Test Rule",
        description="Test validation",
    )
    result = rule.validate("")
    assert not result.passed
    assert result.metadata["total_citations"] == 0

    # Test invalid input type
    with pytest.raises(ValueError, match="Text must be a string"):
        rule.validate(123)  # type: ignore

    # Test malformed citations
    text = "123 USC 456"  # Doesn't match any pattern
    result = rule.validate(text)
    assert not result.passed
    assert result.metadata["total_citations"] == 0


def test_consistent_results():
    """Test that validation results are consistent."""
    rule = create_legal_citation_rule(
        name="Test Rule",
        description="Test validation",
    )
    text = "According to 123 U.S. 456..."

    # Multiple validations should yield the same result
    result1 = rule.validate(text)
    result2 = rule.validate(text)
    assert result1.passed == result2.passed
    assert result1.message == result2.message
    assert result1.metadata == result2.metadata
