"""Tests for the domain rules."""

import pytest
from typing import Dict, Any, List, Set, Protocol, runtime_checkable, Final
from dataclasses import dataclass, field
import re

from sifaka.rules.domain import (
    MedicalRule,
    LegalRule,
    PythonRule,
    ConsistencyRule,
    MedicalConfig,
    LegalConfig,
    PythonConfig,
    ConsistencyConfig,
    MedicalValidator,
    LegalValidator,
    PythonValidator,
    ConsistencyValidator,
)
from sifaka.rules.base import RuleResult, RuleConfig, RuleValidator


# Test configurations
@dataclass(frozen=True)
class TestMedicalConfig(MedicalConfig):
    """Test configuration for medical rules."""

    warning_terms: Set[str] = field(
        default_factory=lambda: {"medication", "treatment", "diagnosis"}
    )
    disclaimer_required: bool = True
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0


@dataclass(frozen=True)
class TestLegalConfig(LegalConfig):
    """Test configuration for legal rules."""

    legal_terms: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "jurisdiction": ["court", "jurisdiction", "legal", "law"],
            "procedure": ["motion", "appeal", "petition", "filing"],
            "parties": ["plaintiff", "defendant", "appellant", "respondent"],
        }
    )
    citation_patterns: List[str] = field(
        default_factory=lambda: [
            r"\d+ U\.S\. \d+",
            r"\d+ F\.\d+",
            r"\d+ [A-Z][a-z]+\. \d+",
            r"\d+ U\.S\.C\. § \d+",
        ]
    )
    disclaimers: List[str] = field(
        default_factory=lambda: [
            "not legal advice",
            "consult.*attorney",
            "seek.*counsel",
            "legal disclaimer",
        ]
    )
    disclaimer_required: bool = True
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0


@dataclass(frozen=True)
class TestPythonConfig(PythonConfig):
    """Test configuration for Python rules."""

    code_style_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "docstring": r'"""[\s\S]*?"""',
            "pep8_imports": r"^import\s+[a-zA-Z0-9_]+$|^from\s+[a-zA-Z0-9_.]+\s+import\s+[a-zA-Z0-9_,\s]+$",
            "snake_case": r"[a-z][a-z0-9_]*$",
            "class_name": r"^[A-Z][a-zA-Z0-9]*$",
        }
    )
    security_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "eval_exec": r"eval\(|exec\(",
            "shell_injection": r"os\.system|subprocess\.call|subprocess\.Popen",
            "sql_injection": r"execute\s*\(.*\%.*\)|execute\s*\(.*\+.*\)",
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0


@dataclass(frozen=True)
class TestConsistencyConfig(ConsistencyConfig):
    """Test configuration for consistency rules."""

    patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "terminology": r"\b(client|customer|user)\b",
            "formatting": r"^[A-Z].*\.$",
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0


# Test validators
class TestMedicalValidator(MedicalValidator):
    """Test implementation of MedicalValidator."""

    def validate(self, text: str) -> RuleResult:
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        text_lower = text.lower()
        issues = []
        found_warning_terms = []

        for term in self.config.warning_terms:
            if term in text_lower:
                found_warning_terms.append(term)

        if self.config.disclaimer_required and found_warning_terms:
            disclaimer_patterns = [
                r"not medical advice",
                r"consult.*doctor",
                r"seek.*professional",
                r"medical disclaimer",
            ]
            has_disclaimer = any(re.search(pattern, text_lower) for pattern in disclaimer_patterns)

            if not has_disclaimer:
                issues.append("Medical disclaimer required but not found")

        if issues:
            return RuleResult(
                passed=False,
                message="Medical content validation failed",
                metadata={"issues": issues, "warning_terms_found": found_warning_terms},
            )

        return RuleResult(
            passed=True,
            message="Medical content validation passed",
            metadata={"warning_terms_found": found_warning_terms},
        )


class TestLegalValidator(LegalValidator):
    """Test implementation of LegalValidator."""

    def validate(self, text: str) -> RuleResult:
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        text_lower = text.lower()
        metadata = {"citations": [], "issues": [], "legal_terms_found": [], "has_disclaimer": False}

        # Check for legal terms
        for category, terms in self.config.legal_terms.items():
            for term in terms:
                if term.lower() in text_lower:
                    metadata["legal_terms_found"].append(term)

        has_legal_terms = len(metadata["legal_terms_found"]) > 0

        # Check for citations
        for pattern in self.config.citation_patterns:
            matches = re.finditer(pattern, text)
            metadata["citations"].extend(match.group(0) for match in matches)

        has_citations = len(metadata["citations"]) > 0

        # Check for disclaimer if required
        if self.config.disclaimer_required:
            metadata["has_disclaimer"] = any(
                re.search(pattern, text_lower) for pattern in self.config.disclaimers
            )
            if not metadata["has_disclaimer"] and (has_legal_terms or has_citations):
                metadata["issues"].append("disclaimer_required")

        if has_legal_terms and not has_citations:
            metadata["issues"].append("missing_citations")
        elif has_citations and not has_legal_terms:
            metadata["issues"].append("missing_legal_terms")

        if not has_legal_terms and not has_citations:
            return RuleResult(passed=True, message="No legal content found", metadata=metadata)

        passed = (has_legal_terms == has_citations) and (
            not self.config.disclaimer_required or metadata["has_disclaimer"]
        )

        message = "Legal content validation " + ("passed" if passed else "failed")
        if not passed:
            message += ": " + ", ".join(metadata["issues"])

        return RuleResult(passed=passed, message=message, metadata=metadata)


# Fixtures
@pytest.fixture
def medical_config() -> TestMedicalConfig:
    """Create a test medical configuration."""
    return TestMedicalConfig()


@pytest.fixture
def legal_config() -> TestLegalConfig:
    """Create a test legal configuration."""
    return TestLegalConfig()


@pytest.fixture
def python_config() -> TestPythonConfig:
    """Create a test Python configuration."""
    return TestPythonConfig()


@pytest.fixture
def consistency_config() -> TestConsistencyConfig:
    """Create a test consistency configuration."""
    return TestConsistencyConfig()


@pytest.fixture
def medical_validator(medical_config: TestMedicalConfig) -> TestMedicalValidator:
    """Create a test medical validator."""
    return TestMedicalValidator(config=medical_config)


@pytest.fixture
def legal_validator(legal_config: TestLegalConfig) -> TestLegalValidator:
    """Create a test legal validator."""
    return TestLegalValidator(config=legal_config)


@pytest.fixture
def medical_rule(
    medical_config: TestMedicalConfig, medical_validator: TestMedicalValidator
) -> MedicalRule:
    """Create a test medical rule."""
    return MedicalRule(
        name="test_medical",
        description="Test medical rule",
        config=medical_config,
        validator=medical_validator,
    )


@pytest.fixture
def legal_rule(legal_config: TestLegalConfig, legal_validator: TestLegalValidator) -> LegalRule:
    """Create a test legal rule."""
    return LegalRule(
        name="test_legal",
        description="Test legal rule",
        config=legal_config,
        validator=legal_validator,
    )


# Tests
def test_medical_rule_initialization(medical_rule: MedicalRule):
    """Test medical rule initialization."""
    assert medical_rule.name == "test_medical"
    assert medical_rule.description == "Test medical rule"
    assert isinstance(medical_rule.config, TestMedicalConfig)
    assert isinstance(medical_rule.validator, TestMedicalValidator)
    assert medical_rule.config.disclaimer_required is True
    assert "medication" in medical_rule.config.warning_terms


def test_medical_rule_validation(medical_rule: MedicalRule):
    """Test medical rule validation."""
    # Test with no medical content
    result = medical_rule.validate("This is a normal text.")
    assert result.passed is True
    assert "warning_terms_found" in result.metadata
    assert len(result.metadata["warning_terms_found"]) == 0

    # Test with medical content but no disclaimer
    result = medical_rule.validate("This medication should be taken twice daily.")
    assert result.passed is False
    assert "warning_terms_found" in result.metadata
    assert "medication" in result.metadata["warning_terms_found"]
    assert "Medical disclaimer required but not found" in result.metadata["issues"]

    # Test with medical content and disclaimer
    result = medical_rule.validate(
        "This medication should be taken twice daily. This is not medical advice, please consult your doctor."
    )
    assert result.passed is True
    assert "medication" in result.metadata["warning_terms_found"]
    assert len(result.metadata.get("issues", [])) == 0


def test_legal_rule_initialization(legal_rule: LegalRule):
    """Test legal rule initialization."""
    assert legal_rule.name == "test_legal"
    assert legal_rule.description == "Test legal rule"
    assert isinstance(legal_rule.config, TestLegalConfig)
    assert isinstance(legal_rule.validator, TestLegalValidator)
    assert legal_rule.config.disclaimer_required is True
    assert "court" in legal_rule.config.legal_terms["jurisdiction"]


def test_legal_rule_validation(legal_rule: LegalRule):
    """Test legal rule validation."""
    # Test with no legal content
    result = legal_rule.validate("This is a normal text.")
    assert result.passed is True
    assert len(result.metadata["legal_terms_found"]) == 0
    assert len(result.metadata["citations"]) == 0

    # Test with legal terms but no citations
    result = legal_rule.validate("The court will hear the case.")
    assert result.passed is False
    assert "court" in result.metadata["legal_terms_found"]
    assert "missing_citations" in result.metadata["issues"]

    # Test with legal terms, citations, and disclaimer
    result = legal_rule.validate(
        "The court in 123 U.S. 456 ruled in favor of the plaintiff. This is not legal advice."
    )
    assert result.passed is True
    assert "court" in result.metadata["legal_terms_found"]
    assert "123 U.S. 456" in result.metadata["citations"]
    assert result.metadata["has_disclaimer"] is True


def test_edge_cases():
    """Test edge cases for domain rules."""
    # Test with empty config
    empty_medical_config = TestMedicalConfig(warning_terms=set())
    empty_medical_validator = TestMedicalValidator(config=empty_medical_config)
    empty_medical_rule = MedicalRule(
        name="empty_medical",
        description="Empty medical rule",
        config=empty_medical_config,
        validator=empty_medical_validator,
    )
    result = empty_medical_rule.validate("Any text")
    assert result.passed is True

    # Test with None input
    with pytest.raises(ValueError):
        empty_medical_rule.validate(None)


def test_error_handling():
    """Test error handling in domain rules."""
    # Test with invalid config
    with pytest.raises(ValueError):
        TestMedicalConfig(cache_size=-1)

    with pytest.raises(ValueError):
        TestLegalConfig(priority=-1)

    # Test with invalid input types
    medical_config = TestMedicalConfig()
    medical_validator = TestMedicalValidator(config=medical_config)
    with pytest.raises(ValueError):
        medical_validator.validate(123)


def test_consistent_results():
    """Test that results are consistent across multiple validations."""
    medical_config = TestMedicalConfig()
    medical_validator = TestMedicalValidator(config=medical_config)
    medical_rule = MedicalRule(
        name="test_medical",
        description="Test medical rule",
        config=medical_config,
        validator=medical_validator,
    )

    text = "This medication requires a prescription. This is not medical advice."
    results = [medical_rule.validate(text) for _ in range(5)]

    # All results should be identical
    first_result = results[0]
    for result in results[1:]:
        assert result.passed == first_result.passed
        assert result.message == first_result.message
        assert result.metadata == first_result.metadata
