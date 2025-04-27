"""Tests for the domain rules."""

import pytest
from typing import Dict, Any, List, Set
import re

from sifaka.rules.domain import (
    MedicalRule,
    LegalRule,
    PythonProgrammingRule,
    ConsistencyRule,
)
from sifaka.rules.base import RuleResult


class TestMedicalRule(MedicalRule):
    """Test implementation of MedicalRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        issues = []
        found_warning_terms = []

        # Check for warning terms
        for term in self.warning_terms:
            if term in output_lower:
                found_warning_terms.append(term)

        # Check for disclaimer if warning terms are found
        if self.disclaimer_required and found_warning_terms:
            disclaimer_patterns = [
                r"not medical advice",
                r"consult.*doctor",
                r"seek.*professional",
                r"medical disclaimer",
            ]
            has_disclaimer = any(
                re.search(pattern, output_lower) for pattern in disclaimer_patterns
            )

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


class TestLegalRule(LegalRule):
    """Test implementation of LegalRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        issues = []
        citations = []

        # Check for citations
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, output)
            citations.extend(matches)

        # Check for disclaimer if legal terms are found
        if self.disclaimer_required:
            legal_terms_found = any(
                any(term in output.lower() for term in terms) for terms in self.legal_terms.values()
            )

            if legal_terms_found:
                disclaimer_patterns = [
                    r"not legal advice",
                    r"consult.*attorney",
                    r"seek.*counsel",
                    r"legal disclaimer",
                ]
                has_disclaimer = any(
                    re.search(pattern, output.lower()) for pattern in disclaimer_patterns
                )

                if not has_disclaimer:
                    issues.append("Legal disclaimer required but not found")

        if not citations and any(terms in output.lower() for terms in self.legal_terms.values()):
            issues.append("Legal citations required but not found")

        if issues:
            return RuleResult(
                passed=False,
                message="Legal content validation failed",
                metadata={"issues": issues, "citations_found": citations},
            )

        return RuleResult(
            passed=True,
            message="Legal content validation passed",
            metadata={"citations_found": citations},
        )


class TestPythonProgrammingRule(PythonProgrammingRule):
    """Test implementation of PythonProgrammingRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        issues = []

        # Check code style patterns
        for pattern_name, pattern in self.code_style_patterns.items():
            if not re.search(pattern, output, re.MULTILINE):
                issues.append(f"Code style issue: {pattern_name}")

        # Check security patterns
        for pattern_name, pattern in self.security_patterns.items():
            if re.search(pattern, output):
                issues.append(f"Security issue: {pattern_name}")

        # Check performance patterns
        for pattern_name, pattern in self.performance_patterns.items():
            if re.search(pattern, output):
                issues.append(f"Performance issue: {pattern_name}")

        if issues:
            return RuleResult(
                passed=False,
                message="Python code validation failed",
                metadata={"issues": issues},
            )

        return RuleResult(
            passed=True,
            message="Python code validation passed",
            metadata={
                "patterns_checked": {
                    "style": list(self.code_style_patterns.keys()),
                    "security": list(self.security_patterns.keys()),
                    "performance": list(self.performance_patterns.keys()),
                }
            },
        )


class TestConsistencyRule(ConsistencyRule):
    """Test implementation of ConsistencyRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        issues = []
        patterns_found = {}

        # Check consistency patterns
        for pattern_name, pattern in self.consistency_patterns.items():
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                patterns_found[pattern_name] = matches

        # Check for contradictions
        contradictions = []
        output_lower = output.lower()
        for indicator in self.contradiction_indicators:
            if indicator in output_lower:
                contradictions.append(indicator)

        # Check for repetition
        words = output_lower.split()
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))

        if repetition_ratio > self.repetition_threshold:
            issues.append(f"Text is too repetitive (ratio: {repetition_ratio:.2f})")

        if contradictions:
            issues.append("Contradictions found")

        if issues:
            return RuleResult(
                passed=False,
                message="Consistency validation failed",
                metadata={
                    "issues": issues,
                    "patterns_found": patterns_found,
                    "contradictions": contradictions,
                    "repetition_ratio": repetition_ratio,
                },
            )

        return RuleResult(
            passed=True,
            message="Consistency validation passed",
            metadata={
                "patterns_found": patterns_found,
                "repetition_ratio": repetition_ratio,
            },
        )


@pytest.fixture
def medical_rule():
    """Create a TestMedicalRule instance."""
    return TestMedicalRule(
        name="test_medical", description="Test medical rule", disclaimer_required=True
    )


@pytest.fixture
def legal_rule():
    """Create a TestLegalRule instance."""
    return TestLegalRule(name="test_legal", description="Test legal rule", disclaimer_required=True)


@pytest.fixture
def python_rule():
    """Create a TestPythonProgrammingRule instance."""
    return TestPythonProgrammingRule(name="test_python", description="Test Python programming rule")


@pytest.fixture
def consistency_rule():
    """Create a TestConsistencyRule instance."""
    return TestConsistencyRule(
        name="test_consistency", description="Test consistency rule", repetition_threshold=0.3
    )


def test_medical_rule_initialization():
    """Test MedicalRule initialization."""
    custom_terms = {"test": ["test1", "test2"], "another": ["another1", "another2"]}
    custom_warnings = ["warning1", "warning2"]

    rule = TestMedicalRule(
        name="test",
        description="test",
        medical_terms=custom_terms,
        warning_terms=custom_warnings,
        disclaimer_required=True,
    )
    assert rule.name == "test"
    assert rule.medical_terms == custom_terms
    assert rule.warning_terms == custom_warnings
    assert rule.disclaimer_required is True


def test_medical_rule_validation(medical_rule):
    """Test medical rule validation."""
    # Test text without medical content
    safe_text = "This text contains no medical information."
    result = medical_rule.validate(safe_text)
    assert result.passed
    assert not result.metadata["warning_terms_found"]

    # Test text with medical content but no disclaimer
    medical_text = "This treatment can help cure your symptoms."
    result = medical_rule.validate(medical_text)
    assert not result.passed
    assert "disclaimer" in result.metadata["issues"][0]

    # Test text with medical content and proper disclaimer
    valid_text = """
    This treatment can help manage symptoms.
    Note: This is not medical advice. Please consult your doctor.
    """
    result = medical_rule.validate(valid_text)
    assert result.passed
    assert "treatment" in result.metadata["warning_terms_found"]


def test_legal_rule_initialization():
    """Test LegalRule initialization."""
    custom_terms = {"test": ["test1", "test2"], "another": ["another1", "another2"]}
    custom_patterns = [r"\d+ TEST \d+", r"[A-Z]+ v\. [A-Z]+"]

    rule = TestLegalRule(
        name="test",
        description="test",
        legal_terms=custom_terms,
        citation_patterns=custom_patterns,
        disclaimer_required=True,
    )
    assert rule.name == "test"
    assert rule.legal_terms == custom_terms
    assert rule.citation_patterns == custom_patterns
    assert rule.disclaimer_required is True


def test_legal_rule_validation(legal_rule):
    """Test legal rule validation."""
    # Test text without legal content
    safe_text = "This text contains no legal information."
    result = legal_rule.validate(safe_text)
    assert result.passed
    assert not result.metadata["citations_found"]

    # Test text with legal content but no disclaimer/citations
    legal_text = "This statute defines the jurisdiction."
    result = legal_rule.validate(legal_text)
    assert not result.passed
    assert any("disclaimer" in issue for issue in result.metadata["issues"])

    # Test text with legal content and proper format
    valid_text = """
    According to 410 U.S. 113, the jurisdiction is clear.
    Note: This is not legal advice. Please consult an attorney.
    """
    result = legal_rule.validate(valid_text)
    assert result.passed
    assert result.metadata["citations_found"]


def test_python_rule_initialization():
    """Test PythonProgrammingRule initialization."""
    custom_style = {"test_style": r"test\s+pattern"}
    custom_security = {"test_security": r"unsafe\s+pattern"}
    custom_performance = {"test_perf": r"slow\s+pattern"}

    rule = TestPythonProgrammingRule(
        name="test",
        description="test",
        code_style_patterns=custom_style,
        security_patterns=custom_security,
        performance_patterns=custom_performance,
    )
    assert rule.name == "test"
    assert rule.code_style_patterns == custom_style
    assert rule.security_patterns == custom_security
    assert rule.performance_patterns == custom_performance


def test_python_rule_validation(python_rule):
    """Test Python programming rule validation."""
    # Test valid Python code
    valid_code = """
    import os
    from typing import List

    class MyClass:
        def my_function(self):
            my_var = 42
            return my_var
    """
    result = python_rule.validate(valid_code)
    assert result.passed
    assert "patterns_checked" in result.metadata

    # Test code with style issues
    invalid_style = """
    Import os
    class myclass:
        def MyFunction():
            MY_VAR = 42
    """
    result = python_rule.validate(invalid_style)
    assert not result.passed
    assert any("style" in issue.lower() for issue in result.metadata["issues"])

    # Test code with security issues
    security_issues = """
    def unsafe_function():
        eval("print('Hello')")
        exec("x = 1")
    """
    result = python_rule.validate(security_issues)
    assert not result.passed
    assert any("security" in issue.lower() for issue in result.metadata["issues"])


def test_consistency_rule_initialization():
    """Test ConsistencyRule initialization."""
    custom_patterns = {"test_pattern": r"test\s+pattern"}
    custom_indicators = ["but", "however"]

    rule = TestConsistencyRule(
        name="test",
        description="test",
        consistency_patterns=custom_patterns,
        contradiction_indicators=custom_indicators,
        repetition_threshold=0.4,
    )
    assert rule.name == "test"
    assert rule.consistency_patterns == custom_patterns
    assert rule.contradiction_indicators == custom_indicators
    assert rule.repetition_threshold == 0.4


def test_consistency_rule_validation(consistency_rule):
    """Test consistency rule validation."""
    # Test consistent text
    consistent_text = """
    The project is progressing well. We are implementing new features.
    The team is focused on quality. Everything is on track.
    """
    result = consistency_rule.validate(consistent_text)
    assert result.passed
    assert "patterns_found" in result.metadata

    # Test text with contradictions
    contradictory_text = """
    The project is going well. However, we are behind schedule.
    Although the quality is good, there are many bugs.
    """
    result = consistency_rule.validate(contradictory_text)
    assert not result.passed
    assert "contradictions" in result.metadata

    # Test repetitive text
    repetitive_text = "The test is a test that tests testing tests for testing purposes."
    result = consistency_rule.validate(repetitive_text)
    assert not result.passed
    assert result.metadata["repetition_ratio"] > consistency_rule.repetition_threshold


def test_edge_cases():
    """Test edge cases for all rules."""
    rules = [
        TestMedicalRule(name="medical", description="test"),
        TestLegalRule(name="legal", description="test"),
        TestPythonProgrammingRule(name="python", description="test"),
        TestConsistencyRule(name="consistency", description="test"),
    ]

    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "unicode": "Hello 世界",
        "newlines": "Line 1\nLine 2\nLine 3",
        "numbers_only": "123 456 789",
    }

    for rule in rules:
        for case_name, text in edge_cases.items():
            result = rule.validate(text)
            assert isinstance(result, RuleResult)
            assert isinstance(result.passed, bool)
            assert isinstance(result.message, str)
            assert isinstance(result.metadata, dict)


def test_error_handling():
    """Test error handling for all rules."""
    rules = [
        TestMedicalRule(name="medical", description="test"),
        TestLegalRule(name="legal", description="test"),
        TestPythonProgrammingRule(name="python", description="test"),
        TestConsistencyRule(name="consistency", description="test"),
    ]

    invalid_inputs = [None, 123, [], {}]

    for rule in rules:
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                rule.validate(invalid_input)


def test_consistent_results():
    """Test consistency of validation results."""
    rules = [
        TestMedicalRule(name="medical", description="test"),
        TestLegalRule(name="legal", description="test"),
        TestPythonProgrammingRule(name="python", description="test"),
        TestConsistencyRule(name="consistency", description="test"),
    ]

    test_text = """
    According to 410 U.S. 113, the treatment must be prescribed by a doctor.
    This is not medical or legal advice. Please consult professionals.

    def example_function():
        return "Hello, World!"
    """

    for rule in rules:
        # Run validation multiple times
        results = [rule.validate(test_text) for _ in range(3)]

        # All results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result.passed == first_result.passed
            assert result.message == first_result.message
            assert result.metadata == first_result.metadata
