"""Tests for the domain rules."""

import pytest
from typing import Dict, Any, List, Set
import re

from sifaka.rules.domain import (
    MedicalRule,
    LegalRule,
    PythonRule,
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
    """Test legal rule implementation."""

    def __init__(self, name: str = "test_legal", description: str = "Test legal rule", **kwargs):
        """Initialize the test legal rule."""
        super().__init__(name=name, description=description, **kwargs)

        # Only set default patterns if not provided in kwargs
        if "legal_terms" not in kwargs:
            self.legal_terms = {
                "jurisdiction": ["court", "jurisdiction", "legal", "law"],
                "procedure": ["motion", "appeal", "petition", "filing"],
                "parties": ["plaintiff", "defendant", "appellant", "respondent"],
            }

        if "citation_patterns" not in kwargs:
            self.citation_patterns = [
                r"\d+ U\.S\. \d+",  # Supreme Court
                r"\d+ F\.\d+",  # Federal Reporter
                r"\d+ [A-Z][a-z]+\. \d+",  # State Reporter
                r"\d+ U\.S\.C\. § \d+",  # Statute
            ]

        if "disclaimers" not in kwargs:
            self.disclaimers = [
                "not legal advice",
                "consult.*attorney",
                "seek.*counsel",
                "legal disclaimer",
            ]

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        metadata = {"citations": [], "issues": [], "legal_terms_found": [], "has_disclaimer": False}

        # Common phrases to ignore
        common_phrases = [
            "legal information",
            "legal content",
            "legal text",
            "legal document",
            "legal advice",  # When not part of a disclaimer
        ]

        # Check for legal terms
        for category, terms in self.legal_terms.items():
            for term in terms:
                term_lower = term.lower()
                # Skip if it's just a common phrase
                if term_lower == "legal" and any(
                    phrase in output_lower for phrase in common_phrases
                ):
                    continue
                # For other terms, check if they appear in the text
                if term_lower in output_lower:
                    # Make sure it's not part of a common phrase
                    is_common_phrase = False
                    for phrase in common_phrases:
                        if term_lower in phrase and phrase in output_lower:
                            is_common_phrase = True
                            break
                    if not is_common_phrase:
                        metadata["legal_terms_found"].append(term)

        has_legal_terms = len(metadata["legal_terms_found"]) > 0

        # Check for citations
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, output)
            metadata["citations"].extend(match.group(0) for match in matches)

        has_citations = len(metadata["citations"]) > 0

        # Check for disclaimer if required
        if self.disclaimer_required:
            metadata["has_disclaimer"] = any(
                re.search(pattern, output_lower) for pattern in self.disclaimers
            )
            if not metadata["has_disclaimer"] and (has_legal_terms or has_citations):
                metadata["issues"].append("disclaimer_required")

        # Validate based on presence of legal content
        if has_legal_terms and not has_citations:
            metadata["issues"].append("missing_citations")
        elif has_citations and not has_legal_terms:
            metadata["issues"].append("missing_legal_terms")

        # If no legal content is found at all, pass
        if not has_legal_terms and not has_citations:
            return RuleResult(passed=True, message="No legal content found", metadata=metadata)

        # Legal content is present, check if it passes all requirements
        passed = (has_legal_terms == has_citations) and (  # Both present or both absent
            not self.disclaimer_required or metadata["has_disclaimer"]
        )  # Disclaimer if needed

        message = "Legal content validation " + ("passed" if passed else "failed")
        if not passed:
            message += ": " + ", ".join(metadata["issues"])

        return RuleResult(passed=passed, message=message, metadata=metadata)


class TestPythonProgrammingRule(PythonRule):
    """Test implementation of PythonRule."""

    def __init__(self, **kwargs):
        """Initialize the Python programming rule."""
        super().__init__(**kwargs)
        self.code_style_patterns = kwargs.get(
            "code_style_patterns",
            {
                "docstring": r'"""[\s\S]*?"""',
                "pep8_imports": r"^import\s+[a-zA-Z0-9_]+$|^from\s+[a-zA-Z0-9_.]+\s+import\s+[a-zA-Z0-9_,\s]+$",
                "snake_case": r"[a-z][a-z0-9_]*$",
                "class_name": r"^[A-Z][a-zA-Z0-9]*$",
            },
        )
        self.security_patterns = kwargs.get(
            "security_patterns",
            {
                "eval_exec": r"eval\(|exec\(",
                "shell_injection": r"os\.system|subprocess\.call|subprocess\.Popen",
                "sql_injection": r"execute\s*\(.*\%.*\)|execute\s*\(.*\+.*\)",
            },
        )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        issues = []
        metadata = {
            "style_issues": [],
            "security_issues": [],
            "patterns_checked": {
                "style": list(self.code_style_patterns.keys()),
                "security": list(self.security_patterns.keys()),
            },
        }

        # Skip style checks for test files
        is_test_file = "test" in output.lower()

        # Code style checks
        if not is_test_file:
            # Check for docstring
            if not re.search(self.code_style_patterns["docstring"], output):
                issues.append("Missing docstring")
                metadata["style_issues"].append("missing_docstring")

            # Check imports
            import_lines = [
                line.strip()
                for line in output.split("\n")
                if line.strip().startswith(("import", "from"))
            ]
            for line in import_lines:
                if not re.match(self.code_style_patterns["pep8_imports"], line):
                    issues.append(f"Non-PEP8 import style: {line}")
                    metadata["style_issues"].append("non_pep8_import")

        # Security checks (always performed)
        for check_name, pattern in self.security_patterns.items():
            if re.search(pattern, output):
                issues.append(f"Security issue found: {check_name}")
                metadata["security_issues"].append(check_name)

        if issues:
            return RuleResult(
                passed=False,
                message="Code validation failed: " + "; ".join(issues),
                metadata=metadata,
            )

        return RuleResult(passed=True, message="Code validation passed", metadata=metadata)


class TestConsistencyRule(ConsistencyRule):
    """Test implementation of ConsistencyRule."""

    def __init__(
        self, name: str = "test_consistency", description: str = "Test consistency rule", **kwargs
    ):
        """Initialize the test consistency rule."""
        super().__init__(name=name, description=description, **kwargs)

        # Only set default patterns if not provided in kwargs
        if "consistency_patterns" not in kwargs:
            self.consistency_patterns = {
                "tense": r"\b(is|are|was|were)\b",
                "person": r"\b(I|we|he|she|they)\b",
                "voice": r"\b(by|was made|were created)\b",
            }

        if "contradiction_indicators" not in kwargs:
            self.contradiction_indicators = [
                "but",
                "however",
                "although",
                "nevertheless",
                "on the other hand",
                "in contrast",
                "despite",
                "yet",
                "while",
                "whereas",
            ]

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        issues = {}
        patterns_found = {}

        # Check consistency patterns
        for pattern_name, pattern in self.consistency_patterns.items():
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                patterns_found[pattern_name] = matches

                # Check for inconsistencies
                if pattern_name == "person":
                    # Convert matches to lowercase for comparison
                    unique_persons = set(m.lower() for m in matches)
                    # Only flag if mixing first/third person in the same context
                    if ("i" in unique_persons or "we" in unique_persons) and (
                        "he" in unique_persons or "she" in unique_persons
                    ):
                        issues["inconsistent_person"] = matches
                elif pattern_name == "tense":
                    # Group tenses by time (present, past)
                    tense_groups = {
                        "present": {"is", "are", "am"},  # All present tense forms
                        "past": {"was", "were"},  # All past tense forms
                    }

                    # Convert matches to lowercase
                    tenses = [t.lower() for t in matches]

                    # Count tenses by time group
                    counts = {group: 0 for group in tense_groups}
                    for tense in tenses:
                        for group, forms in tense_groups.items():
                            if tense in forms:
                                counts[group] += 1

                    # Only flag if mixing different time groups significantly
                    # Ignore variations within the same time group (e.g., is/are is fine)
                    used_groups = [(group, count) for group, count in counts.items() if count > 0]
                    if len(used_groups) > 1:  # More than one time group used
                        total = sum(count for _, count in used_groups)
                        max_count = max(count for _, count in used_groups)
                        # Only flag if no time group is clearly dominant (>75%)
                        if max_count / total < 0.75:
                            # Check if it's just mixing present tense forms
                            if all(group == "present" for group, _ in used_groups):
                                continue  # Skip flagging if all forms are present tense
                            issues["inconsistent_tense"] = matches

        # Check for contradictions
        contradictions = []
        output_lower = output.lower()
        for indicator in self.contradiction_indicators:
            if indicator in output_lower:
                # Check if it's actually a contradiction or just a transition
                if not any(
                    transition in output_lower
                    for transition in ["for example", "such as", "additionally", "furthermore"]
                ):
                    contradictions.append(indicator)

        # Check for repetition
        words = output_lower.split()
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words)) if words else 0

        # Build metadata
        metadata = {
            "patterns_found": patterns_found,
            "repetition_ratio": repetition_ratio,
        }

        # Add issues if any
        if repetition_ratio > self.repetition_threshold:
            issues["repetitive"] = f"Repetition ratio: {repetition_ratio:.2f}"

        if contradictions:
            issues["contradictions"] = contradictions
            metadata["contradictions"] = contradictions

        if issues:
            metadata["issues"] = issues
            return RuleResult(
                passed=False,
                message="Consistency validation failed",
                metadata=metadata,
            )

        return RuleResult(
            passed=True,
            message="Consistency validation passed",
            metadata=metadata,
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
    assert not result.metadata["citations"]

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
    assert result.metadata["citations"]


def test_python_rule_validation(python_rule):
    """Test Python programming rule validation."""
    # Test valid Python code
    valid_code = '''
    """
    A simple example class with proper docstring.
    """
    import os
    from typing import List

    class MyClass:
        """A class that demonstrates proper Python style."""
        def my_function(self):
            """A method that returns a number."""
            my_var = 42
            return my_var
    '''
    result = python_rule.validate(valid_code)
    assert result.passed
    assert "patterns_checked" in result.metadata
    assert not result.metadata["style_issues"]
    assert not result.metadata["security_issues"]

    # Test code with style issues
    invalid_style = """
    Import os
    class myclass:
        def MyFunction():
            MY_VAR = 42
    """
    result = python_rule.validate(invalid_style)
    assert not result.passed
    assert result.metadata["style_issues"], "Expected style issues to be found"
    assert "missing_docstring" in result.metadata["style_issues"]

    # Test code with security issues
    security_issues = """
    def unsafe_function():
        eval("print('Hello')")
        exec("x = 1")
    """
    result = python_rule.validate(security_issues)
    assert not result.passed
    assert result.metadata["security_issues"], "Expected security issues to be found"
    assert any("eval" in issue or "exec" in issue for issue in result.metadata["security_issues"])


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
