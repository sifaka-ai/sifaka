"""
Domain-specific rules for Sifaka.
"""

from typing import Dict, Any, List, Optional, Set
from pydantic import Field
from sifaka.rules.base import Rule, RuleResult
import re


class MedicalRule(Rule):
    """
    Rule that checks for medical content accuracy and safety.

    Attributes:
        medical_terms (Dict[str, List[str]]): Dictionary of medical terms and their correct variations
        warning_terms (List[str]): List of terms that require medical disclaimers
        disclaimer_required (bool): Whether medical disclaimers are required
    """

    medical_terms: Dict[str, List[str]] = Field(
        default={
            "diagnosis": ["diagnosis", "diagnose", "diagnosed"],
            "treatment": ["treatment", "treat", "treating", "therapy"],
            "medication": ["medication", "drug", "prescription", "medicine"],
            "symptom": ["symptom", "symptoms", "sign", "signs"],
        }
    )
    warning_terms: List[str] = Field(
        default=[
            "diagnosis",
            "treatment",
            "medication",
            "prescription",
            "therapy",
            "cure",
            "heal",
            "remedy",
        ]
    )
    disclaimer_required: bool = Field(default=True)

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output contains accurate medical information and proper disclaimers.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation
        """
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


class LegalRule(Rule):
    """
    Rule that checks for legal content accuracy and compliance.

    Attributes:
        legal_terms (Dict[str, List[str]]): Dictionary of legal terms and their correct variations
        citation_patterns (List[str]): List of regex patterns for legal citations
        disclaimer_required (bool): Whether legal disclaimers are required
    """

    legal_terms: Dict[str, List[str]] = Field(
        default={
            "jurisdiction": ["jurisdiction", "court", "venue"],
            "statute": ["statute", "law", "regulation", "code"],
            "precedent": ["precedent", "case law", "ruling"],
            "liability": ["liability", "responsibility", "duty"],
        }
    )
    citation_patterns: List[str] = Field(
        default=[
            r"\d+ [A-Z]+\. \d+",  # Volume Reporter Page
            r"\d+ [A-Z]+\.\d+",  # Volume Reporter.Page
            r"[A-Z]+\. \d+",  # Reporter Page
            r"\d+ U\.S\. \d+",  # Supreme Court
            r"\d+ S\.Ct\. \d+",  # Supreme Court Reporter
            r"\d+ F\.\d+",  # Federal Reporter
            r"\d+ F\.Supp\.\d+",  # Federal Supplement
        ]
    )
    disclaimer_required: bool = Field(default=True)

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output contains accurate legal information and proper citations.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation
        """
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


class PythonProgrammingRule(Rule):
    """
    Rule that checks for Python code quality and best practices.

    Attributes:
        code_style_patterns (Dict[str, str]): Dictionary of code style patterns to check
        security_patterns (Dict[str, str]): Dictionary of security-related patterns to check
        performance_patterns (Dict[str, str]): Dictionary of performance-related patterns to check
    """

    code_style_patterns: Dict[str, str] = Field(
        default={
            "pep8_imports": r"^import\s+[a-z]",
            "pep8_classes": r"class\s+[A-Z]",
            "pep8_functions": r"def\s+[a-z_]+",
            "pep8_variables": r"[a-z_][a-z0-9_]*\s*=",
            "docstring": r'""".*?"""',
        }
    )
    security_patterns: Dict[str, str] = Field(
        default={
            "eval": r"eval\(",
            "exec": r"exec\(",
            "pickle": r"pickle\.load",
            "shell": r"subprocess\.run\(.*shell=True",
            "password": r"password\s*=",
        }
    )
    performance_patterns: Dict[str, str] = Field(
        default={
            "global": r"global\s+",
            "nested_loop": r"for\s+.*:\s*for\s+.*:",
            "list_comprehension": r"\[.*for.*in.*\]",
            "generator": r"\(.*for.*in.*\)",
        }
    )

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output follows Python best practices.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation
        """
        issues = {}

        # Check code style
        for pattern_name, pattern in self.code_style_patterns.items():
            matches = re.findall(pattern, output, re.MULTILINE)
            if matches:
                issues[f"code_style_{pattern_name}"] = matches

        # Check security
        for pattern_name, pattern in self.security_patterns.items():
            matches = re.findall(pattern, output)
            if matches:
                issues[f"security_{pattern_name}"] = matches

        # Check performance
        for pattern_name, pattern in self.performance_patterns.items():
            matches = re.findall(pattern, output)
            if matches:
                issues[f"performance_{pattern_name}"] = matches

        if issues:
            return RuleResult(
                passed=False, message="Python code validation failed", metadata={"issues": issues}
            )

        return RuleResult(passed=True, message="Python code validation passed")


class ConsistencyRule(Rule):
    """
    Rule that checks for consistency in the output.

    Attributes:
        consistency_patterns (Dict[str, str]): Dictionary of patterns to check for consistency
        contradiction_indicators (List[str]): List of terms that indicate contradictions
        repetition_threshold (float): Threshold for considering text repetitive (0.0 to 1.0)
    """

    consistency_patterns: Dict[str, str] = Field(
        default={
            "tense": r"(is|was|will be|has been|had been)",
            "person": r"(I|we|you|he|she|they|it)",
            "voice": r"(active|passive)",
            "format": r"(list|paragraph|table|code)",
        }
    )
    contradiction_indicators: List[str] = Field(
        default=[
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
    )
    repetition_threshold: float = Field(default=0.3)

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output maintains consistency in various aspects.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation
        """
        issues = {}
        output_lower = output.lower()

        # Check for contradictions
        contradictions = []
        for indicator in self.contradiction_indicators:
            if indicator in output_lower:
                contradictions.append(indicator)
        if contradictions:
            issues["contradictions"] = contradictions

        # Check for repetition
        words = output_lower.split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        repeated_words = {
            word: count
            for word, count in word_counts.items()
            if count / len(words) > self.repetition_threshold
        }
        if repeated_words:
            issues["repetition"] = repeated_words

        # Check consistency patterns
        for pattern_name, pattern in self.consistency_patterns.items():
            matches = re.findall(pattern, output_lower)
            if len(set(matches)) > 1:  # More than one variation found
                issues[f"inconsistent_{pattern_name}"] = matches

        if issues:
            return RuleResult(
                passed=False, message="Consistency validation failed", metadata={"issues": issues}
            )

        return RuleResult(passed=True, message="Consistency validation passed")
