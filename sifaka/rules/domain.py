"""
Domain-specific rules for Sifaka.
"""

from typing import Dict, Any, List, Optional, Set
from pydantic import Field
from sifaka.rules.base import Rule, RuleResult
import re
import ast


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

        Raises:
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

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
    Rule that validates legal content.

    Attributes:
        legal_terms: Dictionary of legal terms and their variations
        citation_patterns: List of regex patterns for legal citations
        disclaimers: List of required legal disclaimers
        disclaimer_required: Whether legal disclaimers are required
    """

    legal_terms: Dict[str, List[str]] = Field(
        default={
            "jurisdiction": ["jurisdiction", "court", "venue"],
            "statute": ["statute", "law", "regulation", "code"],
            "precedent": ["precedent", "case law", "ruling"],
            "liability": ["liability", "responsibility", "duty"],
        },
        description="Dictionary of legal terms and their variations",
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
        ],
        description="List of regex patterns for legal citations",
    )

    disclaimers: List[str] = Field(
        default=[
            "not legal advice",
            "consult.*attorney",
            "seek.*counsel",
            "legal disclaimer",
        ],
        description="List of required legal disclaimers",
    )

    disclaimer_required: bool = Field(
        default=True,
        description="Whether legal disclaimers are required",
    )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output contains appropriate legal content and disclaimers.
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        metadata = {"citations": [], "issues": [], "legal_terms_found": [], "has_disclaimer": False}

        # Check for legal terms
        for category, terms in self.legal_terms.items():
            for term in terms:
                if term.lower() in output_lower:
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


class PythonRule(Rule):
    """
    Rule that checks for Python code quality and best practices.

    Attributes:
        code_style_patterns: Dictionary of code style patterns to check
        security_patterns: Dictionary of security-related patterns to check
        performance_patterns: Dictionary of performance-related patterns to check
    """

    code_style_patterns: Dict[str, str] = Field(
        default={
            "pep8_imports": r"^import\s+[a-z]",
            "pep8_classes": r"class\s+[A-Z]",
            "pep8_functions": r"def\s+[a-z_]+",
            "pep8_variables": r"[a-z_][a-z0-9_]*\s*=",
            "docstring": r'"""[\s\S]*?"""',
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

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output is valid Python code and follows style guidelines.

        Args:
            output: The Python code to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results
        """
        metadata = {"style_issues": [], "security_issues": [], "issues": []}
        issues = []

        # Skip validation for test files unless explicitly requested
        is_test = kwargs.get("is_test", False)
        force_validate = kwargs.get("force_validate", False)
        if is_test and not force_validate:
            return RuleResult(
                passed=True, message="Validation skipped for test file", metadata=metadata
            )

        try:
            # Basic syntax check
            ast.parse(output)
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
            metadata["issues"] = issues
            return RuleResult(
                passed=False,
                message="Python validation failed: Invalid syntax",
                metadata=metadata,
            )

        # Style checks (skip for test files)
        if not is_test:
            # Check docstring - only require for non-empty code
            if output.strip() and not re.search(self.code_style_patterns["docstring"], output):
                metadata["style_issues"].append("missing_docstring")

        # Security checks (always perform)
        for pattern_name, pattern in self.security_patterns.items():
            if re.search(pattern, output):
                metadata["security_issues"].append(pattern_name)

        # Combine all issues
        issues.extend(metadata["style_issues"])
        issues.extend(metadata["security_issues"])
        metadata["issues"] = issues

        if issues:
            return RuleResult(
                passed=False,
                message="Code validation failed: " + "; ".join(issues),
                metadata=metadata,
            )

        return RuleResult(
            passed=True,
            message="Code validation passed",
            metadata=metadata,
        )


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
        Validate that the output maintains consistency.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

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
