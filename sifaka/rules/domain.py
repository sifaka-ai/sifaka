"""
Domain-specific rules for Sifaka.
"""

from typing import (
    Dict,
    Any,
    List,
    Optional,
    Set,
    Tuple,
    Protocol,
    runtime_checkable,
    Final,
    TypeVar,
)
from typing_extensions import TypeGuard
from dataclasses import dataclass, field
from pydantic import Field
from sifaka.rules.base import Rule, RuleResult, RuleConfig, RuleValidator
import re
import ast


@dataclass(frozen=True)
class MedicalConfig(RuleConfig):
    """Configuration for medical rules."""

    medical_terms: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "diagnosis": ["diagnosis", "diagnose", "diagnosed"],
            "treatment": ["treatment", "treat", "treating", "therapy"],
            "medication": ["medication", "drug", "prescription", "medicine"],
            "symptom": ["symptom", "symptoms", "sign", "signs"],
        }
    )
    warning_terms: Set[str] = field(
        default_factory=lambda: {
            "diagnosis",
            "treatment",
            "medication",
            "prescription",
            "therapy",
            "cure",
            "heal",
            "remedy",
        }
    )
    disclaimer_required: bool = True
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.medical_terms:
            raise ValueError("Must provide at least one medical term category")
        if not self.warning_terms:
            raise ValueError("Must provide at least one warning term")


@dataclass(frozen=True)
class LegalConfig(RuleConfig):
    """Configuration for legal rules."""

    legal_terms: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "jurisdiction": ["jurisdiction", "court", "venue", "forum", "tribunal"],
            "statute": ["statute", "law", "regulation", "code", "act", "bill", "ordinance"],
            "precedent": ["precedent", "case law", "ruling", "decision", "holding", "opinion"],
            "liability": ["liability", "responsibility", "duty", "obligation", "negligence"],
            "procedure": ["procedure", "motion", "pleading", "filing", "petition", "appeal"],
            "evidence": ["evidence", "proof", "exhibit", "testimony", "witness", "document"],
        }
    )
    citation_patterns: List[str] = field(
        default_factory=lambda: [
            r"\d+\s*(?:U\.?S\.?|F\.?(?:2d|3d)?|S\.?Ct\.?)\s*\d+",  # Federal cases
            r"\d+\s*[A-Z][a-z]*\.?\s*(?:2d|3d)?\s*\d+",  # State cases
            r"(?:\d+\s*)?U\.?S\.?C\.?\s*§*\s*\d+(?:\([a-z]\))?",  # U.S. Code
            r"\d+\s*(?:Cal\.?|N\.?Y\.?|Tex\.?)\s*(?:2d|3d|4th)?\s*\d+",  # State reporters
            r"(?:pub\.?\s*l\.?|P\.?L\.?)\s*\d+[-‐]\d+",  # Public Laws
            r"(?:CFR|C\.F\.R\.)\s*§*\s*\d+\.\d+",  # Code of Federal Regulations
            r"\d+\s*L\.?\s*Ed\.?\s*(?:2d)?\s*\d+",  # Supreme Court (Lawyers' Edition)
        ]
    )
    disclaimers: List[str] = field(
        default_factory=lambda: [
            r"(?i)not\s+(?:intended\s+as\s+)?legal\s+advice",
            r"(?i)consult\s+(?:(?:a|your)\s+)?(?:qualified\s+)?(?:attorney|lawyer|legal\s+counsel)",
            r"(?i)seek\s+legal\s+(?:counsel|advice|representation)",
            r"(?i)legal\s+disclaimer\s*[:\-]?",
            r"(?i)for\s+informational\s+purposes\s+only",
            r"(?i)does\s+not\s+constitute\s+(?:a|an)\s+attorney-client\s+relationship",
            r"(?i)not\s+a\s+substitute\s+for\s+legal\s+(?:counsel|advice)",
        ]
    )
    disclaimer_required: bool = True
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.legal_terms:
            raise ValueError("Must provide at least one legal term category")
        if not self.citation_patterns:
            raise ValueError("Must provide at least one citation pattern")
        if not self.disclaimers:
            raise ValueError("Must provide at least one disclaimer pattern")


@runtime_checkable
class MedicalValidator(Protocol):
    """Protocol for medical content validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> MedicalConfig: ...


@runtime_checkable
class LegalValidator(Protocol):
    """Protocol for legal content validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> LegalConfig: ...


class MedicalRule(Rule):
    """Rule that checks for medical content accuracy and safety."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the medical rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        medical_config = MedicalConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultMedicalValidator(medical_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output medical content."""
        return self._validator.validate(output)


class LegalRule(Rule):
    """Rule that validates legal content."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the legal rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        legal_config = LegalConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultLegalValidator(legal_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output legal content."""
        return self._validator.validate(output)


def create_medical_rule(
    name: str = "medical_rule",
    description: str = "Validates text for medical content",
    config: Optional[Dict[str, Any]] = None,
) -> MedicalRule:
    """
    Create a medical rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured MedicalRule instance
    """
    if config is None:
        config = {
            "medical_terms": {
                "diagnosis": ["diagnosis", "diagnose", "diagnosed"],
                "treatment": ["treatment", "treat", "treating", "therapy"],
                "medication": ["medication", "drug", "prescription", "medicine"],
                "symptom": ["symptom", "symptoms", "sign", "signs"],
            },
            "warning_terms": {
                "diagnosis",
                "treatment",
                "medication",
                "prescription",
                "therapy",
                "cure",
                "heal",
                "remedy",
            },
            "disclaimer_required": True,
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return MedicalRule(
        name=name,
        description=description,
        config=config,
    )


def create_legal_rule(
    name: str = "legal_rule",
    description: str = "Validates text for legal content",
    config: Optional[Dict[str, Any]] = None,
) -> LegalRule:
    """
    Create a legal rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured LegalRule instance
    """
    if config is None:
        config = {
            "legal_terms": {
                "jurisdiction": ["jurisdiction", "court", "venue", "forum", "tribunal"],
                "statute": ["statute", "law", "regulation", "code", "act", "bill", "ordinance"],
                "precedent": ["precedent", "case law", "ruling", "decision", "holding", "opinion"],
                "liability": ["liability", "responsibility", "duty", "obligation", "negligence"],
                "procedure": ["procedure", "motion", "pleading", "filing", "petition", "appeal"],
                "evidence": ["evidence", "proof", "exhibit", "testimony", "witness", "document"],
            },
            "citation_patterns": [
                r"\d+\s*(?:U\.?S\.?|F\.?(?:2d|3d)?|S\.?Ct\.?)\s*\d+",  # Federal cases
                r"\d+\s*[A-Z][a-z]*\.?\s*(?:2d|3d)?\s*\d+",  # State cases
                r"(?:\d+\s*)?U\.?S\.?C\.?\s*§*\s*\d+(?:\([a-z]\))?",  # U.S. Code
                r"\d+\s*(?:Cal\.?|N\.?Y\.?|Tex\.?)\s*(?:2d|3d|4th)?\s*\d+",  # State reporters
                r"(?:pub\.?\s*l\.?|P\.?L\.?)\s*\d+[-‐]\d+",  # Public Laws
                r"(?:CFR|C\.F\.R\.)\s*§*\s*\d+\.\d+",  # Code of Federal Regulations
                r"\d+\s*L\.?\s*Ed\.?\s*(?:2d)?\s*\d+",  # Supreme Court (Lawyers' Edition)
            ],
            "disclaimers": [
                r"(?i)not\s+(?:intended\s+as\s+)?legal\s+advice",
                r"(?i)consult\s+(?:(?:a|your)\s+)?(?:qualified\s+)?(?:attorney|lawyer|legal\s+counsel)",
                r"(?i)seek\s+legal\s+(?:counsel|advice|representation)",
                r"(?i)legal\s+disclaimer\s*[:\-]?",
                r"(?i)for\s+informational\s+purposes\s+only",
                r"(?i)does\s+not\s+constitute\s+(?:a|an)\s+attorney-client\s+relationship",
                r"(?i)not\s+a\s+substitute\s+for\s+legal\s+(?:counsel|advice)",
            ],
            "disclaimer_required": True,
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return LegalRule(
        name=name,
        description=description,
        config=config,
    )


@dataclass(frozen=True)
class PythonConfig(RuleConfig):
    """Configuration for Python code rules."""

    code_style_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "docstring": r'"""[\s\S]*?"""',
            "pep8_imports": r"^(?:from\s+[a-zA-Z0-9_.]+\s+)?import\s+(?:[a-zA-Z0-9_]+(?:\s*,\s*[a-zA-Z0-9_]+)*|\s*\([^)]+\))",
            "pep8_classes": r"^class\s+[A-Z][a-zA-Z0-9]*(?:\([^)]+\))?:",
            "pep8_functions": r"^(?:async\s+)?def\s+[a-z_][a-z0-9_]*\s*\([^)]*\)\s*(?:->\s*[^:]+)?:",
            "pep8_variables": r"^[a-z_][a-z0-9_]*\s*(?::\s*[^=\n]+)?\s*=",
            "type_hints": r":\s*(?:[a-zA-Z_][a-zA-Z0-9_]*(?:\[.*?\])?|None|Any|Optional\[.*?\])",
        }
    )
    security_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "eval": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\beval\s*\(",
            "exec": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\bexec\s*\(",
            "pickle": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\b(?:pickle|marshal|shelve)\.(?:load|loads)\b",
            "shell": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\b(?:os\.system|subprocess\.(?:call|Popen|run))\s*\(.*(?:shell\s*=\s*True|`|\$|;)",
            "sql": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\b(?:execute|executemany)\s*\(.*(?:%s|\+)",
            "unsafe_file": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\b(?:open|file)\s*\([^)]*(?:mode\s*=\s*['\"]w|\s*,\s*['\"]w)",
            "unsafe_network": r"(?<!#.*)(?<!'''.*)(?<!\"\"\".*)\b(?:urllib\.request\.urlopen|requests\.(?:get|post|put|delete))\s*\([^)]*verify\s*=\s*False",
        }
    )
    performance_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "list_comprehension": r"\[.*for.*in.*\]",
            "generator_expression": r"\(.*for.*in.*\)",
            "dict_comprehension": r"\{.*:.*for.*in.*\}",
            "set_comprehension": r"\{.*for.*in.*\}",
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.code_style_patterns:
            raise ValueError("Must provide at least one code style pattern")
        if not self.security_patterns:
            raise ValueError("Must provide at least one security pattern")
        if not self.performance_patterns:
            raise ValueError("Must provide at least one performance pattern")


@runtime_checkable
class PythonValidator(Protocol):
    """Protocol for Python code validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> PythonConfig: ...


class PythonRule(Rule):
    """Rule that checks Python code quality and best practices."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the Python rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        python_config = PythonConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultPythonValidator(python_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output Python code."""
        return self._validator.validate(output)


@dataclass(frozen=True)
class ConsistencyConfig(RuleConfig):
    """Configuration for consistency rules."""

    consistency_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "present": r"\b(?:is|are|am|has|have|do|does)\b",
            "past": r"\b(?:was|were|had|did)\b",
            "future": r"\b(?:will|shall|going to)\b",
            "first_person": r"\b(?:I|we|my|our|myself|ourselves)\b",
            "second_person": r"\b(?:you|your|yourself|yourselves)\b",
            "third_person": r"\b(?:he|she|it|they|his|her|its|their|himself|herself|itself|themselves)\b",
            "active": r"\b(?:subject)\s+(?:verb)\b",
            "passive": r"\b(?:is|are|was|were)\s+(?:\w+ed|\w+en)\b",
            "list_marker": r"(?m)^[-*•]\s+|\d+\.\s+",
            "code_block": r"```[\s\S]*?```|`[^`]+`",
            "table_marker": r"\|[^|]+\|",
            "heading": r"(?m)^#{1,6}\s+\w+",
        }
    )
    contradiction_indicators: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            (r"\b(?:is|are)\b", r"\b(?:is not|are not|isn't|aren't)\b"),
            (r"\b(?:will|shall)\b", r"\b(?:will not|shall not|won't|shan't)\b"),
            (r"\b(?:must|should)\b", r"\b(?:must not|should not|shouldn't)\b"),
            (r"\b(?:always|never)\b", r"\b(?:sometimes|occasionally)\b"),
            (r"\b(?:all|every)\b", r"\b(?:some|few|none)\b"),
            (r"\b(?:increase|rise)\b", r"\b(?:decrease|fall)\b"),
            (r"\b(?:more|greater)\b", r"\b(?:less|fewer)\b"),
            (r"\b(?:begin|start)\b", r"\b(?:end|finish)\b"),
        ]
    )
    repetition_threshold: float = 0.3
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not 0.0 <= self.repetition_threshold <= 1.0:
            raise ValueError("repetition_threshold must be between 0.0 and 1.0")
        if not self.consistency_patterns:
            raise ValueError("Must provide at least one consistency pattern")
        if not self.contradiction_indicators:
            raise ValueError("Must provide at least one contradiction indicator")


@runtime_checkable
class ConsistencyValidator(Protocol):
    """Protocol for consistency validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> ConsistencyConfig: ...


class ConsistencyRule(Rule):
    """Rule that checks for consistency in text."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the consistency rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        consistency_config = ConsistencyConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultConsistencyValidator(consistency_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output consistency."""
        return self._validator.validate(output)


class DefaultMedicalValidator(RuleValidator[str]):
    """Default implementation of medical content validation."""

    def __init__(self, config: MedicalConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> MedicalConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text for medical content."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        found_terms: Dict[str, List[str]] = {}
        warning_terms: List[str] = []

        # Check for medical terms
        for category, terms in self.config.medical_terms.items():
            matches = [term for term in terms if term in text_lower]
            if matches:
                found_terms[category] = matches

        # Check for warning terms
        warning_terms = [term for term in self.config.warning_terms if term in text_lower]

        # Check for disclaimer if required
        has_disclaimer = False
        if self.config.disclaimer_required:
            disclaimer_patterns = [
                r"(?i)not\s+medical\s+advice",
                r"(?i)consult\s+(?:a|your)\s+(?:doctor|physician|healthcare\s+provider)",
                r"(?i)seek\s+medical\s+(?:attention|advice|care)",
                r"(?i)for\s+informational\s+purposes\s+only",
            ]
            has_disclaimer = any(re.search(pattern, text) for pattern in disclaimer_patterns)

        if found_terms:
            if self.config.disclaimer_required and not has_disclaimer:
                return RuleResult(
                    passed=False,
                    message="Medical content requires a disclaimer",
                    metadata={
                        "found_terms": found_terms,
                        "warning_terms": warning_terms,
                        "has_disclaimer": has_disclaimer,
                    },
                )

            if warning_terms:
                return RuleResult(
                    passed=False,
                    message="Contains potentially unsafe medical terms",
                    metadata={
                        "found_terms": found_terms,
                        "warning_terms": warning_terms,
                        "has_disclaimer": has_disclaimer,
                    },
                )

        return RuleResult(
            passed=True,
            message="No unsafe medical content detected",
            metadata={
                "found_terms": found_terms,
                "warning_terms": warning_terms,
                "has_disclaimer": has_disclaimer,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


class DefaultLegalValidator(RuleValidator[str]):
    """Default implementation of legal content validation."""

    def __init__(self, config: LegalConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> LegalConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text for legal content."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        found_terms: Dict[str, List[str]] = {}
        found_citations: List[str] = []
        has_disclaimer = False

        # Check for legal terms
        for category, terms in self.config.legal_terms.items():
            matches = [term for term in terms if term in text_lower]
            if matches:
                found_terms[category] = matches

        # Check for legal citations
        for pattern in self.config.citation_patterns:
            matches = re.finditer(pattern, text)
            found_citations.extend(match.group() for match in matches)

        # Check for disclaimer if required
        if self.config.disclaimer_required:
            has_disclaimer = any(re.search(pattern, text) for pattern in self.config.disclaimers)

        if found_terms or found_citations:
            if self.config.disclaimer_required and not has_disclaimer:
                return RuleResult(
                    passed=False,
                    message="Legal content requires a disclaimer",
                    metadata={
                        "found_terms": found_terms,
                        "found_citations": found_citations,
                        "has_disclaimer": has_disclaimer,
                    },
                )

        return RuleResult(
            passed=True,
            message="No unsafe legal content detected",
            metadata={
                "found_terms": found_terms,
                "found_citations": found_citations,
                "has_disclaimer": has_disclaimer,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


class DefaultPythonValidator(RuleValidator[str]):
    """Default implementation of Python code validation."""

    def __init__(self, config: PythonConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> PythonConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate Python code for style, security, and performance."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        style_issues: Dict[str, List[str]] = {}
        security_issues: Dict[str, List[str]] = {}
        performance_issues: Dict[str, List[str]] = {}

        # Check code style patterns
        for name, pattern in self.config.code_style_patterns.items():
            matches = re.finditer(pattern, text, re.MULTILINE)
            violations = [match.group() for match in matches]
            if violations:
                style_issues[name] = violations

        # Check security patterns
        for name, pattern in self.config.security_patterns.items():
            matches = re.finditer(pattern, text, re.MULTILINE)
            violations = [match.group() for match in matches]
            if violations:
                security_issues[name] = violations

        # Check performance patterns
        for name, pattern in self.config.performance_patterns.items():
            matches = re.finditer(pattern, text, re.MULTILINE)
            violations = [match.group() for match in matches]
            if violations:
                performance_issues[name] = violations

        # Check if code is valid Python syntax
        try:
            ast.parse(text)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        if not syntax_valid:
            return RuleResult(
                passed=False,
                message="Invalid Python syntax",
                metadata={
                    "style_issues": style_issues,
                    "security_issues": security_issues,
                    "performance_issues": performance_issues,
                    "syntax_valid": syntax_valid,
                },
            )

        if security_issues:
            return RuleResult(
                passed=False,
                message="Code contains security issues",
                metadata={
                    "style_issues": style_issues,
                    "security_issues": security_issues,
                    "performance_issues": performance_issues,
                    "syntax_valid": syntax_valid,
                },
            )

        if style_issues:
            return RuleResult(
                passed=False,
                message="Code style issues detected",
                metadata={
                    "style_issues": style_issues,
                    "security_issues": security_issues,
                    "performance_issues": performance_issues,
                    "syntax_valid": syntax_valid,
                },
            )

        return RuleResult(
            passed=True,
            message="Code passes all checks",
            metadata={
                "style_issues": style_issues,
                "security_issues": security_issues,
                "performance_issues": performance_issues,
                "syntax_valid": syntax_valid,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


# Export public classes and functions
__all__ = [
    "MedicalRule",
    "MedicalConfig",
    "DefaultMedicalValidator",
    "LegalRule",
    "LegalConfig",
    "DefaultLegalValidator",
    "create_medical_rule",
    "create_legal_rule",
]
