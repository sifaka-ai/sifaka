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
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


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
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


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
        config: MedicalConfig,
        validator: MedicalValidator,
    ) -> None:
        """Initialize the medical rule."""
        super().__init__(name=name, description=description)
        self._config = config
        self._validator = validator

    @property
    def config(self) -> MedicalConfig:
        """Get the rule configuration."""
        return self._config

    @property
    def validator(self) -> MedicalValidator:
        """Get the rule validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text contains accurate medical information and proper disclaimers.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If text is None or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        return self._validator.validate(text)


class LegalRule(Rule):
    """Rule that validates legal content."""

    def __init__(
        self,
        name: str,
        description: str,
        config: LegalConfig,
        validator: LegalValidator,
    ) -> None:
        """Initialize the legal rule."""
        super().__init__(name=name, description=description)
        self._config = config
        self._validator = validator

    @property
    def config(self) -> LegalConfig:
        """Get the rule configuration."""
        return self._config

    @property
    def validator(self) -> LegalValidator:
        """Get the rule validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text contains appropriate legal content and disclaimers.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If text is None or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        return self._validator.validate(text)


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
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


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
        config: PythonConfig,
        validator: PythonValidator,
    ) -> None:
        """Initialize the Python rule."""
        super().__init__(name=name, description=description)
        self._config = config
        self._validator = validator

    @property
    def config(self) -> PythonConfig:
        """Get the rule configuration."""
        return self._config

    @property
    def validator(self) -> PythonValidator:
        """Get the rule validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text follows Python code quality standards.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If text is None or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        return self._validator.validate(text)


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
        if not 0.0 <= self.repetition_threshold <= 1.0:
            raise ValueError("repetition_threshold must be between 0.0 and 1.0")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


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
        config: ConsistencyConfig,
        validator: ConsistencyValidator,
    ) -> None:
        """Initialize the consistency rule."""
        super().__init__(name=name, description=description)
        self._config = config
        self._validator = validator

    @property
    def config(self) -> ConsistencyConfig:
        """Get the rule configuration."""
        return self._config

    @property
    def validator(self) -> ConsistencyValidator:
        """Get the rule validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text is internally consistent.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If text is None or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        return self._validator.validate(text)
