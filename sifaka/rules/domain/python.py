"""
Python domain-specific validation rules for Sifaka.

This module provides validators and rules for checking Python code quality and best practices.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The PythonConfig class extends RuleConfig and provides type-safe access to parameters
    - Factory functions (create_python_rule, create_python_validator) handle configuration

Usage Example:
    from sifaka.rules.domain.python import create_python_rule

    # Create a Python rule using the factory function
    rule = create_python_rule(
        code_style_patterns={"docstring": r"'''.*?'''", "pep8_classes": r"class [A-Z]"},
        security_patterns={"eval": r"eval\(", "exec": r"exec\("}
    )

    # Validate Python code
    result = rule.validate("def hello_world(): print('Hello, world!')")

    # Alternative: Create with explicit RuleConfig
    from sifaka.rules.base import BaseValidator, RuleConfig, Any
    rule = PythonRule(
        config=RuleConfig(
            params={
                "code_style_patterns": {"docstring": r"'''.*?'''"},
                "security_patterns": {"eval": r"eval\("},
                "performance_patterns": {"list_comprehension": r"\[.*for.*in.*\]"}
            }
        )
    )
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from sifaka.rules.base import Rule, RuleConfig, RuleResult, RuleValidator
from sifaka.rules.domain.base import BaseDomainValidator

# Third-party
from pydantic import BaseModel, Field, PrivateAttr

__all__ = [
    # Config classes
    "PythonConfig",
    # Protocol classes
    "PythonValidator",
    # Validator classes
    "DefaultPythonValidator",
    # Rule classes
    "PythonRule",
    # Factory functions
    "create_python_validator",
    "create_python_rule",
    # Internal helpers
    "_PythonStyleAnalyzer",
    "_PythonSecurityAnalyzer",
    "_PythonPerformanceAnalyzer",
    "_PythonSyntaxAnalyzer",
]


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


class DefaultPythonValidator(BaseDomainValidator):
    """Default implementation of Python code validation composed of analyzers."""

    def __init__(self, config: PythonConfig) -> None:
        super().__init__(config)

        self._style_analyzer = _PythonStyleAnalyzer(patterns=config.code_style_patterns)
        self._security_analyzer = _PythonSecurityAnalyzer(patterns=config.security_patterns)
        self._perf_analyzer = _PythonPerformanceAnalyzer(patterns=config.performance_patterns)
        self._syntax_analyzer = _PythonSyntaxAnalyzer()

    @property
    def config(self) -> PythonConfig:  # type: ignore[override]
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:  # noqa: D401
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        style_matches = self._style_analyzer.analyze(text)
        security_issues = self._security_analyzer.analyze(text)
        perf_matches = self._perf_analyzer.analyze(text)
        syntax_ok, syntax_err = self._syntax_analyzer.analyze(text)

        passed = syntax_ok and not security_issues
        message = (
            "Python code validation passed"
            if passed
            else (
                f"Python syntax error: {syntax_err}" if not syntax_ok else "Security issues found"
            )
        )

        return RuleResult(
            passed=passed,
            message=message,
            metadata={
                "syntax_valid": syntax_ok,
                "error_message": syntax_err,
                "style_matches": style_matches,
                "security_issues": security_issues,
                "performance_matches": perf_matches,
            },
        )


class PythonRule(Rule):
    """Rule that checks Python code quality and best practices."""

    def __init__(
        self,
        name: str = "python_rule",
        description: str = "Checks Python code quality and best practices",
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[RuleConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Python rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration
            **kwargs: Additional keyword arguments for the rule
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultPythonValidator:
        """Create a default validator from config."""
        python_config = PythonConfig(**self._rule_params)
        return DefaultPythonValidator(python_config)


def create_python_validator(
    code_style_patterns: Optional[Dict[str, str]] = None,
    security_patterns: Optional[Dict[str, str]] = None,
    performance_patterns: Optional[Dict[str, str]] = None,
    **kwargs,
) -> DefaultPythonValidator:
    """
    Create a Python validator with the specified configuration.

    This factory function creates a configured Python validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        code_style_patterns: Dictionary of regex patterns for code style checks
        security_patterns: Dictionary of regex patterns for security checks
        performance_patterns: Dictionary of regex patterns for performance checks
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured Python validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if code_style_patterns is not None:
        config_params["code_style_patterns"] = code_style_patterns
    if security_patterns is not None:
        config_params["security_patterns"] = security_patterns
    if performance_patterns is not None:
        config_params["performance_patterns"] = performance_patterns

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = PythonConfig(**config_params)

    # Return configured validator
    return DefaultPythonValidator(config)


def create_python_rule(
    name: str = "python_rule",
    description: str = "Validates Python code",
    code_style_patterns: Optional[Dict[str, str]] = None,
    security_patterns: Optional[Dict[str, str]] = None,
    performance_patterns: Optional[Dict[str, str]] = None,
    **kwargs,
) -> PythonRule:
    """
    Create a Python code validation rule.

    This factory function creates a configured PythonRule instance.
    It uses create_python_validator internally to create the validator.

    Args:
        name: Name of the rule
        description: Description of the rule
        code_style_patterns: Dictionary of regex patterns for code style checks
        security_patterns: Dictionary of regex patterns for security checks
        performance_patterns: Dictionary of regex patterns for performance checks
        **kwargs: Additional keyword arguments for the rule

    Returns:
        A configured PythonRule
    """
    # Create validator using the validator factory
    validator = create_python_validator(
        code_style_patterns=code_style_patterns,
        security_patterns=security_patterns,
        performance_patterns=performance_patterns,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return PythonRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )


# ---------------------------------------------------------------------------
# Analyzer helpers (Single Responsibility)
# ---------------------------------------------------------------------------


class _PythonStyleAnalyzer(BaseModel):
    patterns: Dict[str, str] = Field(default_factory=dict)

    _compiled: Dict[str, re.Pattern[str]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = {k: re.compile(pat, re.MULTILINE) for k, pat in self.patterns.items()}

    def analyze(self, text: str) -> Dict[str, int]:
        return {name: len(p.findall(text)) for name, p in self._compiled.items()}


class _PythonSecurityAnalyzer(BaseModel):
    patterns: Dict[str, str] = Field(default_factory=dict)

    _compiled: Dict[str, re.Pattern[str]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = {k: re.compile(pat, re.MULTILINE) for k, pat in self.patterns.items()}

    def analyze(self, text: str) -> Dict[str, List[str]]:
        issues: Dict[str, List[str]] = {}
        for name, pat in self._compiled.items():
            matches = pat.findall(text)
            if matches:
                issues[name] = matches if isinstance(matches, list) else [matches]
        return issues


class _PythonPerformanceAnalyzer(BaseModel):
    patterns: Dict[str, str] = Field(default_factory=dict)

    _compiled: Dict[str, re.Pattern[str]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = {k: re.compile(pat) for k, pat in self.patterns.items()}

    def analyze(self, text: str) -> Dict[str, int]:
        return {name: len(p.findall(text)) for name, p in self._compiled.items()}


class _PythonSyntaxAnalyzer(BaseModel):
    """Attempt to compile code to detect syntax errors."""

    def analyze(self, text: str) -> tuple[bool, str]:
        try:
            compile(text, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, str(e)
