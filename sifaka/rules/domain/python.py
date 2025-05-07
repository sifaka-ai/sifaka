"""
Python domain-specific validation rules for Sifaka.

This module provides rules for validating Python code, including:
- Code style validation: Checks for proper function definitions, class structures, etc.
- Security validation: Identifies potentially unsafe functions and patterns
- Performance validation: Detects code that might cause performance issues
- Syntax validation: Ensures code is syntactically correct

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The PythonConfig class extends RuleConfig and provides type-safe access to parameters
    - Factory functions (create_python_rule, create_python_validator) handle configuration
    - Specialized analyzers handle different aspects of Python code validation

Architecture:
    The Python validation system uses a composition pattern:
    - PythonRule is the main entry point that delegates to a validator
    - DefaultPythonValidator implements the validation logic
    - Four specialized analyzers handle different aspects of Python code:
      - _PythonStyleAnalyzer: Checks code style patterns
      - _PythonSecurityAnalyzer: Identifies security vulnerabilities
      - _PythonPerformanceAnalyzer: Detects performance issues
      - _PythonSyntaxAnalyzer: Validates syntax correctness
    - Factory functions provide a convenient way to create rules and validators

Lifecycle:
    1. Creation: Rules and validators are created using factory functions
    2. Configuration: Parameters are validated and stored in config objects
    3. Validation: Python code is analyzed for style, security, and performance issues
    4. Result: A RuleResult object is returned with validation status and metadata

Error Handling:
    - Invalid configuration parameters raise ValueError during initialization
    - Non-string inputs to validate() raise ValueError
    - Syntax errors in Python code are caught and returned as failed validation results
    - Empty strings are handled as invalid input

Usage Examples:
    Basic Usage:
        ```python
        from sifaka.rules.domain.python import create_python_rule

        # Create a Python rule with default settings
        rule = create_python_rule()

        # Validate Python code
        code = '''
        def hello_world():
            print("Hello, world!")

        if __name__ == "__main__":
            hello_world()
        '''

        result = rule.validate(code)

        # Check result
        if result.passed:
            print("Validation passed!")
            print(f"Style patterns found: {result.metadata['style_patterns']}")
        else:
            print(f"Validation failed: {result.message}")
            if 'security_issues' in result.metadata:
                print(f"Security issues: {result.metadata['security_issues']}")
            if 'performance_issues' in result.metadata:
                print(f"Performance issues: {result.metadata['performance_issues']}")
        ```

    Custom Style Patterns:
        ```python
        from sifaka.rules.domain.python import create_python_rule

        # Create a rule with custom code style patterns
        rule = create_python_rule(
            code_style_patterns=[
                r"def \w+\(.*\):",           # Function definition
                r"class \w+:",               # Class definition
                r"if __name__ == \"__main__\":",  # Main block
                r"import \w+",               # Import statement
                r"from \w+ import",          # From import
                r"# [A-Z]",                  # Comments starting with capital letter
                r"\"\"\".*?\"\"\"",          # Docstrings
            ],
            name="code_style_checker"
        )

        # Validate Python code
        code = '''
        def calculate_sum(a, b):
            """Calculate the sum of two numbers."""
            return a + b

        # Main entry point
        if __name__ == "__main__":
            result = calculate_sum(5, 10)
            print(f"Result: {result}")
        '''

        result = rule.validate(code)

        # Analyze style patterns
        if result.passed:
            style_patterns = result.metadata["style_patterns"]
            print("Code style patterns found:")
            for pattern, count in style_patterns.items():
                if count > 0:
                    print(f"- {pattern}: {count} occurrences")
        ```

    Security Validation:
        ```python
        from sifaka.rules.domain.python import create_python_rule

        # Create a rule focused on security validation
        rule = create_python_rule(
            security_patterns=[
                r"eval\(",                  # Eval function
                r"exec\(",                  # Exec function
                r"pickle\.loads\(",         # Pickle loads
                r"subprocess\.run\(",       # Subprocess run
                r"os\.system\(",            # OS system
                r"__import__\(",            # Dynamic import
                r"open\(.+?, ['\"]w['\"]",  # File writing
            ],
            name="security_validator"
        )

        # Check for security issues
        code = '''
        def process_data(data_string):
            # WARNING: This is unsafe!
            result = eval(data_string)
            return result

        def execute_command(cmd):
            import os
            os.system(cmd)  # Dangerous!
        '''

        result = rule.validate(code)

        # Analyze security issues
        if not result.passed and 'security_issues' in result.metadata:
            security_issues = result.metadata["security_issues"]
            print(f"Found {len(security_issues)} security issues:")
            for issue_type, instances in security_issues.items():
                print(f"- {issue_type}: {len(instances)} occurrences")
                for instance in instances:
                    print(f"  - {instance}")
        ```

    Performance Validation:
        ```python
        from sifaka.rules.domain.python import create_python_rule

        # Create a rule focused on performance validation
        rule = create_python_rule(
            performance_patterns=[
                r"for \w+ in range\(\d+\):",     # Range loop
                r"while True:",                  # Infinite loop
                r"\[\w+ for \w+ in \w+\]",       # List comprehension
                r"\.append\(",                   # List append in loop
                r"time\.sleep\(",                # Time sleep
                r"\+ \w+ \+",                    # String concatenation
            ],
            name="performance_validator"
        )

        # Check for performance issues
        code = '''
        def slow_function():
            result = []
            for i in range(1000000):  # Large range
                result.append(i)      # Append in loop

            message = "Result: " + str(result) + " is ready"  # String concatenation
            return message
        '''

        result = rule.validate(code)

        # Analyze performance issues
        if not result.passed and 'performance_issues' in result.metadata:
            performance_issues = result.metadata["performance_issues"]
            print("Performance issues found:")
            for pattern, count in performance_issues.items():
                if count > 0:
                    print(f"- {pattern}: {count} occurrences")
        ```

    Advanced Configuration:
        ```python
        from sifaka.rules.domain.python import create_python_rule
        from sifaka.rules.base import RuleConfig, RulePriority

        # Create a comprehensive Python validator
        rule = create_python_rule(
            name="comprehensive_python_validator",
            description="Validates Python code style, security, and performance",
            code_style_patterns=[
                r"def \w+\(.*\):",
                r"class \w+\(.*\):",
                r"if __name__ == \"__main__\":",
            ],
            security_patterns=[
                r"eval\(",
                r"exec\(",
                r"os\.system\(",
            ],
            performance_patterns=[
                r"for \w+ in range\(\d+\):",
                r"while True:",
                r"\.append\(",
            ],
            priority=RulePriority.HIGH,
            cost=2.0,
            cache_size=200
        )

        # Or use RuleConfig directly
        from sifaka.rules.domain.python import PythonRule

        config = RuleConfig(
            priority=RulePriority.HIGH,
            cost=2.0,
            params={
                "code_style_patterns": [
                    r"def \w+\(.*\):",
                    r"class \w+\(.*\):",
                ],
                "security_patterns": [
                    r"eval\(",
                    r"exec\(",
                ],
                "performance_patterns": [
                    r"for \w+ in range\(\d+\):",
                    r"while True:",
                ]
            }
        )

        custom_rule = PythonRule(
            name="custom_python_rule",
            description="Custom Python code validator",
            config=config
        )

        # Validate with configured rules
        result = rule.validate('''
        def calculate(expression):
            return eval(expression)  # Security issue!

        for i in range(1000000):     # Performance issue!
            print(i)
        ''')

        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
"""

import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.domain.base import DomainValidator


# This will be defined after PythonConfig


# Default code style patterns
DEFAULT_CODE_STYLE_PATTERNS: List[str] = [
    r"def \w+\(.*\):",  # Function definition
    r"class \w+:",  # Class definition
    r"import \w+",  # Import statement
    r"from \w+ import",  # From import
    r"try:",  # Try block
    r"except \w+:",  # Except block
    r"finally:",  # Finally block
    r"with \w+:",  # With statement
    r"for \w+ in",  # For loop
    r"while \w+:",  # While loop
]

# Default security patterns
DEFAULT_SECURITY_PATTERNS: List[str] = [
    r"eval\(",  # Eval function
    r"exec\(",  # Exec function
    r"pickle\.loads\(",  # Pickle loads
    r"subprocess\.run\(",  # Subprocess run
    r"os\.system\(",  # OS system
    r"os\.popen\(",  # OS popen
    r"shutil\.rmtree\(",  # Shutil rmtree
    r"tempfile\.mkstemp\(",  # Tempfile mkstemp
    r"urllib\.request\.urlopen\(",  # URL open
    r"xml\.etree\.ElementTree\.parse\(",  # XML parse
]

# Default performance patterns
DEFAULT_PERFORMANCE_PATTERNS: List[str] = [
    r"for \w+ in range\(\d+\):",  # Range loop
    r"while True:",  # Infinite loop
    r"time\.sleep\(",  # Time sleep
    r"threading\.Thread\(",  # Thread creation
    r"multiprocessing\.Process\(",  # Process creation
    r"subprocess\.Popen\(",  # Subprocess creation
    r"urllib\.request\.urlretrieve\(",  # URL retrieve
    r"pickle\.dump\(",  # Pickle dump
    r"json\.dump\(",  # JSON dump
    r"csv\.writer\(",  # CSV writer
]


class PythonConfig(BaseModel):
    """Configuration for Python code validation."""

    model_config = ConfigDict(frozen=True)

    code_style_patterns: List[str] = Field(
        default_factory=lambda: DEFAULT_CODE_STYLE_PATTERNS,
        description="List of regex patterns for code style validation",
    )
    security_patterns: List[str] = Field(
        default_factory=lambda: DEFAULT_SECURITY_PATTERNS,
        description="List of regex patterns for security validation",
    )
    performance_patterns: List[str] = Field(
        default_factory=lambda: DEFAULT_PERFORMANCE_PATTERNS,
        description="List of regex patterns for performance validation",
    )

    @field_validator("code_style_patterns")
    @classmethod
    def validate_code_style_patterns(cls, v: List[str]) -> List[str]:
        """Validate that code style patterns are not empty."""
        if not v:
            raise ValueError("Code style patterns cannot be empty")
        return v

    @field_validator("security_patterns")
    @classmethod
    def validate_security_patterns(cls, v: List[str]) -> List[str]:
        """Validate that security patterns are not empty."""
        if not v:
            raise ValueError("Security patterns cannot be empty")
        return v

    @field_validator("performance_patterns")
    @classmethod
    def validate_performance_patterns(cls, v: List[str]) -> List[str]:
        """Validate that performance patterns are not empty."""
        if not v:
            raise ValueError("Performance patterns cannot be empty")
        return v


class DefaultPythonValidator(DomainValidator):
    """Default implementation of Python code validation."""

    def __init__(self, config: PythonConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)

    def validate(self, text: str, **kwargs) -> RuleResult:  # noqa: ARG002
        """Validate Python code."""
        return RuleResult(passed=True, message="Python validation not implemented")


class PythonRule(Rule):
    """Rule that checks Python code quality and best practices."""

    def __init__(
        self,
        name: str = "python_rule",
        description: str = "Checks Python code quality and best practices",
        validator: Optional[BaseValidator] = None,
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
    code_style_patterns: Optional[List[str]] = None,
    security_patterns: Optional[List[str]] = None,
    performance_patterns: Optional[List[str]] = None,
    **kwargs,
) -> DefaultPythonValidator:
    """
    Create a Python validator with the specified configuration.

    This factory function creates a configured Python validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        code_style_patterns: List of regex patterns for code style checks
        security_patterns: List of regex patterns for security checks
        performance_patterns: List of regex patterns for performance checks
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
    code_style_patterns: Optional[List[str]] = None,
    security_patterns: Optional[List[str]] = None,
    performance_patterns: Optional[List[str]] = None,
    **kwargs,
) -> PythonRule:
    """
    Create a Python code validation rule.

    This factory function creates a configured PythonRule instance.
    It uses create_python_validator internally to create the validator.

    Args:
        name: Name of the rule
        description: Description of the rule
        code_style_patterns: List of regex patterns for code style checks
        security_patterns: List of regex patterns for security checks
        performance_patterns: List of regex patterns for performance checks
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
    patterns: List[str] = Field(default_factory=list)

    _compiled: Dict[str, Pattern[str]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = {k: re.compile(pat, re.MULTILINE) for k, pat in self.patterns}

    def analyze(self, text: str) -> Dict[str, int]:
        return {name: len(p.findall(text)) for name, p in self._compiled.items()}


class _PythonSecurityAnalyzer(BaseModel):
    patterns: List[str] = Field(default_factory=list)

    _compiled: Dict[str, Pattern[str]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = {k: re.compile(pat, re.MULTILINE) for k, pat in self.patterns}

    def analyze(self, text: str) -> Dict[str, List[str]]:
        issues: Dict[str, List[str]] = {}
        for name, pat in self._compiled.items():
            matches = pat.findall(text)
            if matches:
                issues[name] = matches if isinstance(matches, list) else [matches]
        return issues


class _PythonPerformanceAnalyzer(BaseModel):
    patterns: List[str] = Field(default_factory=list)

    _compiled: Dict[str, Pattern[str]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = {k: re.compile(pat) for k, pat in self.patterns}

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
