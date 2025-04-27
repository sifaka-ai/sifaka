"""
Wrapper rules for Sifaka.

This module provides rules for wrapping and formatting text output,
including prefix/suffix addition, text encapsulation, and template-based wrapping.
"""

from typing import Dict, List, Optional, Protocol, runtime_checkable, Final
from dataclasses import dataclass, field
from sifaka.rules.base import Rule, RuleResult


@dataclass(frozen=True)
class WrapperConfig:
    """Configuration for text wrapping."""

    prefix: str = ""
    suffix: str = ""
    template: Optional[str] = None
    strip_whitespace: bool = True
    preserve_newlines: bool = True
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if self.template and not ("{content}" in self.template):
            raise ValueError("Template must contain '{content}' placeholder")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")


@dataclass(frozen=True)
class CodeBlockConfig:
    """Configuration for code block wrapping."""

    language: str
    indent_size: int = 4
    add_syntax_markers: bool = True
    preserve_indentation: bool = True
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if not self.language:
            raise ValueError("Language must be specified")
        if self.indent_size < 0:
            raise ValueError("Indent size must be non-negative")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")


@runtime_checkable
class WrapperValidator(Protocol):
    """Protocol for text wrapping validation."""

    @property
    def config(self) -> WrapperConfig:
        """Get validator configuration."""
        ...

    def validate(self, text: str) -> RuleResult:
        """Validate wrapped text."""
        ...


@runtime_checkable
class CodeBlockValidator(Protocol):
    """Protocol for code block validation."""

    @property
    def config(self) -> CodeBlockConfig:
        """Get validator configuration."""
        ...

    def validate(self, text: str) -> RuleResult:
        """Validate code block."""
        ...


class DefaultWrapperValidator:
    """Default implementation of text wrapping validation."""

    def __init__(self, config: WrapperConfig):
        self._config = config

    @property
    def config(self) -> WrapperConfig:
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text wrapping."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Process text according to configuration
        processed_text = text
        if self.config.strip_whitespace:
            processed_text = processed_text.strip()

        # Apply template if provided, otherwise use prefix/suffix
        if self.config.template:
            wrapped_text = self.config.template.format(content=processed_text)
        else:
            wrapped_text = f"{self.config.prefix}{processed_text}{self.config.suffix}"

        # Handle newlines
        if self.config.preserve_newlines:
            wrapped_text = wrapped_text.replace("\n", "\n" if not self.config.template else "\n")

        # Check if wrapping was applied correctly
        is_wrapped = (
            (not self.config.prefix or wrapped_text.startswith(self.config.prefix))
            and (not self.config.suffix or wrapped_text.endswith(self.config.suffix))
            and (not self.config.template or text in wrapped_text)
        )

        return RuleResult(
            passed=is_wrapped,
            message="Text wrapped successfully" if is_wrapped else "Failed to wrap text correctly",
            metadata={
                "original_length": len(text),
                "wrapped_length": len(wrapped_text),
                "has_prefix": bool(self.config.prefix),
                "has_suffix": bool(self.config.suffix),
                "used_template": bool(self.config.template),
                "wrapped_text": wrapped_text,
            },
        )


class DefaultCodeBlockValidator:
    """Default implementation of code block validation."""

    def __init__(self, config: CodeBlockConfig):
        self._config = config

    @property
    def config(self) -> CodeBlockConfig:
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate code block."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        lines = text.split("\n")
        processed_lines = []
        base_indent = " " * self.config.indent_size

        # Process each line
        for line in lines:
            if self.config.preserve_indentation:
                # Count leading spaces for indentation
                leading_spaces = len(line) - len(line.lstrip())
                indent_level = leading_spaces // self.config.indent_size
                processed_line = base_indent * indent_level + line.lstrip()
            else:
                processed_line = line.strip()
            processed_lines.append(processed_line)

        # Join lines and add syntax markers if needed
        processed_text = "\n".join(processed_lines)
        if self.config.add_syntax_markers:
            processed_text = f"```{self.config.language}\n{processed_text}\n```"

        # Validate the processed text
        is_valid = processed_text.count("```") in (0, 2) and (
            not self.config.add_syntax_markers
            or processed_text.startswith(f"```{self.config.language}")
        )

        return RuleResult(
            passed=is_valid,
            message=(
                "Code block formatted successfully"
                if is_valid
                else "Failed to format code block correctly"
            ),
            metadata={
                "language": self.config.language,
                "line_count": len(lines),
                "has_syntax_markers": self.config.add_syntax_markers,
                "formatted_text": processed_text,
            },
        )


class WrapperRule(Rule):
    """Rule for validating text wrapping."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: WrapperValidator,
    ) -> None:
        super().__init__(name=name, description=description)
        self._validator = validator

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output wrapping."""
        return self._validator.validate(output)


class CodeBlockRule(Rule):
    """Rule for validating code blocks."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: CodeBlockValidator,
    ) -> None:
        super().__init__(name=name, description=description)
        self._validator = validator

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output code block."""
        return self._validator.validate(output)


# Default templates for common use cases
DEFAULT_TEMPLATES: Final[Dict[str, str]] = {
    "quote": "> {content}",
    "bold": "**{content}**",
    "italic": "*{content}*",
    "code": "`{content}`",
    "heading1": "# {content}",
    "heading2": "## {content}",
    "heading3": "### {content}",
}

# Default language configurations
DEFAULT_LANGUAGE_CONFIGS: Final[Dict[str, Dict[str, Any]]] = {
    "python": {
        "indent_size": 4,
        "add_syntax_markers": True,
    },
    "javascript": {
        "indent_size": 2,
        "add_syntax_markers": True,
    },
    "markdown": {
        "indent_size": 2,
        "add_syntax_markers": False,
    },
}


def create_wrapper_rule(
    name: str,
    description: str,
    prefix: str = "",
    suffix: str = "",
    template: Optional[str] = None,
    strip_whitespace: bool = True,
    preserve_newlines: bool = True,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
) -> WrapperRule:
    """Create a wrapper rule with default configuration."""
    config = WrapperConfig(
        prefix=prefix,
        suffix=suffix,
        template=template,
        strip_whitespace=strip_whitespace,
        preserve_newlines=preserve_newlines,
        cache_size=cache_size,
        priority=priority,
        cost=cost,
    )
    validator = DefaultWrapperValidator(config)
    return WrapperRule(name=name, description=description, validator=validator)


def create_code_block_rule(
    name: str,
    description: str,
    language: str,
    indent_size: Optional[int] = None,
    add_syntax_markers: Optional[bool] = None,
    preserve_indentation: bool = True,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
) -> CodeBlockRule:
    """Create a code block rule with default configuration."""
    # Use language-specific defaults if available
    lang_config = DEFAULT_LANGUAGE_CONFIGS.get(language, {})
    config = CodeBlockConfig(
        language=language,
        indent_size=indent_size or lang_config.get("indent_size", 4),
        add_syntax_markers=(
            add_syntax_markers
            if add_syntax_markers is not None
            else lang_config.get("add_syntax_markers", True)
        ),
        preserve_indentation=preserve_indentation,
        cache_size=cache_size,
        priority=priority,
        cost=cost,
    )
    validator = DefaultCodeBlockValidator(config)
    return CodeBlockRule(name=name, description=description, validator=validator)
