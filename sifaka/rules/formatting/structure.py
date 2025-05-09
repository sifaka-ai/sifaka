"""
Structure validation rules for Sifaka.

This module provides rules for validating text structure, including
section organization, heading hierarchy, and document structure.

Usage Example:
    from sifaka.rules.formatting.structure import create_structure_rule

    # Create a structure rule
    structure_rule = create_structure_rule(
        required_sections=["introduction", "body", "conclusion"],
        min_sections=3
    )

    # Validate text
    result = structure_rule.validate("This is a test.")
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult


class StructureConfig(BaseModel):
    """Configuration for structure validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    required_sections: List[str] = Field(
        default_factory=list,
        description="List of required sections",
    )
    min_sections: int = Field(
        default=1,
        ge=0,
        description="Minimum number of sections required",
    )
    max_sections: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of sections allowed",
    )
    cache_size: int = Field(
        default=100,
        ge=1,
        description="Size of the validation cache",
    )


class StructureValidator(BaseValidator):
    """Validator for text structure."""

    def __init__(self, config: Optional[StructureConfig] = None):
        """
        Initialize the validator.

        Args:
            config: Configuration for the validator
        """
        self._config = config or StructureConfig()

    @property
    def config(self) -> StructureConfig:
        """
        Get the validator configuration.

        Returns:
            The validator configuration
        """
        return self._config

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """
        Validate text structure.

        Args:
            text: Text to validate
            **kwargs: Additional validation options

        Returns:
            Validation result
        """
        # Handle empty text
        if not text.strip():
            return RuleResult(
                passed=False,
                rule_name="structure",
                message="Text is empty",
                metadata={"empty": True},
            )

        # Analyze sections
        sections = self._analyze_sections(text)
        section_count = len(sections)

        # Check section count
        if section_count < self.config.min_sections:
            return RuleResult(
                passed=False,
                rule_name="structure",
                message=f"Text has {section_count} sections, but at least {self.config.min_sections} are required",
                metadata={"section_count": section_count, "sections": sections},
            )

        if self.config.max_sections and section_count > self.config.max_sections:
            return RuleResult(
                passed=False,
                rule_name="structure",
                message=f"Text has {section_count} sections, but at most {self.config.max_sections} are allowed",
                metadata={"section_count": section_count, "sections": sections},
            )

        # Check required sections
        missing_sections = [s for s in self.config.required_sections if s not in sections]
        if missing_sections:
            return RuleResult(
                passed=False,
                rule_name="structure",
                message=f"Text is missing required sections: {', '.join(missing_sections)}",
                metadata={"missing_sections": missing_sections, "sections": sections},
            )

        # All checks passed
        return RuleResult(
            passed=True,
            rule_name="structure",
            message="Text structure is valid",
            metadata={"section_count": section_count, "sections": sections},
        )

    def _analyze_sections(self, text: str) -> List[str]:
        """
        Analyze text sections.

        Args:
            text: Text to analyze

        Returns:
            List of section names
        """
        # Simple section analysis based on headings
        sections = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                # Extract section name from heading
                section_name = line.lstrip("#").strip().lower()
                if section_name:
                    sections.append(section_name)
        return sections


class StructureRule(Rule):
    """Rule for validating text structure."""

    def __init__(
        self,
        name: str,
        description: str,
        config: RuleConfig,
        validator: StructureValidator,
    ):
        """
        Initialize the rule.

        Args:
            name: Name of the rule
            description: Description of the rule
            config: Configuration for the rule
            validator: Validator to use for validation
        """
        super().__init__(name=name, description=description, config=config, validator=validator)


def create_structure_validator(
    required_sections: Optional[List[str]] = None,
    min_sections: int = 1,
    max_sections: Optional[int] = None,
    **kwargs: Any,
) -> StructureValidator:
    """
    Create a structure validator.

    Args:
        required_sections: List of required sections
        min_sections: Minimum number of sections required
        max_sections: Maximum number of sections allowed
        **kwargs: Additional configuration parameters

    Returns:
        A new structure validator instance
    """
    config = StructureConfig(
        required_sections=required_sections or [],
        min_sections=min_sections,
        max_sections=max_sections,
        **kwargs,
    )
    return StructureValidator(config=config)


def create_structure_rule(
    name: str = "structure",
    description: str = "Validates text structure",
    required_sections: Optional[List[str]] = None,
    min_sections: int = 1,
    max_sections: Optional[int] = None,
    **kwargs: Any,
) -> StructureRule:
    """
    Create a structure rule.

    Args:
        name: Name of the rule
        description: Description of the rule
        required_sections: List of required sections
        min_sections: Minimum number of sections required
        max_sections: Maximum number of sections allowed
        **kwargs: Additional configuration parameters

    Returns:
        A new structure rule instance
    """
    validator = create_structure_validator(
        required_sections=required_sections,
        min_sections=min_sections,
        max_sections=max_sections,
    )
    config = RuleConfig(params=kwargs)
    return StructureRule(
        name=name,
        description=description,
        config=config,
        validator=validator,
    )


__all__ = [
    # Config classes
    "StructureConfig",
    # Validator classes
    "StructureValidator",
    # Rule classes
    "StructureRule",
    # Factory functions
    "create_structure_validator",
    "create_structure_rule",
]
