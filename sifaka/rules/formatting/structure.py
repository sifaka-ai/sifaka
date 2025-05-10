"""
Structure validation rules for Sifaka.

This module provides rules for validating text structure, including
section organization, heading hierarchy, and document structure.

Usage Example:
    ```python
    from sifaka.rules.formatting.structure import create_structure_rule

    # Create a structure rule
    structure_rule = create_structure_rule(
        required_sections=["introduction", "body", "conclusion"],
        min_sections=3
    )

    # Validate text
    result = structure_rule.validate("This is a test.")
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```
"""

from typing import List, Optional, Any
import time

from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


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


class StructureValidator(BaseValidator[str]):
    """
    Validator for text structure.

    This validator checks if text meets structure requirements including
    section organization, heading hierarchy, and document structure.

    Lifecycle:
        1. Initialization: Set up with structure constraints
        2. Validation: Check text against structure requirements
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.structure import create_structure_validator

        # Create a validator
        validator = create_structure_validator(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3
        )

        # Validate text
        result = validator.validate("# Introduction\\nContent\\n# Body\\nMore content\\n# Conclusion\\nFinal content")
        ```
    """

    def __init__(self, config: Optional[StructureConfig] = None):
        """
        Initialize the validator.

        Args:
            config: Configuration for the validator
        """
        super().__init__(validation_type=str)
        self._config = config or StructureConfig()

    @property
    def config(self) -> StructureConfig:
        """
        Get the validator configuration.

        Returns:
            The validator configuration
        """
        return self._config

    def validate(self, text: str) -> RuleResult:
        """
        Validate text structure.

        Args:
            text: Text to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        # Analyze sections
        sections = self._analyze_sections(text)
        section_count = len(sections)

        issues = []
        suggestions = []

        # Check section count
        if section_count < self.config.min_sections:
            issue = f"Text has {section_count} sections, but at least {self.config.min_sections} are required"
            issues.append(issue)
            suggestions.append(
                f"Add at least {self.config.min_sections - section_count} more sections"
            )

        if self.config.max_sections and section_count > self.config.max_sections:
            issue = f"Text has {section_count} sections, but at most {self.config.max_sections} are allowed"
            issues.append(issue)
            suggestions.append(
                f"Remove at least {section_count - self.config.max_sections} sections"
            )

        # Check required sections
        missing_sections = [s for s in self.config.required_sections if s not in sections]
        if missing_sections:
            issue = f"Text is missing required sections: {', '.join(missing_sections)}"
            issues.append(issue)
            suggestions.append(f"Add the following sections: {', '.join(missing_sections)}")

        # Create result
        result = RuleResult(
            passed=not issues,
            message=issues[0] if issues else "Text structure is valid",
            metadata={
                "section_count": section_count,
                "sections": sections,
                "missing_sections": missing_sections if missing_sections else [],
                "validator_type": self.__class__.__name__,
            },
            score=1.0 if not issues else 0.0,
            issues=issues,
            suggestions=suggestions,
            processing_time_ms=time.time() - start_time,
        )

        # Update statistics
        self.update_statistics(result)

        return result

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


class StructureRule(Rule[str]):
    """
    Rule for validating text structure.

    This rule validates that text meets structure requirements including
    section organization, heading hierarchy, and document structure.

    Lifecycle:
        1. Initialization: Set up with structure constraints
        2. Validation: Check text against structure requirements
        3. Result: Return standardized validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.structure import create_structure_rule

        # Create a rule
        rule = create_structure_rule(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3
        )

        # Validate text
        result = rule.validate("# Introduction\\nContent\\n# Body\\nMore content\\n# Conclusion\\nFinal content")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

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
        self._structure_validator = validator

    def _create_default_validator(self) -> StructureValidator:
        """
        Create a default validator for this rule.

        This method is not used since we create the validator in __init__,
        but it's required by the abstract base class.

        Raises:
            NotImplementedError: Always raised since this method should not be called
        """
        raise NotImplementedError("StructureRule requires a validator to be passed in __init__")


def create_structure_validator(
    required_sections: Optional[List[str]] = None,
    min_sections: int = 1,
    max_sections: Optional[int] = None,
    **kwargs: Any,
) -> StructureValidator:
    """
    Create a structure validator.

    This factory function creates a configured StructureValidator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        required_sections: List of required sections
        min_sections: Minimum number of sections required
        max_sections: Maximum number of sections allowed
        **kwargs: Additional configuration parameters

    Returns:
        A new structure validator instance

    Examples:
        ```python
        from sifaka.rules.formatting.structure import create_structure_validator

        # Create a basic validator
        validator = create_structure_validator(min_sections=3)

        # Create a validator with required sections
        validator = create_structure_validator(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3,
            max_sections=10
        )
        ```
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
    rule_id: Optional[str] = None,
    **kwargs: Any,
) -> StructureRule:
    """
    Create a structure rule.

    This factory function creates a configured StructureRule instance.
    It uses create_structure_validator internally to create the validator.

    Args:
        name: Name of the rule
        description: Description of the rule
        required_sections: List of required sections
        min_sections: Minimum number of sections required
        max_sections: Maximum number of sections allowed
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation
            - params: Dictionary of additional parameters

    Returns:
        A new structure rule instance

    Examples:
        ```python
        from sifaka.rules.formatting.structure import create_structure_rule

        # Create a basic rule
        rule = create_structure_rule(min_sections=3)

        # Create a rule with required sections
        rule = create_structure_rule(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3,
            max_sections=10,
            rule_id="document_structure"
        )

        # Create a rule with additional configuration
        rule = create_structure_rule(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3,
            priority=2,
            severity="warning",
            category="formatting",
            tags=["structure", "formatting"]
        )
        ```
    """
    # Create validator
    validator = create_structure_validator(
        required_sections=required_sections,
        min_sections=min_sections,
        max_sections=max_sections,
    )

    # Determine rule name
    rule_name = name or rule_id or "structure"

    # Create rule config
    config = RuleConfig(
        name=rule_name,
        description=description,
        rule_id=rule_id or rule_name,
        **kwargs,
    )

    # Create and return the rule
    return StructureRule(
        name=rule_name,
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
