"""
Structure validation rules for Sifaka.

This module provides rules for validating text structure, including
section organization, heading hierarchy, and document structure.

## Overview
The structure validation rules help ensure that text follows a specific
organizational structure with required sections and section count constraints.
This is particularly useful for validating documents that need to follow
a specific format or template.

## Components
- **StructureConfig**: Configuration for structure validation
- **StructureValidator**: Validator for text structure
- **StructureRule**: Rule for validating text structure
- **Factory Functions**: create_structure_validator, create_structure_rule

## Usage Examples
```python
from sifaka.rules.formatting.structure import create_structure_rule

# Create a structure rule
structure_rule = create_structure_rule(
    required_sections=["introduction", "body", "conclusion"],
    min_sections=3
)

# Validate text
result = structure_rule.validate("# Introduction\\nContent\\n# Body\\nMore content\\n# Conclusion\\nFinal content")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```

## Error Handling
- Empty text handling through BaseValidator.handle_empty_text
- Section extraction with error handling
- Detailed validation results with issues and suggestions
"""

from typing import List, Optional, Any
import time

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_rule_state
from sifaka.utils.errors.handling import try_operation

logger = get_logger(__name__)


class StructureConfig(BaseModel):
    """
    Configuration for structure validation.

    This class defines the configuration options for text structure validation,
    including required sections, section count constraints, and caching settings.

    ## Architecture
    The class uses Pydantic for validation and immutability, with field validators
    to ensure proper configuration values.

    ## Lifecycle
    1. **Creation**: Instantiate with default or custom values
       - Create directly with parameters
       - Create from dictionary with model_validate

    2. **Validation**: Values are validated by Pydantic
       - Type checking for all fields
       - Range validation for min_sections and max_sections
       - Immutability enforced by frozen=True

    3. **Usage**: Pass to validators and rules
       - Used by StructureValidator
       - Used by StructureRule._create_default_validator

    ## Examples
    ```python
    from sifaka.rules.formatting.structure import StructureConfig

    # Create with default values
    config = StructureConfig()

    # Create with custom values
    config = StructureConfig(
        required_sections=["introduction", "body", "conclusion"],
        min_sections=3,
        max_sections=10,
        cache_size=200
    )
    ```
    """

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

    ## Architecture
    The StructureValidator follows a component-based architecture:
    - Inherits from BaseValidator for common validation functionality
    - Uses StateManager for state management via _state_manager
    - Implements section analysis logic
    - Uses StructureConfig for configuration
    - Implements caching for performance optimization
    - Provides detailed validation results with metadata

    ## Lifecycle
    1. **Initialization**: Set up with structure constraints
       - Initialize with StructureConfig containing structure parameters
       - Store configuration in state manager
       - Set metadata for tracking and debugging
       - Initialize validation cache

    2. **Validation**: Check text against structure requirements
       - Handle empty text through BaseValidator.handle_empty_text
       - Check cache for previously validated text
       - Analyze sections using _analyze_sections
       - Validate section count against min_sections and max_sections
       - Validate required sections
       - Return detailed RuleResult with validation results
       - Cache results if caching is enabled

    3. **Error Handling**: Manage validation errors
       - Use try_operation for error handling
       - Provide detailed error information in result
       - Update validation statistics

    ## Examples
        ```python
        from sifaka.rules.formatting.structure import create_structure_validator

        # Create a validator
        validator = create_structure_validator(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3
        )

        # Validate text
        result = validator.validate("# Introduction\\nContent\\n# Body\\nMore content\\n# Conclusion\\nFinal content")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")

        # Access validation details
        if not result.passed:
            print(f"Issues: {result.issues}")
            print(f"Suggestions: {result.suggestions}")
            print(f"Missing sections: {result.metadata.get('missing_sections', [])}")
        ```
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(self, config: Optional[StructureConfig] = None):
        """
        Initialize the validator.

        Args:
            config: Configuration for the validator
        """
        super().__init__(validation_type=str)

        # Store configuration in state
        self._state_manager.update("config", config or StructureConfig())

        # Set metadata
        self._state_manager.set_metadata("validator_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

        # Initialize cache
        self._state_manager.update("cache", {})

    @property
    def config(self) -> StructureConfig:
        """
        Get the validator configuration.

        Returns:
            The validator configuration
        """
        return self._state_manager.get("config")

    def validate(self, text: str) -> RuleResult:
        """
        Validate text structure.

        Args:
            text: Text to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        # Check cache if enabled
        cache_size = self.config.cache_size
        if cache_size > 0:
            cache = self._state_manager.get("cache", {})
            if text in cache:
                self._state_manager.set_metadata("cache_hit", True)
                return cache[text]
            self._state_manager.set_metadata("cache_hit", False)

        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        # Define the validation operation
        def validation_operation():
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

            return result

        # Use try_operation to handle potential errors
        result = try_operation(
            validation_operation,
            component_name=self.__class__.__name__,
            default_value=RuleResult(
                passed=False,
                message="Error validating text structure",
                metadata={
                    "error_type": "ValidationError",
                    "validator_type": self.__class__.__name__,
                },
                score=0.0,
                issues=["Error validating text structure"],
                suggestions=["Check input format and try again"],
                processing_time_ms=time.time() - start_time,
            ),
        )

        # Update statistics
        self.update_statistics(result)

        # Update validation count in metadata
        validation_count = self._state_manager.get_metadata("validation_count", 0)
        self._state_manager.set_metadata("validation_count", validation_count + 1)

        # Cache result if caching is enabled
        if cache_size > 0:
            cache = self._state_manager.get("cache", {})
            if len(cache) >= cache_size:
                # Clear cache if it's full
                cache = {}
            cache[text] = result
            self._state_manager.update("cache", cache)

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

    ## Architecture
    The StructureRule follows a component-based architecture:
    - Inherits from Rule for common rule functionality
    - Uses StateManager for state management via _state_manager
    - Delegates validation to StructureValidator
    - Uses RuleConfig for configuration
    - Creates a default validator if none is provided
    - Provides standardized validation results with metadata

    ## Lifecycle
    1. **Initialization**: Set up with structure constraints
       - Initialize with name, description, config, and optional validator
       - Create default validator if none is provided
       - Store validator in state manager
       - Store validator config in state for reference

    2. **Validation**: Check text against structure requirements
       - Inherited from Rule base class
       - Delegate to StructureValidator for structure validation
       - Add rule_id to metadata for traceability
       - Return standardized RuleResult with validation results

    3. **Default Validator Creation**: Create validator from config
       - Extract structure parameters from rule config
       - Create StructureValidator with appropriate configuration
       - Store validator config in state for reference

    ## Examples
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

        # Access validation details
        if not result.passed:
            print(f"Issues: {result.issues}")
            print(f"Suggestions: {result.suggestions}")
            print(f"Missing sections: {result.metadata.get('missing_sections', [])}")
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

        # Store validator in state
        self._state_manager.update("structure_validator", validator)

        # Store validator config in state for reference
        self._state_manager.update("validator_config", validator.config)

    def _create_default_validator(self) -> StructureValidator:
        """
        Create a default validator for this rule.

        This method creates a default validator using parameters from the rule config.
        It's used when a validator is not provided in the constructor.

        Returns:
            A configured StructureValidator
        """
        # Extract structure specific params
        params = self.config.params
        config = StructureConfig(
            required_sections=params.get("required_sections", []),
            min_sections=params.get("min_sections", 1),
            max_sections=params.get("max_sections", None),
            cache_size=self.config.cache_size,
        )

        # Store config in state for reference
        self._state_manager.update("validator_config", config)

        return StructureValidator(config=config)


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
    try:
        # Create config with default or provided values
        config_params = {}
        if required_sections is not None:
            config_params["required_sections"] = required_sections
        if min_sections is not None:
            config_params["min_sections"] = min_sections
        if max_sections is not None:
            config_params["max_sections"] = max_sections

        # Add any remaining config parameters
        config_params.update(kwargs)

        # Create config
        config = StructureConfig(**config_params)

        # Create and return the validator
        return StructureValidator(config=config)

    except Exception as e:
        logger.error(f"Error creating structure validator: {e}")
        raise ValueError(f"Error creating structure validator: {str(e)}")


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
    try:
        # Create validator using the validator factory
        validator = create_structure_validator(
            required_sections=required_sections,
            min_sections=min_sections,
            max_sections=max_sections,
        )

        # Create params dictionary for RuleConfig
        params = {}
        if required_sections is not None:
            params["required_sections"] = required_sections
        if min_sections is not None:
            params["min_sections"] = min_sections
        if max_sections is not None:
            params["max_sections"] = max_sections

        # Determine rule name
        rule_name = name or rule_id or "structure"

        # Create rule config
        config_kwargs = {k: v for k, v in kwargs.items() if k not in ["params"]}
        if "params" in kwargs:
            params.update(kwargs["params"])

        config = RuleConfig(
            name=rule_name,
            description=description,
            rule_id=rule_id or rule_name,
            params=params,
            **config_kwargs,
        )

        # Create and return the rule
        return StructureRule(
            name=rule_name,
            description=description,
            config=config,
            validator=validator,
        )

    except Exception as e:
        logger.error(f"Error creating structure rule: {e}")
        raise ValueError(f"Error creating structure rule: {str(e)}")


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
