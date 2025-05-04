"""
Pydantic models for critics.

This module provides Pydantic models for critic configurations and metadata,
ensuring type safety and validation for critic-related data structures.

## Component Overview

1. **Configuration Models**
   - `CriticConfig`: Base configuration for critics
   - `PromptCriticConfig`: Configuration for prompt-based critics
   - `ReflexionCriticConfig`: Configuration for reflexion critics

2. **Metadata Models**
   - `CriticMetadata`: Metadata for critic results

## Model Lifecycle

### Configuration Models

1. **Initialization**
   - Create with required fields
   - Set optional fields with defaults
   - Validate field values
   - Create immutable instance

2. **Validation**
   - Check field types
   - Verify value ranges
   - Ensure required fields
   - Validate custom rules

3. **Usage**
   - Access configuration values
   - Create modified instances
   - Serialize to/from JSON
   - Validate against schema

### Metadata Models

1. **Creation**
   - Set required fields
   - Initialize optional fields
   - Validate values
   - Create immutable instance

2. **Validation**
   - Check score range
   - Verify field types
   - Ensure required fields
   - Validate custom rules

3. **Usage**
   - Access metadata values
   - Create modified instances
   - Serialize to/from JSON
   - Validate against schema

## Error Handling

1. **Validation Errors**
   - Empty required fields
   - Invalid value ranges
   - Type mismatches
   - Custom rule violations

2. **Recovery Strategies**
   - Default values
   - Field validation
   - Error messages
   - Schema validation

## Examples

Creating and using configuration models:

```python
from sifaka.critics.models import CriticConfig, PromptCriticConfig, ReflexionCriticConfig

# Create a basic critic config
basic_config = CriticConfig(
    name="my_critic",
    description="A basic critic",
    min_confidence=0.8,
    max_attempts=3
)

# Create a prompt critic config
prompt_config = PromptCriticConfig(
    name="prompt_critic",
    description="A prompt-based critic",
    system_prompt="You are an expert editor.",
    temperature=0.7
)

# Create a reflexion critic config
reflexion_config = ReflexionCriticConfig(
    name="reflexion_critic",
    description="A reflexion critic",
    memory_buffer_size=5,
    reflection_depth=2
)
```

Creating and using metadata models:

```python
from sifaka.critics.models import CriticMetadata

# Create metadata for a critic result
metadata = CriticMetadata(
    score=0.85,
    feedback="Good text quality",
    issues=["Could use more detail"],
    suggestions=["Add specific examples"],
    processing_time_ms=150.0
)

# Access metadata values
print(f"Score: {metadata.score:.2f}")
print(f"Feedback: {metadata.feedback}")
print(f"Issues: {metadata.issues}")
print(f"Processing time: {metadata.processing_time_ms}ms")
```

Validating configurations:

```python
from sifaka.critics.models import CriticConfig
from pydantic import ValidationError

try:
    # This will raise a validation error
    config = CriticConfig(
        name="",  # Empty name
        description="A critic",
        min_confidence=1.5  # Invalid confidence
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```
"""

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CriticConfig(BaseModel):
    """
    Configuration for critics.

    This model defines the configuration options for critics, including
    name, description, confidence thresholds, and other settings.

    ## Lifecycle Management

    1. **Initialization**
       - Set required fields (name, description)
       - Configure optional fields with defaults
       - Validate field values
       - Create immutable instance

    2. **Validation**
       - Check field types
       - Verify value ranges
       - Ensure required fields
       - Validate custom rules

    3. **Usage**
       - Access configuration values
       - Create modified instances
       - Serialize to/from JSON
       - Validate against schema

    ## Error Handling

    1. **Validation Errors**
       - Empty name or description
       - Invalid confidence range
       - Negative values
       - Invalid parameters

    2. **Recovery**
       - Default values
       - Field validation
       - Error messages
       - Schema validation

    Examples:
        ```python
        from sifaka.critics.models import CriticConfig

        # Create a basic config
        config = CriticConfig(
            name="my_critic",
            description="A custom critic",
            min_confidence=0.8,
            max_attempts=3
        )

        # Access configuration values
        print(f"Name: {config.name}")
        print(f"Confidence: {config.min_confidence}")

        # Create modified config
        new_config = config.model_copy(
            update={"min_confidence": 0.9}
        )
        ```
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Name of the critic", min_length=1)
    description: str = Field(description="Description of the critic", min_length=1)
    min_confidence: float = Field(
        default=0.7, description="Minimum confidence threshold", ge=0.0, le=1.0
    )
    max_attempts: int = Field(default=3, description="Maximum number of improvement attempts", gt=0)
    cache_size: int = Field(default=100, description="Size of the cache", ge=0)
    priority: int = Field(default=1, description="Priority of the critic", ge=0)
    cost: float = Field(default=1.0, description="Cost of using the critic", ge=0.0)
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """
        Validate that name is not empty.

        This validator ensures that the name field is not empty or
        just whitespace.

        Args:
            v: The name value to validate

        Returns:
            The validated name

        Raises:
            ValueError: If the name is empty or whitespace
        """
        if not v.strip():
            raise ValueError("name cannot be empty or whitespace")
        return v

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, v: str) -> str:
        """
        Validate that description is not empty.

        This validator ensures that the description field is not empty or
        just whitespace.

        Args:
            v: The description value to validate

        Returns:
            The validated description

        Raises:
            ValueError: If the description is empty or whitespace
        """
        if not v.strip():
            raise ValueError("description cannot be empty or whitespace")
        return v


class PromptCriticConfig(CriticConfig):
    """
    Configuration for prompt critics.

    This model extends CriticConfig with prompt-specific settings for
    critics that use language models for text processing.

    ## Lifecycle Management

    1. **Initialization**
       - Set base configuration
       - Configure prompt settings
       - Validate field values
       - Create immutable instance

    2. **Validation**
       - Check field types
       - Verify value ranges
       - Ensure required fields
       - Validate custom rules

    3. **Usage**
       - Access configuration values
       - Create modified instances
       - Serialize to/from JSON
       - Validate against schema

    ## Error Handling

    1. **Validation Errors**
       - Empty system prompt
       - Invalid temperature range
       - Invalid token count
       - Invalid parameters

    2. **Recovery**
       - Default values
       - Field validation
       - Error messages
       - Schema validation

    Examples:
        ```python
        from sifaka.critics.models import PromptCriticConfig

        # Create a prompt critic config
        config = PromptCriticConfig(
            name="prompt_critic",
            description="A prompt-based critic",
            system_prompt="You are an expert editor.",
            temperature=0.7,
            max_tokens=1000
        )

        # Access configuration values
        print(f"System prompt: {config.system_prompt}")
        print(f"Temperature: {config.temperature}")

        # Create modified config
        new_config = config.model_copy(
            update={"temperature": 0.8}
        )
        ```
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str = Field(
        default="You are an expert editor that improves text.",
        description="System prompt for the model",
        min_length=1,
    )
    temperature: float = Field(
        default=0.7, description="Temperature for model generation", ge=0.0, le=1.0
    )
    max_tokens: int = Field(default=1000, description="Maximum tokens for model generation", gt=0)

    @field_validator("system_prompt")
    @classmethod
    def system_prompt_must_not_be_empty(cls, v: str) -> str:
        """
        Validate that system_prompt is not empty.

        This validator ensures that the system_prompt field is not empty or
        just whitespace.

        Args:
            v: The system_prompt value to validate

        Returns:
            The validated system_prompt

        Raises:
            ValueError: If the system_prompt is empty or whitespace
        """
        if not v.strip():
            raise ValueError("system_prompt cannot be empty or whitespace")
        return v


class ReflexionCriticConfig(PromptCriticConfig):
    """
    Configuration for reflexion critics.

    This model extends PromptCriticConfig with reflexion-specific settings
    for critics that use reflection to improve text.

    ## Lifecycle Management

    1. **Initialization**
       - Set base configuration
       - Configure reflexion settings
       - Validate field values
       - Create immutable instance

    2. **Validation**
       - Check field types
       - Verify value ranges
       - Ensure required fields
       - Validate custom rules

    3. **Usage**
       - Access configuration values
       - Create modified instances
       - Serialize to/from JSON
       - Validate against schema

    ## Error Handling

    1. **Validation Errors**
       - Invalid buffer size
       - Invalid reflection depth
       - Invalid parameters
       - Custom rule violations

    2. **Recovery**
       - Default values
       - Field validation
       - Error messages
       - Schema validation

    Examples:
        ```python
        from sifaka.critics.models import ReflexionCriticConfig

        # Create a reflexion critic config
        config = ReflexionCriticConfig(
            name="reflexion_critic",
            description="A reflexion critic",
            memory_buffer_size=5,
            reflection_depth=2
        )

        # Access configuration values
        print(f"Buffer size: {config.memory_buffer_size}")
        print(f"Reflection depth: {config.reflection_depth}")

        # Create modified config
        new_config = config.model_copy(
            update={"reflection_depth": 3}
        )
        ```
    """

    memory_buffer_size: int = Field(
        default=5, description="Maximum number of reflections to store", gt=0
    )
    reflection_depth: int = Field(
        default=1, description="How many levels of reflection to perform", gt=0
    )


class CriticMetadata(BaseModel):
    """
    Metadata for critic results.

    This model defines the metadata structure for critic results,
    including score, feedback, issues, and suggestions.

    ## Lifecycle Management

    1. **Creation**
       - Set required fields
       - Initialize optional fields
       - Validate values
       - Create immutable instance

    2. **Validation**
       - Check score range
       - Verify field types
       - Ensure required fields
       - Validate custom rules

    3. **Usage**
       - Access metadata values
       - Create modified instances
       - Serialize to/from JSON
       - Validate against schema

    ## Error Handling

    1. **Validation Errors**
       - Invalid score range
       - Negative processing time
       - Invalid attempt number
       - Missing required fields

    2. **Recovery**
       - Default values
       - Field validation
       - Error messages
       - Schema validation

    Examples:
        ```python
        from sifaka.critics.models import CriticMetadata

        # Create metadata for a critic result
        metadata = CriticMetadata(
            score=0.85,
            feedback="Good text quality",
            issues=["Could use more detail"],
            suggestions=["Add specific examples"],
            processing_time_ms=150.0
        )

        # Access metadata values
        print(f"Score: {metadata.score:.2f}")
        print(f"Feedback: {metadata.feedback}")
        print(f"Issues: {metadata.issues}")
        print(f"Processing time: {metadata.processing_time_ms}ms")

        # Create modified metadata
        new_metadata = metadata.model_copy(
            update={"score": 0.9}
        )
        ```
    """

    model_config = ConfigDict(frozen=True)

    score: float = Field(description="Score between 0 and 1", ge=0.0, le=1.0)
    feedback: str = Field(description="General feedback")
    issues: List[str] = Field(default_factory=list, description="List of issues")
    suggestions: List[str] = Field(default_factory=list, description="List of suggestions")
    attempt_number: int = Field(default=1, description="Attempt number", gt=0)
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds", ge=0.0
    )
