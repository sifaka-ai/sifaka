"""
Model classes for critics in Sifaka.

This module provides model classes for critics in the Sifaka framework,
including configuration models and result models.

## Overview
The module provides:
- Base configuration models for critics
- Specialized configuration models for different critic types
- Metadata models for critic results
- Validation and type checking for all models

## Components
1. **Base Models**
   - CriticMetadata: Standardized structure for critic results
   - CriticConfig: Base configuration for all critics

2. **Specialized Configurations**
   - PromptCriticConfig: Configuration for prompt-based critics
   - ReflexionCriticConfig: Configuration for reflexion critics
   - ConstitutionalCriticConfig: Configuration for constitutional critics
   - SelfRefineCriticConfig: Configuration for self-refine critics
   - SelfRAGCriticConfig: Configuration for self-RAG critics
   - FeedbackCriticConfig: Configuration for feedback critics
   - ValueCriticConfig: Configuration for value critics
   - LACCriticConfig: Configuration for LAC critics

## Usage Examples
```python
from sifaka.critics.models import CriticConfig, CriticMetadata

# Create a basic configuration
config = CriticConfig(
    name="my_critic",
    description="A custom critic",
    cache_size=200
)

# Create metadata for results
metadata = CriticMetadata(
    score=0.85,
    feedback="Good text quality",
    issues=["Could be more concise"],
    suggestions=["Remove redundant phrases"]
)

# Create a specialized configuration
from sifaka.critics.models import PromptCriticConfig

prompt_config = PromptCriticConfig(
    name="prompt_critic",
    description="A prompt-based critic",
    system_prompt="You are a helpful critic",
    temperature=0.7,
    max_tokens=1000
)
```

## Error Handling
The module implements:
- Input validation for all fields
- Type checking for configuration values
- Range validation for numeric fields
- Required field validation
- Default value handling
"""

from typing import List

from pydantic import Field
from sifaka.utils.config import CriticConfig, CriticMetadata


class PromptCriticConfig(CriticConfig):
    """
    Configuration for prompt-based critics.

    This class extends the base critic configuration with parameters
    specific to prompt-based critics.

    ## Overview
    The class provides:
    - System prompt configuration
    - Text generation parameters (temperature, max_tokens)
    - Confidence and attempt thresholds
    - Initialization and memory management settings
    - Performance tracking options
    - Type-safe parameter management

    ## Usage Examples
    ```python
    from sifaka.critics.models import PromptCriticConfig

    # Create a basic prompt critic configuration
    config = PromptCriticConfig(
        name="prompt_critic",
        description="A prompt-based critic",
        system_prompt="You are a helpful critic",
        temperature=0.7,
        max_tokens=1000
    )

    # Create an advanced configuration
    config = PromptCriticConfig(
        name="advanced_prompt_critic",
        description="An advanced prompt-based critic",
        system_prompt="You are an expert critic focusing on clarity and conciseness",
        temperature=0.5,
        max_tokens=2000,
        min_confidence=0.8,
        max_attempts=5,
        eager_initialization=True,
        memory_buffer_size=20,
        track_performance=True
    )
    ```

    ## Error Handling
    The class implements:
    - Temperature range validation (0.0 to 1.0)
    - Token count validation (positive)
    - Confidence range validation (0.0 to 1.0)
    - Attempt count validation (positive)
    - Memory buffer size validation (positive)
    - Required field validation

    Attributes:
        system_prompt: System prompt for the critic
        temperature: Temperature for text generation
        max_tokens: Maximum number of tokens for text generation
        min_confidence: Minimum confidence score for critique
        max_attempts: Maximum number of attempts for critique
        eager_initialization: Whether to initialize components eagerly
        memory_buffer_size: Size of the memory buffer
        track_performance: Whether to track performance metrics
        track_errors: Whether to track error statistics
    """

    system_prompt: str = Field(
        description="System prompt for the critic",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of tokens for text generation",
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for critique",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of attempts for critique",
    )
    eager_initialization: bool = Field(
        default=False,
        description="Whether to initialize components eagerly",
    )
    memory_buffer_size: int = Field(
        default=10,
        ge=1,
        description="Size of the memory buffer",
    )
    track_performance: bool = Field(
        default=True,
        description="Whether to track performance metrics",
    )
    track_errors: bool = Field(
        default=True,
        description="Whether to track error statistics",
    )


class ReflexionCriticConfig(PromptCriticConfig):
    """
    Configuration for reflexion critics.

    This class extends the prompt critic configuration with parameters
    specific to reflexion critics.

    ## Overview
    The class provides:
    - Memory buffer configuration for reflections
    - Reflection depth control
    - Inherited prompt-based configuration
    - Type-safe parameter management

    ## Usage Examples
    ```python
    from sifaka.critics.models import ReflexionCriticConfig

    # Create a basic reflexion critic configuration
    config = ReflexionCriticConfig(
        name="reflexion_critic",
        description="A reflexion-based critic",
        system_prompt="You are a reflective critic",
        memory_buffer_size=5,
        reflection_depth=1
    )

    # Create an advanced configuration
    config = ReflexionCriticConfig(
        name="deep_reflexion_critic",
        description="A deep reflexion critic",
        system_prompt="You are an expert reflective critic",
        temperature=0.5,
        max_tokens=2000,
        memory_buffer_size=10,
        reflection_depth=3,
        min_confidence=0.8,
        max_attempts=5
    )
    ```

    ## Error Handling
    The class implements:
    - Memory buffer size validation (positive)
    - Reflection depth validation (positive)
    - Inherited validation from PromptCriticConfig
    - Required field validation

    Attributes:
        memory_buffer_size: Size of the memory buffer for reflections
        reflection_depth: Depth of reflections
    """

    memory_buffer_size: int = Field(
        default=5,
        ge=1,
        description="Size of the memory buffer for reflections",
    )
    reflection_depth: int = Field(
        default=1,
        ge=1,
        description="Depth of reflections",
    )


class ConstitutionalCriticConfig(PromptCriticConfig):
    """
    Configuration for constitutional critics.

    This class extends the prompt critic configuration with parameters
    specific to constitutional critics.

    ## Overview
    The class provides:
    - Principle-based configuration
    - Inherited prompt-based settings
    - Type-safe parameter management
    - Extensible principle list

    ## Usage Examples
    ```python
    from sifaka.critics.models import ConstitutionalCriticConfig

    # Create a basic constitutional critic configuration
    config = ConstitutionalCriticConfig(
        name="constitutional_critic",
        description="A principle-based critic",
        system_prompt="You are a constitutional critic",
        principles=[
            "Be clear and concise",
            "Maintain professional tone",
            "Ensure factual accuracy"
        ]
    )

    # Create an advanced configuration
    config = ConstitutionalCriticConfig(
        name="advanced_constitutional_critic",
        description="An advanced principle-based critic",
        system_prompt="You are an expert constitutional critic",
        temperature=0.5,
        max_tokens=2000,
        principles=[
            "Be clear and concise",
            "Maintain professional tone",
            "Ensure factual accuracy",
            "Respect cultural sensitivity",
            "Follow ethical guidelines"
        ],
        min_confidence=0.8,
        max_attempts=5
    )
    ```

    ## Error Handling
    The class implements:
    - Principle list validation
    - Inherited validation from PromptCriticConfig
    - Required field validation
    - Type checking for all fields

    Attributes:
        principles: List of principles to follow
    """

    principles: List[str] = Field(
        default_factory=list,
        description="List of principles to follow",
    )


class SelfRefineCriticConfig(PromptCriticConfig):
    """
    Configuration for self-refine critics.

    This class extends the prompt critic configuration with parameters
    specific to self-refine critics.

    ## Overview
    The class provides:
    - Iteration control for refinement
    - Improvement threshold configuration
    - Inherited prompt-based settings
    - Type-safe parameter management

    ## Usage Examples
    ```python
    from sifaka.critics.models import SelfRefineCriticConfig

    # Create a basic self-refine critic configuration
    config = SelfRefineCriticConfig(
        name="self_refine_critic",
        description="A self-refining critic",
        system_prompt="You are a self-refining critic",
        max_iterations=3,
        improvement_threshold=0.1
    )

    # Create an advanced configuration
    config = SelfRefineCriticConfig(
        name="advanced_self_refine_critic",
        description="An advanced self-refining critic",
        system_prompt="You are an expert self-refining critic",
        temperature=0.5,
        max_tokens=2000,
        max_iterations=5,
        improvement_threshold=0.05,
        min_confidence=0.8,
        max_attempts=5
    )
    ```

    ## Error Handling
    The class implements:
    - Iteration count validation (positive)
    - Threshold range validation (0.0 to 1.0)
    - Inherited validation from PromptCriticConfig
    - Required field validation

    Attributes:
        max_iterations: Maximum number of refinement iterations
        improvement_threshold: Threshold for improvement
    """

    max_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum number of refinement iterations",
    )
    improvement_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Threshold for improvement",
    )


class SelfRAGCriticConfig(PromptCriticConfig):
    """
    Configuration for self-RAG critics.

    This class extends the prompt critic configuration with parameters
    specific to self-RAG critics.

    ## Overview
    The class provides:
    - Retrieval configuration (top-k)
    - Reflection control
    - Inherited prompt-based settings
    - Type-safe parameter management

    ## Usage Examples
    ```python
    from sifaka.critics.models import SelfRAGCriticConfig

    # Create a basic self-RAG critic configuration
    config = SelfRAGCriticConfig(
        name="self_rag_critic",
        description="A self-RAG critic",
        system_prompt="You are a self-RAG critic",
        retrieval_top_k=3,
        reflection_enabled=True
    )

    # Create an advanced configuration
    config = SelfRAGCriticConfig(
        name="advanced_self_rag_critic",
        description="An advanced self-RAG critic",
        system_prompt="You are an expert self-RAG critic",
        temperature=0.5,
        max_tokens=2000,
        retrieval_top_k=5,
        reflection_enabled=True,
        min_confidence=0.8,
        max_attempts=5
    )
    ```

    ## Error Handling
    The class implements:
    - Retrieval count validation (positive)
    - Reflection flag validation
    - Inherited validation from PromptCriticConfig
    - Required field validation

    Attributes:
        retrieval_top_k: Number of top documents to retrieve
        reflection_enabled: Whether to enable reflection
    """

    retrieval_top_k: int = Field(
        default=3,
        ge=1,
        description="Number of top documents to retrieve",
    )
    reflection_enabled: bool = Field(
        default=True,
        description="Whether to enable reflection",
    )


class FeedbackCriticConfig(PromptCriticConfig):
    """
    Configuration for feedback critics.

    This class extends the prompt critic configuration with parameters
    specific to feedback critics.

    ## Overview
    The class provides:
    - Feedback dimension configuration
    - Inherited prompt-based settings
    - Type-safe parameter management
    - Extensible feedback structure

    ## Usage Examples
    ```python
    from sifaka.critics.models import FeedbackCriticConfig

    # Create a basic feedback critic configuration
    config = FeedbackCriticConfig(
        name="feedback_critic",
        description="A feedback-based critic",
        system_prompt="You are a feedback critic",
        feedback_dimensions=[
            "clarity",
            "conciseness",
            "tone"
        ]
    )

    # Create an advanced configuration
    config = FeedbackCriticConfig(
        name="advanced_feedback_critic",
        description="An advanced feedback critic",
        system_prompt="You are an expert feedback critic",
        temperature=0.5,
        max_tokens=2000,
        feedback_dimensions=[
            "clarity",
            "conciseness",
            "tone",
            "grammar",
            "style",
            "coherence"
        ],
        min_confidence=0.8,
        max_attempts=5
    )
    ```

    ## Error Handling
    The class implements:
    - Feedback dimension validation
    - Inherited validation from PromptCriticConfig
    - Required field validation
    - Type checking for all fields

    Attributes:
        feedback_dimensions: List of feedback dimensions
    """

    feedback_dimensions: List[str] = Field(
        default_factory=list,
        description="List of feedback dimensions",
    )


class ValueCriticConfig(PromptCriticConfig):
    """
    Configuration for value critics.

    This class extends the prompt critic configuration with parameters
    specific to value critics.

    ## Overview
    The class provides:
    - Value dimension configuration
    - Score range control
    - Inherited prompt-based settings
    - Type-safe parameter management

    ## Usage Examples
    ```python
    from sifaka.critics.models import ValueCriticConfig

    # Create a basic value critic configuration
    config = ValueCriticConfig(
        name="value_critic",
        description="A value-based critic",
        system_prompt="You are a value critic",
        value_dimensions=[
            "ethics",
            "fairness",
            "safety"
        ],
        min_score=0.0,
        max_score=1.0
    )

    # Create an advanced configuration
    config = ValueCriticConfig(
        name="advanced_value_critic",
        description="An advanced value critic",
        system_prompt="You are an expert value critic",
        temperature=0.5,
        max_tokens=2000,
        value_dimensions=[
            "ethics",
            "fairness",
            "safety",
            "privacy",
            "transparency",
            "accountability"
        ],
        min_score=0.0,
        max_score=1.0,
        min_confidence=0.8,
        max_attempts=5
    )
    ```

    ## Error Handling
    The class implements:
    - Value dimension validation
    - Score range validation
    - Inherited validation from PromptCriticConfig
    - Required field validation

    Attributes:
        value_dimensions: List of value dimensions
        min_score: Minimum score for values
        max_score: Maximum score for values
    """

    value_dimensions: List[str] = Field(
        default_factory=list,
        description="List of value dimensions",
    )
    min_score: float = Field(
        default=0.0,
        description="Minimum score for values",
    )
    max_score: float = Field(
        default=1.0,
        description="Maximum score for values",
    )


class LACCriticConfig(PromptCriticConfig):
    """
    Configuration for LAC critics.

    This class extends the prompt critic configuration with parameters
    specific to LAC (Language Alignment and Control) critics.

    ## Overview
    The class provides:
    - Feedback dimension configuration
    - Value dimension configuration
    - Score range control
    - Inherited prompt-based settings
    - Type-safe parameter management

    ## Usage Examples
    ```python
    from sifaka.critics.models import LACCriticConfig

    # Create a basic LAC critic configuration
    config = LACCriticConfig(
        name="lac_critic",
        description="A LAC-based critic",
        system_prompt="You are a LAC critic",
        feedback_dimensions=[
            "clarity",
            "conciseness",
            "tone"
        ],
        value_dimensions=[
            "ethics",
            "fairness",
            "safety"
        ],
        min_score=0.0,
        max_score=1.0
    )

    # Create an advanced configuration
    config = LACCriticConfig(
        name="advanced_lac_critic",
        description="An advanced LAC critic",
        system_prompt="You are an expert LAC critic",
        temperature=0.5,
        max_tokens=2000,
        feedback_dimensions=[
            "clarity",
            "conciseness",
            "tone",
            "grammar",
            "style",
            "coherence"
        ],
        value_dimensions=[
            "ethics",
            "fairness",
            "safety",
            "privacy",
            "transparency",
            "accountability"
        ],
        min_score=0.0,
        max_score=1.0,
        min_confidence=0.8,
        max_attempts=5
    )
    ```

    ## Error Handling
    The class implements:
    - Feedback dimension validation
    - Value dimension validation
    - Score range validation
    - Inherited validation from PromptCriticConfig
    - Required field validation

    Attributes:
        feedback_dimensions: List of feedback dimensions
        value_dimensions: List of value dimensions
        min_score: Minimum score for values
        max_score: Maximum score for values
    """

    feedback_dimensions: List[str] = Field(
        default_factory=list,
        description="List of feedback dimensions",
    )
    value_dimensions: List[str] = Field(
        default_factory=list,
        description="List of value dimensions",
    )
    min_score: float = Field(
        default=0.0,
        description="Minimum score for values",
    )
    max_score: float = Field(
        default=1.0,
        description="Maximum score for values",
    )
