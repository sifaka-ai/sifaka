"""
Model result models for Sifaka.

This module provides standardized result models for model providers,
ensuring consistent result handling across the framework.
"""
from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field, ConfigDict
T = TypeVar('T')


class ModelResult(BaseModel, Generic[T]):
    """
    Result of a model operation.

    This class provides an immutable representation of a model operation result,
    including the generated output and additional metadata.

    ## Lifecycle

    1. **Creation**: Instantiate with operation results
       - Provide output (required)
       - Add optional metadata dictionary
       - Values are validated during creation

    2. **Access**: Read properties to get operation details
       - Access output for the operation result
       - Examine metadata for additional information

    ## Examples

    ```python
    from sifaka.models.result import ModelResult

    # Create a result
    result = ModelResult(
        output="Generated text",
        metadata={
            "token_count": 50,
            "generation_time": 1.2,
            "model": "gpt-4"
        }
    )

    # Access the result
    print(f"Output: {result.output}")
    print(f"Token count: {result.metadata.get('token_count') if metadata else ""}")
    ```
    """
    model_config = ConfigDict(frozen=True, extra='forbid')
    output: T = Field(description='The generated output')
    metadata: Dict[str, Any] = Field(default_factory=dict, description=
        'Additional metadata about the operation')


class GenerationResult(ModelResult[str]):
    """
    Result of a text generation operation.

    This class extends ModelResult with text generation specific fields,
    providing a standardized format for text generation results.

    ## Lifecycle

    1. **Creation**: Instantiate with generation results
       - Provide output (required)
       - Add prompt_tokens and completion_tokens
       - Add optional metadata dictionary
       - Values are validated during creation

    2. **Access**: Read properties to get generation details
       - Access output for the generated text
       - Check prompt_tokens and completion_tokens for token usage
       - Examine metadata for additional information

    ## Examples

    ```python
    from sifaka.models.result import GenerationResult

    # Create a result
    result = GenerationResult(
        output="Generated text",
        prompt_tokens=10,
        completion_tokens=40,
        metadata={
            "generation_time": 1.2,
            "model": "gpt-4"
        }
    )

    # Access the result
    print(f"Output: {result.output}")
    print(f"Prompt tokens: {result.prompt_tokens}")
    print(f"Completion tokens: {result.completion_tokens}")
    print(f"Total tokens: {result.total_tokens}")
    ```
    """
    prompt_tokens: int = Field(default=0, ge=0, description=
        'Number of tokens in the prompt')
    completion_tokens: int = Field(default=0, ge=0, description=
        'Number of tokens in the completion')

    @property
    def total_tokens(self) ->Any:
        """
        Get the total number of tokens.

        Returns:
            The total number of tokens (prompt + completion)
        """
        return self.prompt_tokens + self.completion_tokens


class TokenCountResult(ModelResult[int]):
    """
    Result of a token counting operation.

    This class extends ModelResult with token counting specific fields,
    providing a standardized format for token counting results.

    ## Lifecycle

    1. **Creation**: Instantiate with token counting results
       - Provide token_count (required)
       - Add optional metadata dictionary
       - Values are validated during creation

    2. **Access**: Read properties to get token counting details
       - Access output for the token count
       - Examine metadata for additional information

    ## Examples

    ```python
    from sifaka.models.result import TokenCountResult

    # Create a result
    result = TokenCountResult(
        output=50,
        metadata={
            "text_length": 200,
            "model": "gpt-4"
        }
    )

    # Access the result
    print(f"Token count: {result.output}")
    print(f"Text length: {result.metadata.get('text_length') if metadata else ""}")
    ```
    """

    def __init__(self, token_count: int, **kwargs: Any):
        """
        Initialize a TokenCountResult instance.

        Args:
            token_count: The token count
            **kwargs: Additional keyword arguments
        """
        super().__init__(output=token_count, **kwargs)
