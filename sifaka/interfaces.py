"""
Core interfaces for Sifaka.

This module defines the protocol interfaces that all Sifaka components must follow.
These interfaces establish the contract between different parts of the system
without creating circular dependencies.
"""

from typing import Protocol, Any, Dict, Tuple, TypeVar, Generic, Optional

# Type variables for generic protocols
T = TypeVar('T')
U = TypeVar('U')


class Result(Protocol):
    """Protocol for result objects."""
    
    @property
    def passed(self) -> bool:
        """Whether the operation passed."""
        ...
        
    @property
    def message(self) -> str:
        """Message describing the result."""
        ...


class ValidationResult(Result):
    """Protocol for validation results."""
    
    @property
    def details(self) -> Dict[str, Any]:
        """Additional details about the validation result."""
        ...


class ImprovementResult(Result):
    """Protocol for improvement results."""
    
    @property
    def original_text(self) -> str:
        """The original text before improvement."""
        ...
        
    @property
    def improved_text(self) -> str:
        """The improved text."""
        ...
        
    @property
    def changes_made(self) -> bool:
        """Whether changes were made to the text."""
        ...
        
    @property
    def details(self) -> Dict[str, Any]:
        """Additional details about the improvement result."""
        ...


class Model(Protocol):
    """Protocol defining the interface for model providers."""

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the model.

        Returns:
            The generated text.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens in the text.
        """
        ...


class Validator(Protocol):
    """Protocol defining the interface for validators."""

    def validate(self, text: str) -> ValidationResult:
        """Validate text and return a result.
        
        Args:
            text: The text to validate.
            
        Returns:
            A validation result.
        """
        ...


class Improver(Protocol):
    """Protocol defining the interface for improvers."""

    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        """Improve text and return the improved text and a result.
        
        Args:
            text: The text to improve.
            
        Returns:
            A tuple of (improved_text, improvement_result).
        """
        ...


class Factory(Protocol, Generic[T]):
    """Protocol for factory functions."""
    
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Create an instance of the component.
        
        Args:
            *args: Positional arguments to pass to the component constructor.
            **kwargs: Keyword arguments to pass to the component constructor.
            
        Returns:
            An instance of the component.
        """
        ...


class ModelFactory(Factory[Model]):
    """Protocol for model factory functions."""
    
    def __call__(self, model_name: str, **options: Any) -> Model:
        """Create a model instance.
        
        Args:
            model_name: The name of the model to create.
            **options: Additional options to pass to the model constructor.
            
        Returns:
            A model instance.
        """
        ...


class ValidatorFactory(Factory[Validator]):
    """Protocol for validator factory functions."""
    
    def __call__(self, **options: Any) -> Validator:
        """Create a validator instance.
        
        Args:
            **options: Options to pass to the validator constructor.
            
        Returns:
            A validator instance.
        """
        ...


class ImproverFactory(Factory[Improver]):
    """Protocol for improver factory functions."""
    
    def __call__(self, model: Model, **options: Any) -> Improver:
        """Create an improver instance.
        
        Args:
            model: The model to use for improvement.
            **options: Additional options to pass to the improver constructor.
            
        Returns:
            An improver instance.
        """
        ...


class Registry(Protocol):
    """Protocol for the component registry."""
    
    def register_model(self, provider: str, factory: ModelFactory) -> None:
        """Register a model factory.
        
        Args:
            provider: The provider name (e.g., "openai", "anthropic").
            factory: The factory function that creates models for this provider.
        """
        ...
        
    def get_model_factory(self, provider: str) -> Optional[ModelFactory]:
        """Get a model factory.
        
        Args:
            provider: The provider name (e.g., "openai", "anthropic").
            
        Returns:
            The factory function, or None if not found.
        """
        ...
        
    def register_validator(self, name: str, factory: ValidatorFactory) -> None:
        """Register a validator factory.
        
        Args:
            name: The name of the validator.
            factory: The factory function that creates validators.
        """
        ...
        
    def get_validator_factory(self, name: str) -> Optional[ValidatorFactory]:
        """Get a validator factory.
        
        Args:
            name: The name of the validator.
            
        Returns:
            The factory function, or None if not found.
        """
        ...
        
    def register_improver(self, name: str, factory: ImproverFactory) -> None:
        """Register an improver factory.
        
        Args:
            name: The name of the improver.
            factory: The factory function that creates improvers.
        """
        ...
        
    def get_improver_factory(self, name: str) -> Optional[ImproverFactory]:
        """Get an improver factory.
        
        Args:
            name: The name of the improver.
            
        Returns:
            The factory function, or None if not found.
        """
        ...
