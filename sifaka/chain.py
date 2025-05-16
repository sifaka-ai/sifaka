"""
Chain orchestration for Sifaka.

This module defines the Chain class, which is the main entry point for the Sifaka framework.
It orchestrates the generation, validation, and improvement of text using LLMs.
"""

from typing import Union, List, Dict, Any, Protocol, Optional

from sifaka.models.base import Model, create_model
from sifaka.results import Result, ValidationResult, ImprovementResult
from sifaka.errors import ChainError


class Validator(Protocol):
    """Protocol defining the interface for validators."""
    
    def validate(self, text: str) -> ValidationResult:
        """Validate text and return a result."""
        ...


class Improver(Protocol):
    """Protocol defining the interface for improvers."""
    
    def improve(self, text: str) -> tuple[str, ImprovementResult]:
        """Improve text and return the improved text and a result."""
        ...


class Chain:
    """Main orchestrator for the generation, validation, and improvement flow."""
    
    def __init__(self):
        """Initialize a new Chain instance."""
        self._model: Optional[Model] = None
        self._prompt: Optional[str] = None
        self._validators: List[Validator] = []
        self._improvers: List[Improver] = []
        self._options: Dict[str, Any] = {}
    
    def with_model(self, model: Union[str, Model]) -> "Chain":
        """Set the model to use for generation.
        
        Args:
            model: Either a model instance or a string in the format "provider:model_name".
            
        Returns:
            The chain instance for method chaining.
        """
        if isinstance(model, str):
            # Parse model string (e.g., "openai:gpt-4")
            provider, model_name = model.split(":", 1)
            self._model = create_model(provider, model_name)
        else:
            self._model = model
        return self
    
    def with_prompt(self, prompt: str) -> "Chain":
        """Set the prompt to use for generation.
        
        Args:
            prompt: The prompt to use for generation.
            
        Returns:
            The chain instance for method chaining.
        """
        self._prompt = prompt
        return self
    
    def validate_with(self, validator: Validator) -> "Chain":
        """Add a validator to the chain.
        
        Args:
            validator: The validator to add.
            
        Returns:
            The chain instance for method chaining.
        """
        self._validators.append(validator)
        return self
    
    def improve_with(self, improver: Improver) -> "Chain":
        """Add an improver to the chain.
        
        Args:
            improver: The improver to add.
            
        Returns:
            The chain instance for method chaining.
        """
        self._improvers.append(improver)
        return self
    
    def with_options(self, **options: Any) -> "Chain":
        """Set options for the model.
        
        Args:
            **options: Options to pass to the model.
            
        Returns:
            The chain instance for method chaining.
        """
        self._options.update(options)
        return self
    
    def run(self) -> Result:
        """Execute the chain and return the result.
        
        Returns:
            The result of the chain execution.
            
        Raises:
            ChainError: If the chain is not properly configured.
        """
        if not self._model:
            raise ChainError("Model not specified")
        if not self._prompt:
            raise ChainError("Prompt not specified")
        
        # Generate initial text
        text = self._model.generate(self._prompt, **self._options)
        
        # Validate text
        validation_results = []
        for validator in self._validators:
            result = validator.validate(text)
            validation_results.append(result)
            if not result.passed:
                return Result(
                    text=text,
                    passed=False,
                    validation_results=validation_results,
                    improvement_results=[]
                )
        
        # Improve text
        improvement_results = []
        for improver in self._improvers:
            improved_text, result = improver.improve(text)
            improvement_results.append(result)
            text = improved_text
        
        return Result(
            text=text,
            passed=True,
            validation_results=validation_results,
            improvement_results=improvement_results
        )
