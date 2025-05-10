"""
Chain Adapters Module

This module provides adapter classes for integrating existing Sifaka components
with the new chain system. These adapters implement the new interfaces while
delegating to the existing components.

## Adapter Classes
1. **ModelAdapter**: Adapts existing model providers to the Model interface
2. **ValidatorAdapter**: Adapts existing rules to the Validator interface
3. **ImproverAdapter**: Adapts existing critics to the Improver interface
4. **FormatterAdapter**: Adapts existing formatters to the Formatter interface

## Usage Examples
```python
from sifaka.chain.v2.adapters import ModelAdapter, ValidatorAdapter, ImproverAdapter
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model_provider = OpenAIProvider("gpt-3.5-turbo")
rule = create_length_rule(min_chars=10, max_chars=1000)
critic = create_prompt_critic(
    llm_provider=model_provider,
    system_prompt="You are an expert editor that improves text."
)

# Create adapters
model = ModelAdapter(model_provider)
validator = ValidatorAdapter(rule)
improver = ImproverAdapter(critic)

# Use adapters
output = model.generate("Write a short story")
validation_result = validator.validate(output)
if not validation_result.passed:
    improved_output = improver.improve(output, [validation_result])
```
"""

from typing import Any, Dict, List, Optional
import asyncio

from .interfaces import Model, Validator, Improver, Formatter, ValidationResult
from .errors import ModelError, ValidationError, ImproverError, FormatterError, safely_execute


class ModelAdapter(Model):
    """Adapter for existing model providers."""
    
    def __init__(self, model: Any):
        """
        Initialize the model adapter.
        
        Args:
            model: The model provider to adapt
        """
        self._model = model
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            
        Returns:
            The generated text
            
        Raises:
            ModelError: If text generation fails
        """
        def generate_operation():
            # Check for different model provider interfaces
            if hasattr(self._model, "invoke"):
                return self._model.invoke(prompt)
            elif hasattr(self._model, "generate"):
                return self._model.generate(prompt)
            elif hasattr(self._model, "run"):
                return self._model.run(prompt)
            elif hasattr(self._model, "process"):
                return self._model.process(prompt)
            else:
                raise ModelError(f"Unsupported model provider: {type(self._model).__name__}")
        
        return safely_execute(
            operation=generate_operation,
            component_name="model_adapter",
            component_type="Model",
            error_class=ModelError,
        )
    
    async def generate_async(self, prompt: str) -> str:
        """
        Generate text asynchronously.
        
        Args:
            prompt: The prompt to generate text from
            
        Returns:
            The generated text
            
        Raises:
            ModelError: If text generation fails
        """
        # Check if model has async methods
        if hasattr(self._model, "invoke_async"):
            return await self._model.invoke_async(prompt)
        elif hasattr(self._model, "generate_async"):
            return await self._model.generate_async(prompt)
        elif hasattr(self._model, "run_async"):
            return await self._model.run_async(prompt)
        elif hasattr(self._model, "process_async"):
            return await self._model.process_async(prompt)
        
        # Fall back to running synchronous method in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)


class ValidatorAdapter(Validator):
    """Adapter for existing rules."""
    
    def __init__(self, validator: Any):
        """
        Initialize the validator adapter.
        
        Args:
            validator: The rule to adapt
        """
        self._validator = validator
    
    def validate(self, output: str) -> ValidationResult:
        """
        Validate an output.
        
        Args:
            output: The output to validate
            
        Returns:
            The validation result
            
        Raises:
            ValidationError: If validation fails
        """
        def validate_operation():
            # Check for different rule interfaces
            if hasattr(self._validator, "validate"):
                result = self._validator.validate(output)
            elif hasattr(self._validator, "process"):
                result = self._validator.process(output)
            elif hasattr(self._validator, "run"):
                result = self._validator.run(output)
            else:
                raise ValidationError(f"Unsupported validator: {type(self._validator).__name__}")
            
            # Convert result to ValidationResult
            return self._convert_result(result)
        
        return safely_execute(
            operation=validate_operation,
            component_name="validator_adapter",
            component_type="Validator",
            error_class=ValidationError,
        )
    
    async def validate_async(self, output: str) -> ValidationResult:
        """
        Validate an output asynchronously.
        
        Args:
            output: The output to validate
            
        Returns:
            The validation result
            
        Raises:
            ValidationError: If validation fails
        """
        # Check if validator has async methods
        if hasattr(self._validator, "validate_async"):
            result = await self._validator.validate_async(output)
            return self._convert_result(result)
        elif hasattr(self._validator, "process_async"):
            result = await self._validator.process_async(output)
            return self._convert_result(result)
        elif hasattr(self._validator, "run_async"):
            result = await self._validator.run_async(output)
            return self._convert_result(result)
        
        # Fall back to running synchronous method in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.validate, output)
    
    def _convert_result(self, result: Any) -> ValidationResult:
        """
        Convert a rule result to a ValidationResult.
        
        Args:
            result: The rule result to convert
            
        Returns:
            The converted ValidationResult
        """
        # If already a ValidationResult, return as is
        if isinstance(result, ValidationResult):
            return result
        
        # Extract fields from result
        passed = getattr(result, "passed", False)
        message = getattr(result, "message", "")
        score = getattr(result, "score", 0.0)
        issues = getattr(result, "issues", [])
        suggestions = getattr(result, "suggestions", [])
        metadata = getattr(result, "metadata", {})
        
        # Create ValidationResult
        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata=metadata,
        )


class ImproverAdapter(Improver):
    """Adapter for existing critics."""
    
    def __init__(self, improver: Any):
        """
        Initialize the improver adapter.
        
        Args:
            improver: The critic to adapt
        """
        self._improver = improver
    
    def improve(
        self, 
        output: str, 
        validation_results: List[ValidationResult]
    ) -> str:
        """
        Improve an output based on validation results.
        
        Args:
            output: The output to improve
            validation_results: The validation results to use for improvement
            
        Returns:
            The improved output
            
        Raises:
            ImproverError: If improvement fails
        """
        def improve_operation():
            # Check for different critic interfaces
            if hasattr(self._improver, "improve"):
                return self._improver.improve(output, validation_results)
            elif hasattr(self._improver, "refine"):
                return self._improver.refine(output, validation_results)
            elif hasattr(self._improver, "process"):
                return self._improver.process(output, validation_results)
            elif hasattr(self._improver, "run"):
                return self._improver.run(output, validation_results)
            else:
                raise ImproverError(f"Unsupported improver: {type(self._improver).__name__}")
        
        return safely_execute(
            operation=improve_operation,
            component_name="improver_adapter",
            component_type="Improver",
            error_class=ImproverError,
        )
    
    async def improve_async(
        self, 
        output: str, 
        validation_results: List[ValidationResult]
    ) -> str:
        """
        Improve an output asynchronously.
        
        Args:
            output: The output to improve
            validation_results: The validation results to use for improvement
            
        Returns:
            The improved output
            
        Raises:
            ImproverError: If improvement fails
        """
        # Check if improver has async methods
        if hasattr(self._improver, "improve_async"):
            return await self._improver.improve_async(output, validation_results)
        elif hasattr(self._improver, "refine_async"):
            return await self._improver.refine_async(output, validation_results)
        elif hasattr(self._improver, "process_async"):
            return await self._improver.process_async(output, validation_results)
        elif hasattr(self._improver, "run_async"):
            return await self._improver.run_async(output, validation_results)
        
        # Fall back to running synchronous method in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.improve, output, validation_results)


class FormatterAdapter(Formatter):
    """Adapter for existing formatters."""
    
    def __init__(self, formatter: Any):
        """
        Initialize the formatter adapter.
        
        Args:
            formatter: The formatter to adapt
        """
        self._formatter = formatter
    
    def format(
        self, 
        output: str, 
        validation_results: List[ValidationResult]
    ) -> Any:
        """
        Format a result.
        
        Args:
            output: The output to format
            validation_results: The validation results to include
            
        Returns:
            The formatted result
            
        Raises:
            FormatterError: If formatting fails
        """
        def format_operation():
            # Check for different formatter interfaces
            if hasattr(self._formatter, "format"):
                return self._formatter.format(output, validation_results)
            elif hasattr(self._formatter, "format_result"):
                return self._formatter.format_result(output, validation_results)
            elif hasattr(self._formatter, "process"):
                return self._formatter.process(output, validation_results)
            elif hasattr(self._formatter, "run"):
                return self._formatter.run(output, validation_results)
            else:
                raise FormatterError(f"Unsupported formatter: {type(self._formatter).__name__}")
        
        return safely_execute(
            operation=format_operation,
            component_name="formatter_adapter",
            component_type="Formatter",
            error_class=FormatterError,
        )
    
    async def format_async(
        self, 
        output: str, 
        validation_results: List[ValidationResult]
    ) -> Any:
        """
        Format a result asynchronously.
        
        Args:
            output: The output to format
            validation_results: The validation results to include
            
        Returns:
            The formatted result
            
        Raises:
            FormatterError: If formatting fails
        """
        # Check if formatter has async methods
        if hasattr(self._formatter, "format_async"):
            return await self._formatter.format_async(output, validation_results)
        elif hasattr(self._formatter, "format_result_async"):
            return await self._formatter.format_result_async(output, validation_results)
        elif hasattr(self._formatter, "process_async"):
            return await self._formatter.process_async(output, validation_results)
        elif hasattr(self._formatter, "run_async"):
            return await self._formatter.run_async(output, validation_results)
        
        # Fall back to running synchronous method in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.format, output, validation_results)
