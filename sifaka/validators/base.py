"""Base validator classes for Sifaka.

This module provides the base validator interface and common functionality
for all validators in the new PydanticAI-based Sifaka architecture.

Key features:
- Async-first design compatible with PydanticAI
- Rich validation results with detailed feedback
- Integration with Sifaka's logging and error handling
- Support for both sync and async validation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import asyncio
import time

from sifaka.core.thought import SifakaThought
from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation.
    
    This class encapsulates the result of validating text against specific criteria.
    It provides detailed information about whether validation passed, any issues found,
    and suggestions for improvement.
    
    Attributes:
        passed: Whether the validation passed
        message: Human-readable message describing the result
        score: Numeric score (0.0 to 1.0) indicating validation quality
        issues: List of specific issues found
        suggestions: List of suggestions for improvement
        metadata: Additional metadata about the validation
        validator_name: Name of the validator that produced this result
        processing_time_ms: Time taken to perform validation in milliseconds
    """
    passed: bool
    message: str
    score: float = 0.0
    issues: List[str] = None
    suggestions: List[str] = None
    metadata: Dict[str, Any] = None
    validator_name: str = "unknown"
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.issues is None:
            self.issues = []
        if self.suggestions is None:
            self.suggestions = []
        if self.metadata is None:
            self.metadata = {}


class BaseValidator(ABC):
    """Base class for all validators in Sifaka.
    
    This abstract base class defines the interface that all validators must implement.
    It provides common functionality for validation operations and integrates with
    Sifaka's logging and error handling systems.
    
    Validators should be async-first but also support sync operations for backward
    compatibility. The base class handles the sync/async coordination.
    
    Attributes:
        name: Human-readable name of the validator
        description: Description of what the validator checks
    """
    
    def __init__(self, name: str, description: str = ""):
        """Initialize the base validator.
        
        Args:
            name: Human-readable name of the validator
            description: Description of what the validator checks
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    async def validate_async(self, thought: SifakaThought) -> ValidationResult:
        """Validate a thought asynchronously.
        
        This is the main validation method that all validators must implement.
        It should be async-first and return a detailed ValidationResult.
        
        Args:
            thought: The SifakaThought to validate
            
        Returns:
            ValidationResult with detailed validation information
            
        Raises:
            ValidationError: If validation cannot be performed
        """
        pass
    
    def validate(self, thought: SifakaThought) -> ValidationResult:
        """Validate a thought synchronously.
        
        This method provides sync compatibility by running the async validation
        in an event loop. Use validate_async() when possible for better performance.
        
        Args:
            thought: The SifakaThought to validate
            
        Returns:
            ValidationResult with detailed validation information
            
        Raises:
            ValidationError: If validation cannot be performed
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.validate_async(thought))
                    return future.result()
            else:
                # No running loop, we can use asyncio.run
                return asyncio.run(self.validate_async(thought))
        except Exception as e:
            logger.error(
                f"Sync validation failed for {self.name}",
                extra={
                    "validator": self.name,
                    "thought_id": thought.id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True
            )
            raise ValidationError(
                f"Validation failed for {self.name}: {str(e)}",
                error_code="validation_execution_error",
                context={
                    "validator": self.name,
                    "thought_id": thought.id,
                    "error_type": type(e).__name__,
                },
                suggestions=[
                    "Check validator configuration",
                    "Verify input text is valid",
                    "Check system resources",
                ]
            ) from e
    
    def create_validation_result(
        self,
        passed: bool,
        message: str,
        score: float = None,
        issues: List[str] = None,
        suggestions: List[str] = None,
        metadata: Dict[str, Any] = None,
        processing_time_ms: float = 0.0,
    ) -> ValidationResult:
        """Create a ValidationResult with consistent formatting.
        
        This helper method creates a ValidationResult with the validator's name
        and ensures consistent formatting across all validators.
        
        Args:
            passed: Whether validation passed
            message: Human-readable message
            score: Numeric score (defaults to 1.0 if passed, 0.0 if failed)
            issues: List of specific issues
            suggestions: List of suggestions
            metadata: Additional metadata
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            Properly formatted ValidationResult
        """
        if score is None:
            score = 1.0 if passed else 0.0
        
        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues or [],
            suggestions=suggestions or [],
            metadata=metadata or {},
            validator_name=self.name,
            processing_time_ms=processing_time_ms,
        )
    
    def create_empty_text_result(self) -> ValidationResult:
        """Create a result for empty or None text.
        
        Returns:
            ValidationResult indicating empty text failure
        """
        return self.create_validation_result(
            passed=False,
            message="No text available for validation",
            score=0.0,
            issues=["Text is empty or None"],
            suggestions=["Provide non-empty text for validation"],
            metadata={"reason": "empty_text"},
        )
    
    def __str__(self) -> str:
        """String representation of the validator."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the validator."""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}')"


class TextLengthMixin:
    """Mixin for validators that need text length utilities."""
    
    @staticmethod
    def get_text_length(text: str, unit: str = "characters") -> int:
        """Get text length in specified units.
        
        Args:
            text: Text to measure
            unit: Unit of measurement ("characters" or "words")
            
        Returns:
            Length in specified units
            
        Raises:
            ValueError: If unit is not supported
        """
        if unit == "characters":
            return len(text)
        elif unit == "words":
            return len(text.split())
        else:
            raise ValueError(f"Unsupported unit: {unit}. Use 'characters' or 'words'.")
    
    @staticmethod
    def format_length_message(length: int, unit: str, min_val: int = None, max_val: int = None) -> str:
        """Format a length-related message.
        
        Args:
            length: Current length
            unit: Unit of measurement
            min_val: Minimum required length
            max_val: Maximum allowed length
            
        Returns:
            Formatted message string
        """
        unit_name = unit.rstrip('s')  # Remove plural 's' if present
        
        if min_val is not None and max_val is not None:
            return f"Text has {length} {unit} (required: {min_val}-{max_val} {unit})"
        elif min_val is not None:
            return f"Text has {length} {unit} (minimum: {min_val} {unit})"
        elif max_val is not None:
            return f"Text has {length} {unit} (maximum: {max_val} {unit})"
        else:
            return f"Text has {length} {unit}"


class TimingMixin:
    """Mixin for validators that need performance timing."""
    
    def time_operation(self, operation_name: str = "validation"):
        """Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            
        Returns:
            Context manager that tracks timing
        """
        return logger.performance_timer(operation_name, validator=getattr(self, 'name', 'unknown'))
