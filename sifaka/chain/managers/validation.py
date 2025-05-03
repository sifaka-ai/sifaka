"""
Validation manager module for Sifaka.

This module provides the ValidationManager class which is responsible for
validating outputs against rules.
"""

from typing import Generic, List, TypeVar

from ...rules import Rule
from ...validation import ValidationResult, Validator
from ...utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class ValidationManager(Generic[OutputType]):
    """
    Manages validation for chains.
    
    This class is responsible for validating outputs against rules and
    managing rule-related functionality.
    """
    
    def __init__(self, rules: List[Rule]):
        """
        Initialize a ValidationManager instance.
        
        Args:
            rules: The rules to validate against
        """
        self._rules = rules
        self._validator = Validator[OutputType](rules)
        
    @property
    def rules(self) -> List[Rule]:
        """
        Get the rules.
        
        Returns:
            The rules
        """
        return self._rules
        
    @property
    def validator(self) -> Validator[OutputType]:
        """
        Get the validator.
        
        Returns:
            The validator
        """
        return self._validator
        
    def validate(self, output: OutputType) -> ValidationResult[OutputType]:
        """
        Validate the output against rules.
        
        Args:
            output: The output to validate
            
        Returns:
            The validation result
        """
        return self._validator.validate(output)
        
    def get_error_messages(self, validation_result: ValidationResult[OutputType]) -> List[str]:
        """
        Get error messages from a validation result.
        
        Args:
            validation_result: The validation result
            
        Returns:
            The error messages
        """
        return self._validator.get_error_messages(validation_result)
