"""Validation component of the Sifaka engine."""

from typing import List

from ..models import SifakaResult, ValidationResult
from ..interfaces import Validator


class ValidationRunner:
    """Handles running validators and collecting results."""
    
    async def run_validators(
        self,
        text: str,
        result: SifakaResult,
        validators: List[Validator]
    ) -> bool:
        """Run all validators on the text.
        
        Args:
            text: Text to validate
            result: Result object to add validations to
            validators: List of validators to run
            
        Returns:
            True if all validators pass
        """
        if not validators:
            return True
        
        all_passed = True
        
        for validator in validators:
            try:
                validation_result = await validator.validate(text, result)
                
                # Add to result
                result.add_validation(
                    validator=validation_result.validator,
                    passed=validation_result.passed,
                    score=validation_result.score,
                    details=validation_result.details
                )
                
                if not validation_result.passed:
                    all_passed = False
                    
            except Exception as e:
                # Create error validation
                result.add_validation(
                    validator=getattr(validator, 'name', 'unknown'),
                    passed=False,
                    score=0.0,
                    details=f"Validation error: {str(e)}"
                )
                all_passed = False
        
        return all_passed
    
    def check_memory_bounds(self, result: SifakaResult, max_elements: int = 1000) -> None:
        """Ensure result doesn't exceed memory bounds.
        
        Args:
            result: Result to check
            max_elements: Maximum elements to keep
        """
        # Trim old generations if needed
        if len(result.generations) > max_elements:
            # Convert to list, slice, and recreate deque
            from collections import deque
            result.generations = deque(list(result.generations)[-max_elements:], maxlen=max_elements)
        
        # Trim old critiques if needed  
        if len(result.critiques) > max_elements:
            from collections import deque
            result.critiques = deque(list(result.critiques)[-max_elements:], maxlen=max_elements)
        
        # Trim old validations if needed
        if len(result.validations) > max_elements:
            from collections import deque
            result.validations = deque(list(result.validations)[-max_elements:], maxlen=max_elements)