"""Base validator class with common functionality."""

from abc import ABC, abstractmethod
from typing import Optional, Union
from pydantic import BaseModel, Field

from ..core.interfaces import Validator
from ..core.models import ValidationResult, SifakaResult


class ValidatorConfig(BaseModel):
    """Configuration for validator behavior."""
    
    # Score calculation
    min_score: float = Field(0.0, ge=0.0, le=1.0)
    max_score: float = Field(1.0, ge=0.0, le=1.0)
    pass_threshold: float = Field(0.7, ge=0.0, le=1.0)
    
    # Behavior flags
    strict_mode: bool = Field(False, description="Fail on any violation")
    detailed_feedback: bool = Field(True, description="Provide detailed feedback")


class BaseValidator(Validator, ABC):
    """Base class for validators with common functionality."""
    
    def __init__(self, config: Optional[ValidatorConfig] = None):
        """Initialize validator with configuration."""
        self.config = config or ValidatorConfig()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the validator's name."""
        pass
    
    @abstractmethod
    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> tuple[bool, float, str]:
        """
        Perform the actual validation logic.
        
        Returns:
            tuple: (passed, score, details)
        """
        pass
    
    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate text with standardized result creation."""
        try:
            # Perform validation
            passed, score, details = await self._perform_validation(text, result)
            
            # Apply configuration rules
            if self.config.strict_mode and score < 1.0:
                passed = False
            
            # Ensure score is in valid range
            score = max(self.config.min_score, min(score, self.config.max_score))
            
            # Apply pass threshold
            if score < self.config.pass_threshold:
                passed = False
            
            # Format details based on config
            if not self.config.detailed_feedback:
                details = "Validation " + ("passed" if passed else "failed")
            
            return ValidationResult(
                validator=self.name,
                passed=passed,
                score=score,
                details=details
            )
            
        except Exception as e:
            # Error handling with validation result
            return ValidationResult(
                validator=self.name,
                passed=False,
                score=0.0,
                details=f"Validation error: {str(e)}"
            )
    
    def _calculate_score(
        self,
        value: Union[int, float],
        target: Union[int, float],
        tolerance: float = 0.1
    ) -> float:
        """
        Calculate a score based on how close a value is to target.
        
        Args:
            value: The actual value
            target: The target value
            tolerance: Acceptable deviation (0.1 = 10%)
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        if target == 0:
            return 1.0 if value == 0 else 0.0
        
        deviation = abs(value - target) / target
        if deviation <= tolerance:
            # Linear scoring within tolerance
            score = 1.0 - (deviation / tolerance)
        else:
            # Exponential decay beyond tolerance
            score = max(0.0, 1.0 - deviation)
        
        return score
    
    def _format_details(
        self,
        primary: str,
        secondary: Optional[str] = None,
        suggestions: Optional[list[str]] = None
    ) -> str:
        """
        Format validation details consistently.
        
        Args:
            primary: Primary validation message
            secondary: Optional secondary information
            suggestions: Optional list of suggestions
            
        Returns:
            str: Formatted details string
        """
        parts = [primary]
        
        if secondary:
            parts.append(secondary)
        
        if suggestions:
            parts.append("Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                parts.append(f"  {i}. {suggestion}")
        
        return "\n".join(parts)