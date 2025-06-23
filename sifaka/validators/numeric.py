"""Numeric range validator for Sifaka."""

import re
from typing import Optional, List, Tuple

from ..core.interfaces import Validator
from ..core.models import ValidationResult, SifakaResult


class NumericRangeValidator(Validator):
    """Validates numeric values in text are within specified ranges."""
    
    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allowed_ranges: Optional[List[Tuple[float, float]]] = None,
        forbidden_ranges: Optional[List[Tuple[float, float]]] = None,
        check_percentages: bool = True,
        check_currency: bool = True,
    ):
        """Initialize numeric range validator.
        
        Args:
            min_value: Minimum allowed numeric value
            max_value: Maximum allowed numeric value
            allowed_ranges: List of (min, max) tuples for allowed ranges
            forbidden_ranges: List of (min, max) tuples for forbidden ranges
            check_percentages: Whether to validate percentage values
            check_currency: Whether to validate currency amounts
        """
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_ranges = allowed_ranges or []
        self.forbidden_ranges = forbidden_ranges or []
        self.check_percentages = check_percentages
        self.check_currency = check_currency
        
        # Numeric patterns
        self.number_pattern = re.compile(r'-?\d+\.?\d*')
        self.percentage_pattern = re.compile(r'(\d+\.?\d*)\s*%')
        self.currency_pattern = re.compile(r'\$\s*(\d+\.?\d*)')
    
    @property
    def name(self) -> str:
        return "numeric_range_validator"
    
    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate numeric values in the text."""
        issues = []
        all_numbers = []
        
        # Find all numbers
        plain_numbers = self.number_pattern.findall(text)
        for num_str in plain_numbers:
            try:
                num = float(num_str)
                all_numbers.append((num, num_str, "number"))
            except ValueError:
                pass
        
        # Find percentages
        if self.check_percentages:
            percentages = self.percentage_pattern.findall(text)
            for pct_str in percentages:
                try:
                    pct = float(pct_str)
                    all_numbers.append((pct, f"{pct_str}%", "percentage"))
                    
                    # Check percentage bounds
                    if pct < 0 or pct > 100:
                        issues.append(f"Invalid percentage: {pct}%")
                except ValueError:
                    pass
        
        # Find currency amounts
        if self.check_currency:
            amounts = self.currency_pattern.findall(text)
            for amt_str in amounts:
                try:
                    amt = float(amt_str)
                    all_numbers.append((amt, f"${amt_str}", "currency"))
                    
                    # Check for negative currency
                    if amt < 0:
                        issues.append(f"Negative currency amount: ${amt}")
                except ValueError:
                    pass
        
        # Validate each number
        for value, display, num_type in all_numbers:
            # Skip percentage validation for general ranges
            if num_type == "percentage":
                continue
                
            # Check min/max
            if self.min_value is not None and value < self.min_value:
                issues.append(f"Value {display} is below minimum {self.min_value}")
            
            if self.max_value is not None and value > self.max_value:
                issues.append(f"Value {display} is above maximum {self.max_value}")
            
            # Check allowed ranges
            if self.allowed_ranges:
                in_allowed_range = any(
                    min_val <= value <= max_val 
                    for min_val, max_val in self.allowed_ranges
                )
                if not in_allowed_range:
                    issues.append(f"Value {display} is outside allowed ranges")
            
            # Check forbidden ranges
            for min_val, max_val in self.forbidden_ranges:
                if min_val <= value <= max_val:
                    issues.append(f"Value {display} is in forbidden range [{min_val}, {max_val}]")
        
        # Build result
        if issues:
            return ValidationResult(
                validator=self.name,
                passed=False,
                score=0.0,
                details="; ".join(issues[:3])  # Limit to first 3 issues
            )
        
        # Calculate score
        if not all_numbers:
            details = "No numeric values found"
            score = 1.0
        else:
            details = f"Validated {len(all_numbers)} numeric value(s)"
            score = 1.0
        
        return ValidationResult(
            validator=self.name,
            passed=True,
            score=score,
            details=details
        )


# Convenience factory functions

def create_percentage_validator() -> NumericRangeValidator:
    """Create a validator for percentage values (0-100)."""
    return NumericRangeValidator(
        check_percentages=True,
        check_currency=False,
    )


def create_price_validator(max_price: float = 10000.0) -> NumericRangeValidator:
    """Create a validator for reasonable prices."""
    return NumericRangeValidator(
        min_value=0,
        max_value=max_price,
        check_currency=True,
        check_percentages=False,
    )


def create_age_validator() -> NumericRangeValidator:
    """Create a validator for human ages."""
    return NumericRangeValidator(
        min_value=0,
        max_value=150,
        forbidden_ranges=[(200, float('inf'))],  # No unrealistic ages
    )