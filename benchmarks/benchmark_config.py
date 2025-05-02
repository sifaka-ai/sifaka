"""
Configuration for Sifaka benchmarks.
"""

from guardrails.validator_base import Validator, register_validator
from guardrails.classes import ValidationResult, PassResult, FailResult
import re

@register_validator(name="benchmark_validator", data_type="string")
class BenchmarkValidator(Validator):
    """Validator that checks if text meets basic benchmark requirements."""

    rail_alias = "benchmark_validator"

    def __init__(self, on_fail="exception"):
        """Initialize the validator."""
        super().__init__(on_fail=on_fail)
        self.min_length = 50
        self.max_length = 500

    def _validate(self, value, metadata):
        """Validate if the text meets length requirements."""
        length = len(value.split())
        if self.min_length <= length <= self.max_length:
            return PassResult(actual_value=value, validated_value=value)
        else:
            return FailResult(
                actual_value=value,
                error_message=f"Text must be between {self.min_length} and {self.max_length} words",
            )

# Create an instance of our validator
benchmark_validator = BenchmarkValidator(on_fail="exception")

# Guardrails configuration
GUARDRAILS_CONFIG = {
    "validators": [benchmark_validator],
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000,
}