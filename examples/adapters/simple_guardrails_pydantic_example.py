"""
Simple GuardRails and Pydantic Adapters Example

This example demonstrates the basic usage of GuardRails and Pydantic adapters in Sifaka.
It focuses on the core functionality without complex dependencies.

Requirements:
- guardrails-ai (pip install guardrails-ai)
- pydantic (pip install pydantic)
"""

import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# Import GuardRails components
try:
    from guardrails.validator_base import Validator, register_validator
    from guardrails.classes import ValidationResult, PassResult, FailResult

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    print("⚠️ GuardRails is not installed. Please install it with 'pip install guardrails-ai'")

# Define a simple rule result class
class RuleResult:
    """Result of a rule validation."""
    def __init__(self, passed: bool, message: str, metadata=None):
        self.passed = passed
        self.message = message
        self.metadata = metadata or {}


# --- GuardRails Validator ---

if GUARDRAILS_AVAILABLE:
    @register_validator(name="email_validator", data_type="string")
    class EmailValidator(Validator):
        """Validator that checks if a value contains a valid email address."""

        rail_alias = "email_validator"

        def __init__(self, on_fail="exception"):
            """Initialize the validator."""
            super().__init__(on_fail=on_fail)
            # Simple regex for email validation
            self.pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

        def _validate(self, value, metadata):
            """Validate if the value contains a valid email address."""
            if self.pattern.search(value):
                return PassResult(actual_value=value, validated_value=value)
            else:
                return FailResult(
                    actual_value=value,
                    error_message="Value must contain a valid email address (e.g., user@example.com)",
                )

    # Create a simple adapter for GuardRails validators
    class GuardrailsAdapter:
        """Simple adapter for GuardRails validators."""
        
        def __init__(self, validator, name="guardrails_rule"):
            self.validator = validator
            self._name = name
            
        @property
        def name(self):
            return self._name
            
        def validate(self, text):
            """Validate text using the GuardRails validator."""
            # Handle empty text
            if not text or not text.strip():
                return RuleResult(passed=False, message="Text is empty")
                
            # Run the validator
            result = self.validator._validate(text, {})
            
            # Convert GuardRails result to RuleResult
            if isinstance(result, PassResult):
                return RuleResult(
                    passed=True,
                    message="Validation passed",
                    metadata={"validator": self.validator.rail_alias}
                )
            else:
                return RuleResult(
                    passed=False,
                    message=result.error_message,
                    metadata={"validator": self.validator.rail_alias}
                )


# --- Pydantic Model ---

class ContactInfo(BaseModel):
    """Pydantic model for contact information."""
    name: str = Field(..., description="Full name of the contact")
    email: str = Field(..., description="Email address of the contact")
    phone: Optional[str] = Field(None, description="Phone number (optional)")
    notes: str = Field("", description="Additional notes about the contact")


# --- Pydantic Adapter ---

class PydanticAdapter:
    """Simple adapter for Pydantic models."""
    
    def __init__(self, model_class, name="pydantic_rule"):
        self.model_class = model_class
        self._name = name
        
    @property
    def name(self):
        return self._name
        
    def validate(self, text):
        """Validate that text can be parsed into the Pydantic model."""
        # Handle empty text
        if not text or not text.strip():
            return RuleResult(passed=False, message="Text is empty")
            
        # Try to extract structured data
        try:
            # Simple extraction of key-value pairs
            lines = text.strip().split('\n')
            data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip().lower()] = value.strip()
            
            # Try to create a model instance
            model_instance = self.model_class(**data)
            
            return RuleResult(
                passed=True,
                message=f"Successfully parsed into {self.model_class.__name__}",
                metadata={"model": model_instance.model_dump()}
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Failed to parse into {self.model_class.__name__}: {str(e)}",
                metadata={"error": str(e)}
            )


# --- Example Usage ---

def main():
    """Demonstrate using GuardRails and Pydantic adapters."""
    print("=== Simple GuardRails and Pydantic Adapters Example ===\n")
    
    rules = []
    
    # Add GuardRails rule if available
    if GUARDRAILS_AVAILABLE:
        email_validator = EmailValidator(on_fail="exception")
        email_rule = GuardrailsAdapter(email_validator, name="email_format_rule")
        rules.append(email_rule)
        print("✓ GuardRails adapter is available and added to rules")
    else:
        print("✗ GuardRails adapter is not available")
    
    # Add Pydantic rule
    pydantic_rule = PydanticAdapter(ContactInfo, name="contact_info_rule")
    rules.append(pydantic_rule)
    print("✓ Pydantic adapter added to rules")
    
    print("\n--- Testing with different inputs ---\n")
    
    # Test texts
    test_texts = [
        """
        Name: John Doe
        Email: john.doe@example.com
        Phone: 555-123-4567
        Notes: This is a valid contact with proper email format
        """,
        
        """
        Name: Jane Smith
        Email: not-an-email
        Phone: 555-987-6543
        Notes: This contact has an invalid email format
        """,
        
        """
        Name: Short
        Email: short@example.com
        """,
        
        """
        This text doesn't have any structured data or email address.
        """
    ]
    
    # Validate each text
    for i, text in enumerate(test_texts):
        print(f"\n=== Test Case {i+1} ===")
        print(f"Input:\n{text.strip()}")
        
        print("\nRule results:")
        all_passed = True
        for rule in rules:
            result = rule.validate(text)
            all_passed = all_passed and result.passed
            print(f"  - {rule.name}: {'✓' if result.passed else '✗'} - {result.message}")
            
            # Show extracted data for Pydantic rule if available
            if rule.name == "contact_info_rule" and result.passed:
                print("    Extracted data:")
                for key, value in result.metadata["model"].items():
                    print(f"      {key}: {value or 'N/A'}")
        
        print(f"\nOverall validation: {'✓' if all_passed else '✗'}")
        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
