"""
Combined GuardRails and Pydantic Adapters Example

This example demonstrates how to use both GuardRails and Pydantic adapters together in Sifaka:
1. Create a GuardRails validator for email validation
2. Create a Pydantic model for structured data validation
3. Use both adapters together to validate content
4. Show how they can be combined for comprehensive validation

Requirements:
- Sifaka
- guardrails-ai (pip install guardrails-ai)
- pydantic (pip install pydantic)
- pydantic-ai (pip install pydantic-ai)
"""

import os
import re
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

# Import Sifaka components
from sifaka.rules.base import Rule, RuleResult
from sifaka.rules.formatting.length import create_length_rule

# Import GuardRails components
try:
    from guardrails.validator_base import Validator, register_validator
    from guardrails.classes import ValidationResult, PassResult, FailResult
    from sifaka.adapters.guardrails import create_guardrails_rule

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    print("⚠️ GuardRails is not installed. Please install it with 'pip install guardrails-ai'")

# Import PydanticAI components
try:
    from pydantic_ai import Agent, RunContext
    from sifaka.adapters.pydantic_ai import create_pydantic_adapter

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    print("⚠️ PydanticAI is not installed. Please install it with 'pip install pydantic-ai'")


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


# --- Pydantic Model ---

class ContactInfo(BaseModel):
    """Pydantic model for contact information."""
    name: str = Field(..., description="Full name of the contact")
    email: str = Field(..., description="Email address of the contact")
    phone: Optional[str] = Field(None, description="Phone number (optional)")
    notes: str = Field("", description="Additional notes about the contact")


# --- Standalone Rule Implementation ---

class EmailRule:
    """Simple rule that validates email format."""

    def __init__(self):
        self._name = "email_rule"
        # Simple regex for email validation
        self.pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    @property
    def name(self) -> str:
        return self._name

    def validate(self, text: str) -> RuleResult:
        """Validate that text contains a valid email."""
        # Handle empty text
        if not text or not text.strip():
            return RuleResult(passed=False, message="Text is empty")

        # Check if text contains a valid email
        if self.pattern.search(text):
            return RuleResult(passed=True, message="Text contains a valid email address")
        else:
            return RuleResult(
                passed=False, message="Text does not contain a valid email address"
            )


# --- Combined Validation Function ---

def validate_contact_info(text: str, rules: List[Rule]) -> Dict[str, Any]:
    """
    Validate text using multiple rules and extract structured data.
    
    Args:
        text: The text to validate
        rules: List of rules to validate against
        
    Returns:
        Dictionary with validation results and extracted data
    """
    # Validate with rules
    rule_results = [rule.validate(text) for rule in rules]
    all_passed = all(result.passed for result in rule_results)
    
    # Try to extract structured data
    contact_data = None
    try:
        # Simple extraction of key-value pairs (this is a naive implementation)
        lines = text.strip().split('\n')
        data = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip().lower()] = value.strip()
        
        # Try to create a ContactInfo model
        contact_data = ContactInfo(
            name=data.get('name', ''),
            email=data.get('email', ''),
            phone=data.get('phone', None),
            notes=data.get('notes', '')
        )
    except Exception as e:
        pass
    
    return {
        "passed": all_passed,
        "rule_results": rule_results,
        "structured_data": contact_data,
        "is_valid_structure": contact_data is not None
    }


# --- Example Usage ---

def main():
    """Demonstrate using GuardRails and Pydantic adapters together."""
    print("=== GuardRails and Pydantic Adapters Example ===\n")
    
    # Create rules
    rules = [
        EmailRule(),  # Our custom email rule
        create_length_rule(min_chars=20, max_chars=500)  # Length rule from Sifaka
    ]
    
    # Add GuardRails rule if available
    if GUARDRAILS_AVAILABLE:
        email_validator = EmailValidator(on_fail="exception")
        email_rule = create_guardrails_rule(
            guardrails_validator=email_validator, 
            rule_id="guardrails_email_format"
        )
        rules.append(email_rule)
        print("✓ GuardRails adapter is available and added to rules")
    else:
        print("✗ GuardRails adapter is not available")
    
    # Check if PydanticAI is available
    if PYDANTIC_AI_AVAILABLE:
        print("✓ PydanticAI adapter is available")
    else:
        print("✗ PydanticAI adapter is not available")
    
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
        
        # Validate with our combined function
        result = validate_contact_info(text, rules)
        
        print(f"\nValidation passed: {'✓' if result['passed'] else '✗'}")
        print(f"Valid structure: {'✓' if result['is_valid_structure'] else '✗'}")
        
        print("\nRule results:")
        for rule_result in result["rule_results"]:
            print(f"  - {rule_result.name}: {'✓' if rule_result.passed else '✗'} - {rule_result.message}")
        
        if result["structured_data"]:
            print("\nExtracted data:")
            data = result["structured_data"]
            print(f"  Name: {data.name}")
            print(f"  Email: {data.email}")
            print(f"  Phone: {data.phone or 'N/A'}")
            print(f"  Notes: {data.notes or 'N/A'}")
        
        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
