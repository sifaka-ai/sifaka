"""
Advanced GuardRails and Pydantic Adapters Example

This example demonstrates a more realistic use case combining GuardRails and Pydantic adapters:
1. Use a language model to generate content
2. Validate the content with GuardRails rules
3. Extract structured data with Pydantic models
4. Refine the content if validation fails

Requirements:
- guardrails-ai (pip install guardrails-ai)
- pydantic (pip install pydantic)
- pydantic-ai (pip install pydantic-ai)
- An OpenAI API key (set as OPENAI_API_KEY environment variable)
"""

import os
import re
import json
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("⚠️ OPENAI_API_KEY environment variable not set.")
    print("This example requires an OpenAI API key to run.")
    print("Please set it in your environment or in a .env file.")
    exit(1)

# Import GuardRails components
try:
    from guardrails.validator_base import Validator, register_validator
    from guardrails.classes import ValidationResult, PassResult, FailResult

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    print("⚠️ GuardRails is not installed. Please install it with 'pip install guardrails-ai'")
    exit(1)

# Import OpenAI
try:
    import openai
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI is not installed. Please install it with 'pip install openai'")
    exit(1)

# Define a simple rule result class
class RuleResult:
    """Result of a rule validation."""
    def __init__(self, passed: bool, message: str, metadata=None):
        self.passed = passed
        self.message = message
        self.metadata = metadata or {}


# --- GuardRails Validator ---

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

@register_validator(name="phone_validator", data_type="string")
class PhoneValidator(Validator):
    """Validator that checks if a value contains a valid phone number."""

    rail_alias = "phone_validator"

    def __init__(self, on_fail="exception"):
        """Initialize the validator."""
        super().__init__(on_fail=on_fail)
        # Simple regex for US phone number validation
        self.pattern = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

    def _validate(self, value, metadata):
        """Validate if the value contains a valid phone number."""
        if self.pattern.search(value):
            return PassResult(actual_value=value, validated_value=value)
        else:
            return FailResult(
                actual_value=value,
                error_message="Value must contain a valid phone number (e.g., 555-123-4567)",
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
    company: Optional[str] = Field(None, description="Company name (optional)")
    position: Optional[str] = Field(None, description="Job position (optional)")
    notes: str = Field("", description="Additional notes about the contact")


class ContactResponse(BaseModel):
    """Pydantic model for the complete response."""
    contact: ContactInfo = Field(..., description="Contact information")
    summary: str = Field(..., description="Summary of the contact information")


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


# --- OpenAI Integration ---

class SimpleLanguageModel:
    """Simple wrapper for OpenAI API."""
    
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7, max_tokens=500):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        
    def generate(self, prompt):
        """Generate text using the OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text: {e}")
            return None
            
    def refine(self, original_text, feedback):
        """Refine text based on feedback."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful editor."},
                    {"role": "user", "content": f"Original text:\n\n{original_text}\n\nFeedback:\n{feedback}\n\nPlease revise the text based on the feedback."}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error refining text: {e}")
            return original_text


# --- Main Example Function ---

def run_example():
    """Run the advanced example with GuardRails and Pydantic adapters."""
    print("=== Advanced GuardRails and Pydantic Adapters Example ===\n")

    # Create rules
    email_validator = EmailValidator(on_fail="exception")
    phone_validator = PhoneValidator(on_fail="exception")
    
    email_rule = GuardrailsAdapter(email_validator, name="email_format_rule")
    phone_rule = GuardrailsAdapter(phone_validator, name="phone_format_rule")
    contact_rule = PydanticAdapter(ContactInfo, name="contact_info_rule")
    
    rules = [email_rule, phone_rule, contact_rule]
    print("✓ Created GuardRails and Pydantic rules")
    
    # Create language model
    model = SimpleLanguageModel(model="gpt-3.5-turbo", temperature=0.7, max_tokens=500)
    print("✓ Created language model")
    
    # Prompt designed to generate contact information
    prompt = """
    Generate a fictional contact card for a technology professional.
    Include their name, email address, phone number, company, position, and a brief note.
    Format the information clearly with labels (Name:, Email:, etc.).
    """

    print("\n--- Generating and validating contact information ---\n")
    print(f"Prompt: {prompt.strip()}")
    
    # Generate initial text
    generated_text = model.generate(prompt)
    if not generated_text:
        print("❌ Failed to generate text")
        return
        
    print(f"\n✅ Generated text:\n{generated_text}")
    
    # Validate the generated text
    print("\n--- Validating generated text ---\n")
    
    all_passed = True
    feedback = []
    
    for rule in rules:
        result = rule.validate(generated_text)
        all_passed = all_passed and result.passed
        
        print(f"  - {rule.name}: {'✓' if result.passed else '✗'} - {result.message}")
        
        # Show extracted data for Pydantic rule if available
        if rule.name == "contact_info_rule" and result.passed:
            print("    Extracted data:")
            for key, value in result.metadata["model"].items():
                print(f"      {key}: {value or 'N/A'}")
                
        # Collect feedback for failed rules
        if not result.passed:
            feedback.append(f"- {rule.name}: {result.message}")
    
    print(f"\nOverall validation: {'✓' if all_passed else '✗'}")
    
    # Refine text if validation failed
    if not all_passed:
        print("\n--- Refining text based on feedback ---\n")
        feedback_text = "Please fix the following issues:\n" + "\n".join(feedback)
        print(f"Feedback:\n{feedback_text}")
        
        refined_text = model.refine(generated_text, feedback_text)
        if not refined_text:
            print("❌ Failed to refine text")
            return
            
        print(f"\n✅ Refined text:\n{refined_text}")
        
        # Validate the refined text
        print("\n--- Validating refined text ---\n")
        
        all_passed = True
        for rule in rules:
            result = rule.validate(refined_text)
            all_passed = all_passed and result.passed
            
            print(f"  - {rule.name}: {'✓' if result.passed else '✗'} - {result.message}")
            
            # Show extracted data for Pydantic rule if available
            if rule.name == "contact_info_rule" and result.passed:
                print("    Extracted data:")
                for key, value in result.metadata["model"].items():
                    print(f"      {key}: {value or 'N/A'}")
        
        print(f"\nOverall validation after refinement: {'✓' if all_passed else '✗'}")
        
        # Extract structured data if validation passed
        if all_passed and contact_rule.validate(refined_text).passed:
            try:
                # Extract data from the refined text
                lines = refined_text.strip().split('\n')
                data = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        data[key.strip().lower()] = value.strip()
                
                # Create ContactInfo model
                contact_info = ContactInfo(
                    name=data.get('name', ''),
                    email=data.get('email', ''),
                    phone=data.get('phone', None),
                    company=data.get('company', None),
                    position=data.get('position', None),
                    notes=data.get('notes', '')
                )
                
                # Create a summary
                summary = f"{contact_info.name} works at {contact_info.company or 'an unknown company'} as a {contact_info.position or 'professional'}."
                
                # Create the full response
                response = ContactResponse(contact=contact_info, summary=summary)
                
                print("\n📋 Final structured data:")
                print(json.dumps(response.model_dump(), indent=2))
                
            except Exception as e:
                print(f"\n❌ Error extracting structured data: {e}")


if __name__ == "__main__":
    run_example()
