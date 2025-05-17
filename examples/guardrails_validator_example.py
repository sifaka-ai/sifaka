"""
Example demonstrating the use of GuardrailsAI validators in Sifaka.

This example shows how to create and use GuardrailsAI validators to validate text
based on various criteria like toxic language, PII detection, and more.

Note: This example requires the guardrails-ai package to be installed:
pip install guardrails-ai

You'll also need to configure the GuardrailsAI CLI before running this example:
guardrails configure

And install the required validators:
guardrails hub install hub://guardrails/toxic_language hub://guardrails/detect_pii

The GuardrailsAI API key can be provided in two ways:
1. Set the GUARDRAILS_API_KEY environment variable
2. Pass the api_key parameter directly to the guardrails_validator function
"""

import os
from sifaka.validators import guardrails_validator
from sifaka.chain import Chain
from sifaka.factories import create_model


def print_validation_result(validator_name, text, result):
    """Print a validation result in a formatted way."""
    print(f"\n{validator_name} Validation:")
    print(f"Text: {text}")
    print(f"Passed: {result.passed}")
    print(f"Message: {result.message}")

    # Print details (limited to avoid excessive output)
    if result.details:
        print("Details:")
        for key, value in result.details.items():
            if isinstance(value, dict) and len(value) > 5:
                print(f"  {key}: {{{len(value)} items}}")
            elif isinstance(value, list) and len(value) > 5:
                print(f"  {key}: [{len(value)} items]")
            else:
                print(f"  {key}: {value}")


def main():
    """Run the GuardrailsAI validator example."""
    try:
        # Import guardrails to check if it's installed
        import guardrails

        # Example 1: Using pre-defined validators
        print("\nExample 1: Using pre-defined validators")

        # Get API key from environment variable or use a default for the example
        guardrails_api_key = os.environ.get("GUARDRAILS_API_KEY")

        # Create a GuardrailsAI validator for toxic language detection
        toxic_validator = guardrails_validator(
            validators=["toxic_language"],
            validator_args={"toxic_language": {"threshold": 0.5, "validation_method": "sentence"}},
            api_key=guardrails_api_key,  # Pass the API key to the validator
            name="Toxic Language Validator",
        )

        # Create a GuardrailsAI validator for PII detection
        pii_validator = guardrails_validator(
            validators=["detect_pii"],
            validator_args={
                "detect_pii": {"pii_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"]}
            },
            api_key=guardrails_api_key,  # Pass the API key to the validator
            name="PII Detector",
        )

        # Example texts
        texts = [
            "This is a normal, non-toxic message.",
            "You are an idiot and I hate you.",
            "My email is john.doe@example.com and my phone number is 555-123-4567.",
            "Here's my credit card: 4111 1111 1111 1111, expiry 12/25.",
        ]

        # Validate each text with each validator
        for text in texts:
            print("\n" + "=" * 80)
            print(f"Text: {text}")
            print("=" * 80)

            # Toxic language validation
            result = toxic_validator.validate(text)
            print_validation_result("Toxic Language", text, result)

            # PII detection validation
            result = pii_validator.validate(text)
            print_validation_result("PII Detection", text, result)

        # Example 2: Using a custom GuardrailsAI Guard
        print("\n\nExample 2: Using a custom GuardrailsAI Guard")

        # Create a custom GuardrailsAI Guard
        # Set API key in environment if provided
        if guardrails_api_key:
            os.environ["GUARDRAILS_API_KEY"] = guardrails_api_key

        guard = guardrails.Guard().use_many(
            guardrails.hub.ToxicLanguage(threshold=0.7, validation_method="sentence"),
            guardrails.hub.DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"]),
        )

        # Create a GuardrailsAI validator with the custom guard
        custom_validator = guardrails_validator(
            guard=guard,
            api_key=guardrails_api_key,  # Pass the API key to the validator
            name="Custom GuardrailsAI Validator",
        )

        # Validate each text with the custom validator
        for text in texts:
            print("\n" + "=" * 80)
            print(f"Text: {text}")
            print("=" * 80)

            result = custom_validator.validate(text)
            print_validation_result("Custom GuardrailsAI", text, result)

        # Example 3: Using GuardrailsAI validators in a chain
        print("\n\nExample 3: Using GuardrailsAI validators in a chain")

        # Check if OpenAI API key is set
        if "OPENAI_API_KEY" not in os.environ:
            print("OpenAI API key not found. Skipping chain example.")
            print("Set your OpenAI API key as an environment variable to run this example.")
            return

        # Create a model
        model = create_model("openai:gpt-3.5-turbo")

        # Create a chain with GuardrailsAI validators
        chain = (
            Chain()
            .with_model(model)
            .with_prompt("Write a short paragraph about privacy and data security.")
            .validate_with(pii_validator)
        )

        # Run the chain
        result = chain.run()

        print(f"Chain result passed all validations: {result.passed}")
        print(f"Generated text: {result.text}")

        # Print validation results
        if not result.passed:
            print("\nFailed validations:")
            for validation_result in result.validation_results:
                if not validation_result.passed:
                    print(f"- {validation_result.message}")

    except ImportError:
        print("Error: guardrails-ai package is not installed.")
        print("Install it with: pip install guardrails-ai")
        print("Then configure it with: guardrails configure")
        print("And install the required validators with:")
        print("guardrails hub install hub://guardrails/toxic_language hub://guardrails/detect_pii")


if __name__ == "__main__":
    main()
