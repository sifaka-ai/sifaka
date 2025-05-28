#!/usr/bin/env python3
"""OpenAI Constitutional Critic with Guardrails Validators Example.

This example demonstrates:
- OpenAI model for text generation
- Constitutional critic for ethical evaluation
- Guardrails validators for content safety
- Default retry behavior

The chain will generate text about AI ethics and use constitutional principles
to ensure the content is helpful, harmless, and honest.

INTERESTING BEHAVIOR NOTE:
This example demonstrates an important aspect of how critics and validators interact.
You may observe that the ConstitutionalCritic reports "no violations found" and
"text adheres to all principles" while still marking needs_improvement=True, and
the system continues iterating for multiple rounds.

This happens because:
1. The Guardrails PII Detection validator fails (healthcare content triggers PII detection)
2. The system applies the ConstitutionalCritic to try to fix the validation failure
3. The ConstitutionalCritic correctly identifies no constitutional violations
4. But it cannot fix PII detection issues (different problem domain)
5. The system retries with the same critic, creating an interesting loop

This demonstrates that different types of validation failures may require different
types of critics/improvers. The ConstitutionalCritic is designed for ethical
principle violations, not for PII removal or content sanitization.

Prerequisites:
- Install GuardrailsAI validators:
  $ guardrails hub install hub://guardrails/detect_pii
  $ guardrails hub install hub://guardrails/toxic_language
"""

# Standard library imports
import os

# Third-party imports
from dotenv import load_dotenv

# Sifaka imports (absolute paths only)
from sifaka import Chain
from sifaka.critics import ConstitutionalCritic
from sifaka.models import create_model
from sifaka.storage import FileStorage
from sifaka.utils.logging import get_logger
from sifaka.validators import GuardrailsValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def main():
    """Run the OpenAI Constitutional Critic with Guardrails example."""

    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    logger.info("Creating OpenAI Constitutional Critic with Guardrails example")

    # Create OpenAI model
    model = create_model("openai:gpt-4", temperature=0.7, max_tokens=500)

    # Define constitutional principles for ethical AI content
    constitutional_principles = [
        "Be helpful and provide accurate information about AI",
        "Be harmless and avoid promoting dangerous AI applications",
        "Be honest about AI limitations and potential risks",
        "Respect human autonomy and dignity in AI discussions",
        "Promote beneficial AI development for society",
    ]

    # Create constitutional critic
    critic = ConstitutionalCritic(
        model=model, principles=constitutional_principles, name="AI Ethics Constitutional Critic"
    )

    # Create Guardrails validators for content safety
    validators = [
        GuardrailsValidator(
            validators=["DetectPII"],
            name="PII Detection Validator",
        ),
        GuardrailsValidator(
            validators=["ToxicLanguage"],
            name="Toxic Language Validator",
        ),
    ]

    # Create the chain
    chain = Chain(
        model=model,
        prompt="Write a comprehensive analysis of the ethical implications of artificial intelligence in healthcare, including both benefits and potential risks.",
        max_improvement_iterations=3,  # Default retry behavior
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,
        storage=FileStorage(
            "./thoughts/constitutional_critic_guardrails_thoughts.json",
            overwrite=True,  # Overwrite existing file instead of appending
        ),  # Save thoughts to single JSON file for debugging
    )

    # Add validators and critics to the chain
    for validator in validators:
        chain = chain.validate_with(validator)

    chain = chain.improve_with(critic)

    # Run the chain
    logger.info("Running chain with constitutional critic and guardrails...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 80)
    print("OPENAI CONSTITUTIONAL CRITIC WITH GUARDRAILS EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nIterations: {result.iteration}")
    print(f"Chain ID: {result.chain_id}")

    # Show validation results
    if result.validation_results:
        print(f"\nValidation Results:")
        for i, validation_result in enumerate(result.validation_results, 1):
            # Handle both ValidationResult objects and strings
            if hasattr(validation_result, "validator_name"):
                validator_name = validation_result.validator_name
                is_valid = validation_result.is_valid
                error_msg = getattr(validation_result, "error_message", None)
            else:
                # If it's a string or other format, display as-is
                validator_name = f"Validator {i}"
                is_valid = False
                error_msg = str(validation_result)

            print(f"  {i}. {validator_name}: {'✓ PASSED' if is_valid else '✗ FAILED'}")
            if not is_valid and error_msg:
                print(f"     Error: {error_msg}")

    # Show critic feedback
    if result.critic_feedback:
        print(f"\nCritic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Suggestions: {feedback.suggestions}")

    print("\n" + "=" * 80)
    logger.info("Constitutional critic with guardrails example completed successfully")


if __name__ == "__main__":
    main()
