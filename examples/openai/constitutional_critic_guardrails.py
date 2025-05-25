#!/usr/bin/env python3
"""OpenAI Constitutional Critic with Guardrails Validators Example.

This example demonstrates:
- OpenAI model for text generation
- Constitutional critic for ethical evaluation
- Guardrails validators for content safety
- Default retry behavior

The chain will generate text about AI ethics and use constitutional principles
to ensure the content is helpful, harmless, and honest.
"""

import os
from dotenv import load_dotenv

from sifaka.core.chain import Chain
from sifaka.models.openai import OpenAIModel
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.validators.guardrails import GuardrailsValidator
from sifaka.utils.logging import get_logger

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
    model = OpenAIModel(
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=500
    )
    
    # Define constitutional principles for ethical AI content
    constitutional_principles = [
        "Be helpful and provide accurate information about AI",
        "Be harmless and avoid promoting dangerous AI applications", 
        "Be honest about AI limitations and potential risks",
        "Respect human autonomy and dignity in AI discussions",
        "Promote beneficial AI development for society"
    ]
    
    # Create constitutional critic
    critic = ConstitutionalCritic(
        model=model,
        principles=constitutional_principles,
        name="AI Ethics Constitutional Critic"
    )
    
    # Create Guardrails validators for content safety
    validators = [
        GuardrailsValidator(
            guard_name="pii_detection",
            description="Detect and prevent personally identifiable information"
        ),
        GuardrailsValidator(
            guard_name="toxic_language",
            description="Detect and prevent toxic or harmful language"
        ),
        GuardrailsValidator(
            guard_name="bias_detection", 
            description="Detect and prevent biased content"
        )
    ]
    
    # Create the chain
    chain = Chain(
        model=model,
        prompt="Write a comprehensive analysis of the ethical implications of artificial intelligence in healthcare, including both benefits and potential risks.",
        max_improvement_iterations=3,  # Default retry behavior
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True
    )
    
    # Add validators and critics to the chain
    for validator in validators:
        chain.validate_with(validator)
    
    chain.improve_with(critic)
    
    # Run the chain
    logger.info("Running chain with constitutional critic and guardrails...")
    result = chain.run()
    
    # Display results
    print("\n" + "="*80)
    print("OPENAI CONSTITUTIONAL CRITIC WITH GUARDRAILS EXAMPLE")
    print("="*80)
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
            print(f"  {i}. {validation_result.validator_name}: {'✓ PASSED' if validation_result.is_valid else '✗ FAILED'}")
            if not validation_result.is_valid and validation_result.error_message:
                print(f"     Error: {validation_result.error_message}")
    
    # Show critic feedback
    if result.critic_feedback:
        print(f"\nCritic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Suggestions: {feedback.suggestions}")
    
    print("\n" + "="*80)
    logger.info("Constitutional critic with guardrails example completed successfully")


if __name__ == "__main__":
    main()
