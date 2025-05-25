#!/usr/bin/env python3
"""Basic Chain Demo with Mock Model.

This example demonstrates:
- Basic chain setup and execution
- Mock model for reliable testing
- Simple validator and critic configuration
- Core Sifaka functionality without external dependencies

This is the simplest possible Sifaka example, perfect for getting started
and understanding the basic chain workflow.
"""

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.models.base import MockModel
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)


def main():
    """Run the basic chain demo with mock model."""
    
    logger.info("Creating basic chain demo with mock model")
    
    # Create mock model with predefined responses
    model = MockModel(
        name="Demo Model",
        responses=[
            "Artificial intelligence is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence.",
            "Artificial intelligence is a rapidly evolving field of computer science focused on creating intelligent systems that can perform complex tasks typically requiring human intelligence, such as learning, reasoning, and problem-solving.",
            "Artificial intelligence represents a transformative field of computer science dedicated to creating sophisticated intelligent systems that can perform complex cognitive tasks typically requiring human intelligence, including learning from experience, reasoning through problems, and adapting to new situations."
        ]
    )
    
    # Create a simple length validator
    length_validator = LengthValidator(
        min_length=50,   # At least 50 characters
        max_length=500,  # At most 500 characters
        name="Basic Length Validator"
    )
    
    # Create a Reflexion critic for improvement
    critic = ReflexionCritic(
        model=model,
        reflection_depth=1,
        name="Basic Reflexion Critic"
    )
    
    # Create the chain
    chain = Chain(
        model=model,
        prompt="Explain what artificial intelligence is in simple terms.",
        max_improvement_iterations=2,
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True
    )
    
    # Add validator and critic
    chain.validate_with(length_validator)
    chain.improve_with(critic)
    
    # Run the chain
    logger.info("Running basic chain...")
    result = chain.run()
    
    # Display results
    print("\n" + "="*60)
    print("BASIC CHAIN DEMO WITH MOCK MODEL")
    print("="*60)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 40)
    print(result.text)
    
    print(f"\nChain Execution Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Model: Mock (no API required)")
    
    # Show validation results
    if result.validation_results:
        print(f"\nValidation Results:")
        for validation_result in result.validation_results:
            status = "✓ PASSED" if validation_result.is_valid else "✗ FAILED"
            print(f"  {validation_result.validator_name}: {status}")
            if validation_result.is_valid:
                print(f"    Length: {len(result.text)} characters (within 50-500 range)")
    
    # Show critic feedback
    if result.critic_feedback:
        print(f"\nCritic Feedback:")
        for feedback in result.critic_feedback:
            print(f"  {feedback.critic_name}:")
            print(f"    Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"    Suggestions: {feedback.suggestions[:150]}...")
    
    # Show thought evolution
    print(f"\nThought Evolution ({len(result.history)} iterations):")
    for i, historical_thought in enumerate(result.history, 1):
        print(f"  Iteration {i}: {len(historical_thought.text)} characters")
        print(f"    Preview: {historical_thought.text[:60]}...")
    
    print(f"\nKey Features Demonstrated:")
    print(f"  ✓ Chain creation and configuration")
    print(f"  ✓ Model integration (mock)")
    print(f"  ✓ Validation with length constraints")
    print(f"  ✓ Improvement through criticism")
    print(f"  ✓ Iterative refinement process")
    print(f"  ✓ Thought history tracking")
    
    print("\n" + "="*60)
    logger.info("Basic chain demo completed successfully")


if __name__ == "__main__":
    main()
