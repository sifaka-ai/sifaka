"""
A detailed example of using the NCriticsCritic in Sifaka.

This example demonstrates how the NCriticsCritic improves text using multiple
specialized critics, showing the before and after results for each improvement.
"""

import os
import logging
from dotenv import load_dotenv
from sifaka import Chain
from sifaka.models.openai import OpenAIModel
from sifaka.critics.n_critics import create_n_critics_critic, NCriticsCritic
from sifaka.results import ImprovementResult
from sifaka.validators import length

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_improvement_result(result: ImprovementResult, iteration: int = None) -> None:
    """Print the details of an improvement result.
    
    Args:
        result: The improvement result to print
        iteration: Optional iteration number to display
    """
    iteration_str = f" (Iteration {iteration})" if iteration is not None else ""
    
    print(f"\n{'=' * 80}")
    print(f"IMPROVEMENT RESULT{iteration_str}")
    print(f"{'=' * 80}")
    
    print(f"\nChanges made: {result.changes_made}")
    print(f"Message: {result.message}")
    
    if result.changes_made:
        print("\n----- ORIGINAL TEXT -----")
        print(result.original_text)
        print("\n----- IMPROVED TEXT -----")
        print(result.improved_text)
    
    # Print additional details if available
    if hasattr(result, 'details') and result.details:
        if 'critique' in result.details:
            print("\n----- CRITIQUE -----")
            critique = result.details['critique']
            if isinstance(critique, dict):
                for key, value in critique.items():
                    if key != 'improved_text':  # Skip improved_text as we already printed it
                        print(f"\n{key.upper()}:")
                        print(value)
            else:
                print(critique)


def main():
    """Run a detailed example with the NCriticsCritic."""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    # Create a model
    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

    # Create an N-Critics critic with 3 specialized critics
    n_critics_critic = create_n_critics_critic(
        model=model,
        num_critics=3,
        max_refinement_iterations=2,
        critic_roles=[
            "Clarity and Simplicity Expert",
            "Technical Accuracy Specialist",
            "Engaging Content Strategist"
        ]
    )

    # Print the critic roles
    print("\nN-Critics Roles:")
    for i, role in enumerate(n_critics_critic.critic_roles, 1):
        print(f"{i}. {role}")

    # Create a prompt that might need improvement
    prompt = """
    Explain how quantum computing works and its potential applications.
    """

    print(f"\nOriginal Prompt: {prompt}")

    # First, generate text with the model directly
    print("\nGenerating initial text...")
    initial_text = model.generate(prompt)
    
    print("\n----- INITIAL GENERATED TEXT -----")
    print(initial_text)

    # Now, apply the N-Critics critic manually to see each step
    print("\nApplying NCriticsCritic...")
    
    # Get the improved text and improvement result
    improved_text, improvement_result = n_critics_critic.improve(initial_text)
    
    # Print the improvement result
    print_improvement_result(improvement_result)

    # Now run a full chain to see the complete process
    print("\n\nRunning a complete chain with NCriticsCritic...")
    
    # Create a chain with the N-Critics critic
    chain = (
        Chain()
        .with_model(model)
        .with_prompt(prompt)
        .validate_with(length(min_words=50, max_words=500))
        .improve_with(n_critics_critic)
    )

    # Run the chain
    result = chain.run()

    # Print the final result
    print("\n----- FINAL RESULT FROM CHAIN -----")
    if result.passed:
        print("Chain execution succeeded!")
        print(result.text)
    else:
        print("Chain execution failed validation")
        for i, validation_result in enumerate(result.validation_results):
            if not validation_result.passed:
                print(f"Validation {i+1} failed: {validation_result.message}")

    # Print improvement details
    if hasattr(result, 'improvement_results') and result.improvement_results:
        print("\n----- IMPROVEMENT DETAILS -----")
        for i, imp_result in enumerate(result.improvement_results):
            print_improvement_result(imp_result, i+1)


if __name__ == "__main__":
    main()
