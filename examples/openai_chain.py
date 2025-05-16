"""
Example of using the Sifaka Chain API with OpenAI and Reflexion Critic.

This example demonstrates how to create a chain with an OpenAI model and use
the Reflexion critic for text improvement.

The Reflexion approach is based on the paper:
"Reflexion: Language Agents with Verbal Reinforcement Learning"
(Shinn et al., 2023, https://arxiv.org/abs/2303.11366)

Reflexion involves a multi-step process where the model reflects on its own
output and iteratively improves it through self-critique.
"""

import sys
import os
import argparse

# Add the project root to the path so we can import sifaka
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sifaka.chain import Chain
from sifaka.validators import length
from sifaka.critics.reflexion import create_reflexion_critic
from sifaka.factories import create_model, create_model_from_string


def main():
    """Run the example with Reflexion critic."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run a Sifaka chain with OpenAI and Reflexion critic"
    )
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    parser.add_argument(
        "--prompt", default="Write a short story about a robot.", help="Prompt to use"
    )
    parser.add_argument(
        "--system-message",
        default="You are a creative writing assistant that excels at storytelling.",
        help="System message to use",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument(
        "--reflection-rounds", type=int, default=2, help="Number of reflection rounds"
    )
    args = parser.parse_args()

    try:
        # Create a model instance for the critic
        model_name = f"openai:{args.model}"
        model = create_model_from_string(model_name)

        # Create a Reflexion critic
        reflexion_critic = create_reflexion_critic(
            model=model,
            reflection_rounds=args.reflection_rounds,
            system_prompt=(
                "You are an expert editor who specializes in self-reflection and improvement. "
                "Your goal is to reflect on text and iteratively improve it through self-critique."
            ),
            temperature=args.temperature,
        )

        # Create a chain with an OpenAI model
        chain = Chain()

        # Configure the chain
        chain.with_model(model_name)
        chain.with_prompt(args.prompt)

        # Use the built-in length validator
        chain.validate_with(length(min_words=10, max_words=1000))

        # Use the Reflexion critic for improvement
        chain.improve_with(reflexion_critic)

        chain.with_options(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_message=args.system_message,
        )

        # Run the chain
        print(f"\nRunning chain with {args.reflection_rounds} reflection rounds...")
        result = chain.run()

        # Print the result
        print("\nResult:")
        print(f"Passed: {result.passed}")

        print("\nGenerated Text:")
        print("=" * 40)
        print(result.text)
        print("=" * 40)

        print("\nValidation Results:")
        for i, validation_result in enumerate(result.validation_results):
            print(f"  {i+1}. Passed: {validation_result.passed}")
            print(f"     Message: {validation_result.message}")

        print("\nImprovement Results:")
        for i, improvement_result in enumerate(result.improvement_results):
            print(f"  {i+1}. Changes Made: {improvement_result.changes_made}")
            print(f"     Message: {improvement_result.message}")

            # Print reflection details if available
            if improvement_result.details and "reflection_history" in improvement_result.details:
                reflection_history = improvement_result.details["reflection_history"]
                print(f"     Reflection Rounds: {len(reflection_history)}")
                for j, reflection in enumerate(reflection_history):
                    print(
                        f"       Round {j+1}: {reflection.get('reflection', 'No reflection available')[:100]}..."
                    )

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This example requires the OpenAI package to be installed.")
        print("Install it with: pip install openai tiktoken")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
