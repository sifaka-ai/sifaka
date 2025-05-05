"""
Example of using the SelfRefineCritic.

This example demonstrates how to use the SelfRefineCritic to iteratively
improve text through self-critique and revision.
"""

import os
from sifaka.critics import create_self_refine_critic
from sifaka.models.openai import create_openai_provider
from sifaka.models.anthropic import create_anthropic_provider


def main():
    """Run the example."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_api_key and not anthropic_api_key:
        print(
            "Error: No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables."
        )
        return

    # Create a language model provider
    if openai_api_key:
        provider = create_openai_provider(
            model_name="gpt-4", api_key=openai_api_key, temperature=0.7, max_tokens=1000
        )
        print("Using OpenAI GPT-4 model")
    else:
        provider = create_anthropic_provider(
            model_name="claude-3-opus-20240229",
            api_key=anthropic_api_key,
            temperature=0.7,
            max_tokens=1000,
        )
        print("Using Anthropic Claude model")

    # Create a self-refine critic
    critic = create_self_refine_critic(
        llm_provider=provider,
        name="example_critic",
        description="A critic for improving explanations",
        max_iterations=3,
        system_prompt="You are an expert at explaining complex concepts clearly and concisely.",
    )

    # Define a task and initial output
    task = "Explain quantum computing to a high school student."
    initial_output = "Quantum computing uses qubits instead of regular bits. Qubits can be in multiple states at once due to superposition."

    print("\nTask:")
    print(task)
    print("\nInitial Output:")
    print(initial_output)

    # Improve the output using the critic
    improved_output = critic.improve(initial_output, {"task": task})

    print("\nImproved Output:")
    print(improved_output)

    # Get critique of the improved output
    critique = critic.critique(improved_output, {"task": task})

    print("\nCritique of Improved Output:")
    print(f"Score: {critique['score']}")
    print(f"Feedback: {critique['feedback']}")

    if critique["issues"]:
        print("\nIssues:")
        for issue in critique["issues"]:
            print(f"- {issue}")

    if critique["suggestions"]:
        print("\nSuggestions:")
        for suggestion in critique["suggestions"]:
            print(f"- {suggestion}")


if __name__ == "__main__":
    main()
