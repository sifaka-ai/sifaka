"""
Example demonstrating the use of the N-Critics critic.

This example shows how to use the N-Critics critic to improve text
using an ensemble of specialized critics.
"""

import os
import sys

# Add the parent directory to the path so we can import from sifaka
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sifaka.models.openai import OpenAIModel
from sifaka.critics.n_critics import NCriticsCritic


def main():
    """Run the N-Critics example."""
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key as an environment variable.")
        print("Example: export OPENAI_API_KEY='your-api-key'")
        return

    # Initialize the OpenAI model
    model = OpenAIModel(model_name="gpt-4")

    # Initialize the N-Critics critic
    critic = NCriticsCritic(
        model=model,
        num_critics=3,  # Use 3 specialized critics
        max_refinement_iterations=2,  # Maximum 2 refinement iterations
        temperature=0.7,
    )

    # Sample text to improve
    text = """
    Artificial intelligence is a technology that enables computers to perform tasks that typically require human intelligence.
    It was invented in the 1950s and has been growing rapidly in recent years.
    AI can be used for many things like image recognition, natural language processing, and decision making.
    Many companies are investing in AI research and development.
    Some people are worried about AI taking over jobs or becoming too powerful.
    """

    print("Original text:")
    print(text)
    print("\n" + "-" * 80 + "\n")

    # Improve the text using the N-Critics critic
    improved_text, result = critic.improve(text)

    print("Improved text:")
    print(improved_text)
    print("\n" + "-" * 80 + "\n")

    print("Improvement details:")
    print(f"Changes made: {result.changes_made}")
    print(f"Message: {result.message}")

    # Print the critiques from each specialized critic
    print("\nCritiques from specialized critics:")
    for i, critique in enumerate(result.details.get("critic_critiques", [])):
        print(f"\nCritic {i+1}: {critique.get('role', 'Unknown')}")
        print(f"Score: {critique.get('score', 0)}/10")
        print(f"Explanation: {critique.get('explanation', 'No explanation provided')}")
        print("Issues:")
        for issue in critique.get("issues", []):
            print(f"- {issue}")
        print("Suggestions:")
        for suggestion in critique.get("suggestions", []):
            print(f"- {suggestion}")


if __name__ == "__main__":
    main()
