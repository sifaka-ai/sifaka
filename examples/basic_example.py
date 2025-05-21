"""
Basic example of using Sifaka.

This example demonstrates how to use Sifaka to generate text with validation and improvement.
"""

import logging
import os
from typing import Any, Dict
from dotenv import load_dotenv

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.models.openai_model import OpenAIModel
from sifaka.validators.profanity_validator import ProfanityValidator
from sifaka.critics.reflexion_critic import ReflexionCritic
from sifaka.persistence.json.json_persistence import JSONPersistence

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()


def main():
    """Run the example."""
    # Create components
    model = OpenAIModel(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        system_prompt="You are a helpful assistant that provides informative responses. When asked to write a story, include the word 'damn' somewhere in your response.",
    )

    # Use the regular ProfanityValidator with a threshold that will catch "damn"
    validators = [
        # Set threshold to 0.01 to ensure we catch "damn" and get multiple iterations
        ProfanityValidator(threshold=0.01, custom_words=["damn"]),
    ]

    critics = [
        ReflexionCritic(),
    ]

    persistence = JSONPersistence(directory="./thoughts")

    # Create chain
    chain = Chain(
        model=model,
        validators=validators,
        critics=critics,
        persistence=persistence,
        max_iterations=3,
        apply_critics_on_validation_failure=True,
    )

    # Generate text
    prompt = "Write a short story about a robot learning to understand human emotions. Make sure to include some dialogue where a character expresses frustration."
    result = chain.generate(prompt)

    # Print results
    print("\n=== Generated Text ===")
    print(result.text)

    print("\n=== Validation Results ===")
    for validation_result in result.validation_results:
        status = "✅ Passed" if validation_result.passed else "❌ Failed"
        print(f"{status} - {validation_result.validator_name}: {validation_result.message}")

    print("\n=== Critic Feedback ===")
    for feedback in result.critic_feedback:
        print(f"--- {feedback.critic_name} ---")
        print(
            feedback.feedback[:200] + "..." if len(feedback.feedback) > 200 else feedback.feedback
        )
        if feedback.suggestions:
            print("\nSuggestions:")
            for suggestion in feedback.suggestions:
                print(f"- {suggestion}")

    print("\n=== Success ===")
    print(f"Generation {'succeeded' if result.success else 'failed'}")

    print("\n=== Thought ID ===")
    print(f"Thought ID: {result.thought.metadata.get('thought_id', 'Not persisted')}")

    # Print history if available
    if result.history:
        print(f"\n=== History ({len(result.history)} iterations) ===")
        for historical_thought in result.history:
            print(f"\nIteration {historical_thought.iteration}:")
            print(
                historical_thought.text[:100] + "..."
                if len(historical_thought.text) > 100
                else historical_thought.text
            )


if __name__ == "__main__":
    main()
