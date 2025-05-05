"""
LAC (LLM-Based Actor-Critic) Critic Demo

This example demonstrates how to use the LAC critic in Sifaka, which combines
language feedback and value scoring to improve language model-based decision making.

Based on: Language Feedback Improves Language Model-based Decision Making
https://arxiv.org/abs/2403.03692
"""

import os
import sys

from sifaka.critics import create_lac_critic
from sifaka.models.openai import create_openai_provider
from sifaka.models.anthropic import create_anthropic_provider
from sifaka.models.config import ModelConfig


def main():
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_api_key and not anthropic_api_key:
        print(
            "Error: No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables."
        )
        return

    # Create a language model provider
    provider = create_openai_provider(
        model_name="gpt-3.5-turbo", api_key=openai_api_key, temperature=0.7, max_tokens=1000
    )

    # Create a LAC critic
    critic = create_lac_critic(
        llm_provider=provider,
        name="demo_lac_critic",
        description="A demo LAC critic",
        system_prompt="You are an expert at evaluating and improving text.",
        temperature=0.7,
        max_tokens=1000,
    )

    # Define a task and generate an initial response
    task = "Summarize the causes of World War I in 3 bullet points."
    print(f"\nTask: {task}")

    # Generate an initial response
    initial_response = provider.generate(f"Task: {task}")
    print(f"\nInitial Response:\n{initial_response}")

    # Use the critic to critique the response
    critique_result = critic.critique(initial_response, {"task": task})
    print(f"\nFeedback:\n{critique_result['feedback']}")
    print(f"\nValue Score: {critique_result['value']:.2f}")

    # Use the critic to improve the response
    improved_response = critic.improve(initial_response, {"task": task})
    print(f"\nImproved Response:\n{improved_response}")

    # Critique the improved response
    improved_critique = critic.critique(improved_response, {"task": task})
    print(f"\nImproved Response Feedback:\n{improved_critique['feedback']}")
    print(f"\nImproved Response Value Score: {improved_critique['value']:.2f}")

    # Try with Anthropic if available
    if anthropic_api_key:
        print("\n\n--- Using Anthropic Model ---\n")
        anthropic_provider = create_anthropic_provider(
            model_name="claude-3-haiku-20240307",
            api_key=anthropic_api_key,
            temperature=0.7,
            max_tokens=1000,
        )

        # Create a LAC critic with Anthropic
        anthropic_critic = create_lac_critic(
            llm_provider=anthropic_provider,
            name="anthropic_lac_critic",
            description="A LAC critic using Anthropic",
            system_prompt="You are an expert at evaluating and improving text.",
            temperature=0.7,
            max_tokens=1000,
        )

        # Generate a response with Anthropic
        anthropic_response = anthropic_provider.generate(f"Task: {task}")
        print(f"\nAnthropic Response:\n{anthropic_response}")

        # Use the critic to critique the response
        anthropic_critique = anthropic_critic.critique(anthropic_response, {"task": task})
        print(f"\nFeedback:\n{anthropic_critique['feedback']}")
        print(f"\nValue Score: {anthropic_critique['value']:.2f}")

        # Use the critic to improve the response
        anthropic_improved = anthropic_critic.improve(anthropic_response, {"task": task})
        print(f"\nImproved Anthropic Response:\n{anthropic_improved}")


if __name__ == "__main__":
    main()
