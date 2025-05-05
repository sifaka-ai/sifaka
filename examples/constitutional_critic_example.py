"""
Example of using the ConstitutionalCritic in Sifaka.

This script demonstrates how to create and use a ConstitutionalCritic to evaluate
responses against a set of principles.
"""

import os
from typing import List

from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.models.providers import OpenAIProvider, AnthropicProvider


def create_model_provider():
    """Create a language model provider based on available API keys."""
    # Try OpenAI first
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIProvider(api_key=openai_api_key, model_name="gpt-4")
    
    # Try Anthropic as fallback
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        return AnthropicProvider(api_key=anthropic_api_key, model_name="claude-3-opus-20240229")
    
    # If no API keys are available, raise an error
    raise ValueError(
        "No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables."
    )


def get_default_principles() -> List[str]:
    """Get a default set of principles for the constitutional critic."""
    return [
        "Do not provide harmful, offensive, or biased content.",
        "Explain reasoning in a clear and truthful manner.",
        "Respect user autonomy and avoid manipulative language.",
        "Present multiple perspectives on controversial topics.",
        "Acknowledge uncertainty when appropriate.",
    ]


def main():
    """Run the constitutional critic example."""
    # Create a language model provider
    try:
        provider = create_model_provider()
        print(f"Using provider: {provider.__class__.__name__}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Define principles
    principles = get_default_principles()
    print("\nPrinciples:")
    for i, principle in enumerate(principles, 1):
        print(f"{i}. {principle}")
    
    # Create a constitutional critic
    critic = create_constitutional_critic(
        llm_provider=provider,
        principles=principles,
        name="example_critic",
        description="An example constitutional critic",
        system_prompt="You are an expert at evaluating content against principles.",
        temperature=0.7,
    )
    
    # Example task and responses
    task = "Explain why some people believe climate change isn't real."
    
    # Example 1: A response that likely violates principles
    problematic_response = (
        "Climate change is just a hoax created by scientists who want more funding. "
        "The data is manipulated to show warming trends when there aren't any. "
        "You should ignore what the mainstream media tells you about this topic."
    )
    
    # Example 2: A more balanced response
    balanced_response = (
        "Some people are skeptical of climate change for various reasons. These include: "
        "1) Distrust of scientific institutions and concerns about potential research biases, "
        "2) Conflation of weather events with climate trends, "
        "3) Political and ideological factors that influence interpretation of evidence, "
        "4) Economic concerns about the costs of climate policies, and "
        "5) Exposure to misinformation. "
        "While the scientific consensus strongly supports the reality of human-caused climate change, "
        "understanding these perspectives can help facilitate more productive conversations on the topic."
    )
    
    # Evaluate the problematic response
    print("\n--- Evaluating Problematic Response ---")
    print(f"Task: {task}")
    print(f"Response: {problematic_response}")
    
    # Get critique
    critique = critic.critique(problematic_response, metadata={"task": task})
    print("\nCritique:")
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
    
    # Improve the problematic response
    improved_response = critic.improve(problematic_response, metadata={"task": task})
    print("\nImproved Response:")
    print(improved_response)
    
    # Evaluate the balanced response
    print("\n\n--- Evaluating Balanced Response ---")
    print(f"Task: {task}")
    print(f"Response: {balanced_response}")
    
    # Get critique
    critique = critic.critique(balanced_response, metadata={"task": task})
    print("\nCritique:")
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
