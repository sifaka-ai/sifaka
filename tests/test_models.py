"""Test script for the OpenAI and Anthropic model implementations.

This script tests the basic functionality of the OpenAI and Anthropic model
implementations, including creating models, generating text, and counting tokens.
"""

import os
from dotenv import load_dotenv

from sifaka.models import create_model
from sifaka.core.thought import Document, Thought

# Load environment variables from .env file
load_dotenv()

def test_openai_model():
    """Test the OpenAI model implementation."""
    print("\n=== Testing OpenAI Model ===")
    
    # Create a model using the factory function
    model = create_model("openai:gpt-3.5-turbo")
    
    # Test basic generation
    prompt = "Write a one-sentence story about a robot."
    print(f"Generating text with prompt: '{prompt}'")
    response = model.generate(prompt, max_tokens=50)
    print(f"Response: {response}")
    
    # Test token counting
    text = "This is a test of the token counting functionality."
    token_count = model.count_tokens(text)
    print(f"Token count for '{text}': {token_count}")
    
    # Test generation with Thought container
    thought = Thought(
        prompt="Write a one-sentence story about a robot.",
        system_prompt="You are a creative writer.",
        pre_generation_context=[
            Document(text="Robots are machines that can be programmed to perform tasks."),
            Document(text="Asimov's Three Laws of Robotics are rules for robots in his science fiction."),
        ]
    )
    print("Generating text with Thought container")
    response = model.generate_with_thought(thought, max_tokens=50)
    print(f"Response: {response}")

def test_anthropic_model():
    """Test the Anthropic model implementation."""
    print("\n=== Testing Anthropic Model ===")
    
    # Create a model using the factory function
    model = create_model("anthropic:claude-3-haiku-20240307")
    
    # Test basic generation
    prompt = "Write a one-sentence story about a robot."
    print(f"Generating text with prompt: '{prompt}'")
    response = model.generate(prompt, max_tokens=50)
    print(f"Response: {response}")
    
    # Test token counting
    text = "This is a test of the token counting functionality."
    token_count = model.count_tokens(text)
    print(f"Token count for '{text}': {token_count}")
    
    # Test generation with Thought container
    thought = Thought(
        prompt="Write a one-sentence story about a robot.",
        system_prompt="You are a creative writer.",
        pre_generation_context=[
            Document(text="Robots are machines that can be programmed to perform tasks."),
            Document(text="Asimov's Three Laws of Robotics are rules for robots in his science fiction."),
        ]
    )
    print("Generating text with Thought container")
    response = model.generate_with_thought(thought, max_tokens=50)
    print(f"Response: {response}")

if __name__ == "__main__":
    # Check if API keys are available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if openai_api_key:
        try:
            test_openai_model()
        except Exception as e:
            print(f"Error testing OpenAI model: {e}")
    else:
        print("Skipping OpenAI model test: OPENAI_API_KEY not found in environment variables")
    
    if anthropic_api_key:
        try:
            test_anthropic_model()
        except Exception as e:
            print(f"Error testing Anthropic model: {e}")
    else:
        print("Skipping Anthropic model test: ANTHROPIC_API_KEY not found in environment variables")
