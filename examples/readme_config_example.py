"""
An example from the README demonstrating configuration usage.

This example is taken directly from the README to verify that it works correctly.
"""

import os
from dotenv import load_dotenv
from sifaka import Chain
from sifaka.config import SifakaConfig, ModelConfig, ValidatorConfig, CriticConfig
from sifaka.validators import length, prohibited_content
from sifaka.critics.self_refine import create_self_refine_critic

# Load environment variables from .env file if it exists
load_dotenv()

# Get API key from environment variables
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Create a custom configuration
config = SifakaConfig(
    model=ModelConfig(
        temperature=0.8,
        max_tokens=500,
        top_p=0.9,
        api_key=api_key,  # Use environment variable for API key
    ),
    validator=ValidatorConfig(
        min_words=100,
        max_words=500,
        prohibited_content=["violence", "hate speech"]
    ),
    critic=CriticConfig(
        temperature=0.5,
        refinement_rounds=3,
        system_prompt="You are an expert editor that improves text for clarity and conciseness."
    ),
    debug=True,
    log_level="DEBUG",
)

# Create a model
from sifaka.models.openai import OpenAIModel
model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

# Use the configuration with a chain
chain = (
    Chain(config)
    .with_model(model)
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=config.validator.min_words, max_words=config.validator.max_words))
    .validate_with(prohibited_content(prohibited=config.validator.prohibited_content))
    .improve_with(create_self_refine_critic(model=model, max_refinement_iterations=config.critic.refinement_rounds))
)

# Run the chain
result = chain.run()

# Check the result
if result.passed:
    print("Chain execution succeeded!")
    print(result.text)
else:
    print("Chain execution failed validation")
    for i, validation_result in enumerate(result.validation_results):
        if not validation_result.passed:
            print(f"Validation {i+1} failed: {validation_result.message}")
