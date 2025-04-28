"""
Language Correction Example

This example demonstrates a system where:
1. An OpenAI chatbot starts responding in Spanish
2. A critic detects the wrong language using a language classifier
3. The critic provides feedback to correct the language to English
4. The system continues the conversation in English
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (containing OPENAI_API_KEY)
load_dotenv()

from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.classifiers.language import LanguageClassifier, ClassifierConfig
from sifaka.rules.adapters.classifier import create_classifier_rule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import Chain

# Configure OpenAI model
model = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    config=ModelConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=1000,
    ),
)

# Create a language classifier using the factory method
language_classifier = LanguageClassifier.create(
    name="language_detector",
    description="Detects the language of text",
    labels=["en", "es", "fr", "de"],  # Focus on a few languages
    params={
        "min_confidence": 0.6,
        "fallback_lang": "en",
        "fallback_confidence": 0.0,
    },
)

# Create a language rule that requires English
language_rule = create_classifier_rule(
    classifier=language_classifier,
    name="english_only_rule",
    description="Ensures text is in English",
    threshold=0.7,  # Confidence threshold
    valid_labels=["en"],  # Only English is valid
)

# Create a critic to help improve responses that aren't in English
language_critic = PromptCritic(
    llm_provider=model,
    config=PromptCriticConfig(
        name="language_critic",
        description="Helps correct non-English responses to English",
        system_prompt=(
            "You are a language critic that helps correct non-English text to English. "
            "Your job is to identify when text is not in English and provide guidance "
            "on how to express the same content in natural, fluent English. "
            "Be specific about what was wrong with the original text and how to fix it."
        ),
    ),
)

# Create a chain with the OpenAI model, language rule, and language critic
chain = Chain(
    model=model,
    rules=[language_rule],
    critic=language_critic,
    max_attempts=3,
)


# Function to simulate a conversation starting in Spanish
def run_conversation_example():
    """Run an example conversation that starts in Spanish but corrects to English."""

    # Start with a prompt that will likely generate Spanish
    initial_prompt = "Responde en español: ¿Cómo estás hoy? Háblame de tu día."

    try:
        # The first attempt should generate a response in Spanish, which will fail validation
        # The critic should provide feedback, and the chain will retry until English is used
        result = chain.run(initial_prompt)

        print("Final validated response (in English):")
        print(result.output)
        print("\nCritique details:")
        print(result.critique_details)

        # Continuing the conversation in English
        follow_up_prompt = "Tell me more about what you can do to help me."
        follow_up_result = chain.run(follow_up_prompt)

        print("\nFollow-up response (should remain in English):")
        print(follow_up_result.output)

    except ValueError as e:
        print(f"Chain validation failed: {e}")


if __name__ == "__main__":
    run_conversation_example()
