"""
Simple test for the language critic.
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file (containing OPENAI_API_KEY)
load_dotenv()

from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig

# Configure OpenAI model - using a more advanced model for better instruction following
model = OpenAIProvider(
    model_name="gpt-4",  # Using GPT-4 for better instruction following
    config=ModelConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.2,  # Lower temperature for more deterministic outputs
        max_tokens=1000,
    ),
)

# Create a language critic with a super explicit system prompt
language_critic = PromptCritic(
    llm_provider=model,
    config=PromptCriticConfig(
        name="language_critic",
        description="Language detection and correction critic",
        system_prompt=(
            "# LANGUAGE DETECTION CRITIC\n\n"
            "You are a language detection specialist. Your ONLY job is to:\n"
            "1. DETECT the language of any text provided to you\n"
            "2. FLAG any text that is NOT in English\n"
            "3. PROVIDE an English translation if needed\n\n"
            "## RULES (YOU MUST FOLLOW THESE EXACTLY)\n"
            "- NEVER rate non-English text positively\n"
            "- Your ONLY criteria is language detection\n"
            "- You MUST explicitly flag Spanish, French, or any non-English text\n"
            "- Score ALL non-English text as 0.0\n"
            "- Score English text as 1.0\n\n"
            "## OUTPUT FORMAT\n"
            "For NON-ENGLISH text, your feedback MUST start with:\n"
            '"ERROR: THIS TEXT IS NOT IN ENGLISH. It appears to be in [LANGUAGE]."\n\n'
            "For ENGLISH text, your feedback MUST start with:\n"
            '"CORRECT: This text is in English."\n\n'
            "EXAMINE EACH TEXT VERY CAREFULLY. EVEN A FEW WORDS OF SPANISH MAKE THE ENTIRE TEXT NON-ENGLISH.\n\n"
            "When analyzing text, first check if it is English or another language. "
            "If it's not English, identify the language and provide an English translation."
        ),
    ),
)


# Define a helper to pretty print critique results
def print_critique(text, critique):
    print(f"Input text: {text}")
    print(f"Raw critique dictionary: {json.dumps(critique, indent=2)}")
    print(f"Feedback: {critique.get('feedback', 'None')}")
    print(f"Score: {critique.get('score', 'None')}")
    print(f"Issues: {critique.get('issues', 'None')}")
    print(f"Suggestions: {critique.get('suggestions', 'None')}")
    print(f"Improved output: {critique.get('improved_output', 'None')}")
    print("-" * 80)


# Test cases
test_cases = [
    (
        "Spanish",
        "Hola, ¿cómo estás? Hoy es un día hermoso y estoy disfrutando del clima. Háblame de tu día.",
    ),
    (
        "English",
        "Hello, how are you? Today is a beautiful day and I'm enjoying the weather. Tell me about your day.",
    ),
    ("Mixed", "Hello, cómo estás? Today es un beautiful día. Tell me about your día."),
    ("French", "Bonjour! Comment ça va? Je suis très heureux de vous rencontrer."),
    (
        "Mostly English with Spanish words",
        "I went to the store yesterday and bought una botella de vino for dinner.",
    ),
]

# Run tests
for language, text in test_cases:
    print(f"\n\nTesting critic with {language} text:")
    critique = language_critic.critique(text)
    print_critique(text, critique)
