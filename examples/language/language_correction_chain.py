"""
Language Correction Example

This example demonstrates a system where:
1. An OpenAI chatbot starts responding in Spanish
2. A rule using the LanguageClassifier correctly detects non-English text
3. A custom language critic provides feedback to correct the language to English
4. The system continues the conversation in English
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (containing OPENAI_API_KEY)
load_dotenv()

from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.classifiers.language import LanguageClassifier, ClassifierConfig, ClassificationResult
from sifaka.adapters.rules.classifier import create_classifier_rule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import ChainCore, ChainResult


# Create a concrete implementation of LanguageClassifier
class ConcreteLanguageClassifier(LanguageClassifier):
    """A concrete implementation of LanguageClassifier that implements the required abstract method."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Delegate to the existing implementation in the parent class."""
        return self._classify_impl(text)


# Custom prompt factory for language detection
class LanguagePromptFactory:
    """Custom factory specifically for language detection and correction prompts."""

    def create_critique_prompt(self, text: str) -> str:
        """
        Create a language detection and correction prompt.

        Args:
            text: The text to analyze for language

        Returns:
            str: The language detection prompt
        """
        return f"""IMPORTANT: Your ONLY task is to detect the language of this text:

TEXT TO ANALYZE:
{text}

First, determine if this text is in English or another language.

If it IS in English, respond EXACTLY in this format:
SCORE: 1.0
FEEDBACK: CORRECT: This text is in English.
ISSUES: []
SUGGESTIONS: []

If it is NOT in English, respond EXACTLY in this format:
SCORE: 0.0
FEEDBACK: ERROR: This text is NOT in English. It appears to be in [LANGUAGE NAME].
ISSUES: ["Text is in [LANGUAGE] instead of English"]
SUGGESTIONS: ["Respond in English only", "Translate the text to English"]

ANALYZE LANGUAGE:"""

    # Required by the PromptCritic interface
    def create_validation_prompt(self, text: str) -> str:
        """Create validation prompt (not used for language detection)."""
        return self.create_critique_prompt(text)

    # Required by the PromptCritic interface
    def create_improvement_prompt(self, text: str, feedback: str) -> str:
        """Create improvement prompt for translating non-English text."""
        return f"""TASK: Translate the following non-English text to English.

ORIGINAL TEXT:
{text}

IMPORTANT INSTRUCTIONS:
1. Translate the above text to proper English
2. Maintain the same meaning and tone
3. Your output should ONLY be the English translation with NO additional explanation

Format your response with EXACTLY this format:
IMPROVED_TEXT: [Your English translation here]

YOUR TRANSLATION:"""


# Configure main model for generating responses
generator_model = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    config=ModelConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=1000,
    ),
)

# Configure critic model
critic_model = OpenAIProvider(
    model_name="gpt-4",  # Using GPT-4 for better instruction following
    config=ModelConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.2,  # Lower temperature for more deterministic outputs
        max_tokens=1000,
    ),
)

# Create a language classifier using the factory method
language_classifier = ConcreteLanguageClassifier.create(
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

# Create a critic that ONLY focuses on language detection and correction
language_critic = PromptCritic(
    llm_provider=critic_model,  # Using GPT-4 for the critic
    prompt_factory=LanguagePromptFactory(),  # Using our custom prompt factory
    config=PromptCriticConfig(
        name="language_critic",
        description="Language detection and correction critic",
        system_prompt=(
            "# LANGUAGE DETECTOR\n\n"
            "You are a language detection specialist. Your ONLY job is to detect non-English text "
            "and provide clear instructions to use English instead.\n\n"
            "- Score 0.0 for any non-English text\n"
            "- Score 1.0 for English text only\n"
            "- Provide an English translation for non-English text\n"
            "- Be direct and explicit about language requirements"
        ),
    ),
)


class DebugChain:
    """Chain implementation for debugging purposes."""

    def __init__(
        self,
        model,
        language_classifier,
        language_critic,
        max_attempts: int = 2,
        verbose: bool = True,
    ):
        """Initialize debug chain."""
        self.model = model
        self.language_classifier = language_classifier
        self.language_critic = language_critic
        self.max_attempts = max_attempts
        self.verbose = verbose
        # Store the prompt factory for direct access
        self.prompt_factory = LanguagePromptFactory()

    def run(self, prompt: str) -> str:
        """Run the chain."""
        attempt = 0
        current_prompt = prompt

        while attempt < self.max_attempts:
            attempt += 1
            print(f"\n[Step {attempt}] Detecting language...")

            # Generate output
            model_output = self.model.generate(current_prompt)
            print(
                f"Output: {model_output[:100]}..."
                if len(model_output) > 100
                else f"Output: {model_output}"
            )

            # Detect language
            detection_result = self.language_classifier.classify(model_output)
            detected_lang = detection_result.label

            # Get critique from language critic
            critique = self.language_critic.critique(model_output)
            print(
                f"Language detected: {detected_lang} ({detection_result.confidence:.2f} confidence)"
            )

            # Validate if language is English
            is_english = detected_lang == "en" or "CORRECT:" in critique["feedback"]

            if is_english:
                print("[✓] Text is in English - validation passed")
                return model_output

            print(f"[✗] Text is not in English - detected {detected_lang}")
            print(f"Critique: {critique['feedback']}")

            # Improve - translate to English
            print(f"\n[Step {attempt}.2] Improving language...")
            improvement_prompt = self.prompt_factory.create_improvement_prompt(
                model_output, critique["feedback"]
            )

            improved_output = self.model.generate(improvement_prompt)

            # Extract the actual translation
            if "IMPROVED_TEXT:" in improved_output:
                translation = improved_output.split("IMPROVED_TEXT:")[1].strip()
                print(
                    f"Translation: {translation[:100]}..."
                    if len(translation) > 100
                    else f"Translation: {translation}"
                )
            else:
                translation = improved_output
                print(
                    f"Raw improvement: {translation[:100]}..."
                    if len(translation) > 100
                    else f"Raw improvement: {translation}"
                )

            # Update prompt for next attempt with language instruction
            current_prompt = f"{prompt}\n\nIMPORTANT: YOU MUST RESPOND IN ENGLISH ONLY."

        # If we've exceeded max attempts, return the last output
        return model_output


# Create a chain with the OpenAI model, language rule, and language critic
chain = DebugChain(
    model=generator_model,  # Generator model for responses
    language_classifier=language_classifier,  # Language classifier for evaluation
    language_critic=language_critic,  # Language critic for improvement
    max_attempts=2,  # Maximum number of attempts
    verbose=True,  # Enable verbose output
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

        print("\n=== SUCCESS ===")
        print("Final validated response (in English):")
        print(result)

        # Continuing the conversation in English
        follow_up_prompt = "Tell me more about what you can do to help me."
        follow_up_result = chain.run(follow_up_prompt)

        print("\nFollow-up response (should remain in English):")
        print(follow_up_result)

    except ValueError as e:
        print(f"\n=== FAILURE ===\nChain validation failed: {e}")


if __name__ == "__main__":
    run_conversation_example()
