"""
Language Correction Example

This example demonstrates a system where:
1. An OpenAI chatbot starts responding in Spanish
2. A rule using the LanguageClassifier correctly detects non-English text
3. A critic provides feedback to correct the language to English
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
from sifaka.chain import Chain, ChainResult

# Configure main model for generating responses
generator_model = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    config=ModelConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=1000,
    ),
)

# Configure critic model (using a more capable model for better instruction following)
critic_model = OpenAIProvider(
    model_name="gpt-4",  # Using GPT-4 for better instruction following
    config=ModelConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.2,  # Lower temperature for more deterministic outputs
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

# Create a critic that ONLY focuses on language detection and correction
language_critic = PromptCritic(
    llm_provider=critic_model,  # Using GPT-4 for the critic
    config=PromptCriticConfig(
        name="language_critic",
        description="Language detection and correction critic",
        system_prompt=(
            "# LANGUAGE ENFORCEMENT CRITIC\n\n"
            "You have ONE JOB: When text is not in English, instruct to use English instead.\n\n"
            "## YOUR EXACT INSTRUCTIONS:\n"
            "1. If the text is NOT in English:\n"
            "   - Set score: 0.0\n"
            '   - Feedback starts with: "ERROR: NOT ENGLISH - DETECTED [LANGUAGE]."\n'
            '   - Always state: "YOU MUST RESPOND IN ENGLISH ONLY."\n'
            "   - Provide an English translation\n\n"
            "2. If the text IS in English:\n"
            "   - Set score: 1.0\n"
            '   - Feedback: "Correct: Text is in English."\n\n'
            "DO NOT comment on quality, content, grammar, or anything else.\n"
            "ONLY focus on language detection and English enforcement."
        ),
    ),
)


# Modified Chain class for debugging
class DebugChain(Chain):
    def run(self, prompt):
        """Run Chain with debugging output."""
        attempts = 0
        last_critique = None

        # Store the original prompt
        current_prompt = prompt

        while attempts < self.max_attempts:
            print(f"\n--- Attempt {attempts + 1} ---")
            print(f"Prompt: {current_prompt[:100]}...")

            # Generate output
            output = self.model.generate(current_prompt)
            print(f"Model output: {output}")

            # Validate output
            rule_results = []
            all_passed = True

            for rule in self.rules:
                result = rule.validate(output)
                rule_results.append(result)
                if not result.passed:
                    all_passed = False
                    print(f"Rule failed: {result.message}")
                    print(f"Rule metadata: {result.metadata}")

            # If validation passed, return result
            if all_passed:
                critique_details = None
                if last_critique:
                    from sifaka.critics.prompt import CriticMetadata

                    if isinstance(last_critique, CriticMetadata):
                        critique_details = last_critique.__dict__
                    elif isinstance(last_critique, dict):
                        critique_details = last_critique

                return ChainResult(
                    output=output,
                    rule_results=rule_results,
                    critique_details=critique_details,
                )

            # If validation failed but we have no critic, raise error
            if not self.critic:
                error_messages = [r.message for r in rule_results if not r.passed]
                raise ValueError(f"Validation failed. Errors:\n" + "\n".join(error_messages))

            # If we have a critic and validation failed, try to improve
            if attempts < self.max_attempts - 1:
                critique = self.critic.critique(output)
                last_critique = critique

                from sifaka.critics.prompt import CriticMetadata

                if isinstance(critique, CriticMetadata):
                    feedback = critique.feedback
                    improved_output = critique.improved_output
                    print(f"Critic feedback: {feedback}")
                    print(f"Improved output: {improved_output}")
                else:
                    feedback = critique.get("feedback", "")
                    improved_output = critique.get("improved_output", "")
                    print(f"Critic feedback: {feedback}")
                    print(f"Improved output: {improved_output}")

                # Update the prompt with the feedback and explicit English instruction
                current_prompt = f"{prompt}\n\nLanguage feedback: {feedback}\n\nIMPORTANT: YOU MUST RESPOND IN ENGLISH ONLY."
                attempts += 1
                continue

            # If we're out of attempts or no critic, raise error
            error_messages = [r.message for r in rule_results if not r.passed]
            raise ValueError(
                f"Validation failed after {attempts + 1} attempts. Errors:\n"
                + "\n".join(error_messages)
            )

        # Should never reach here due to while loop condition
        raise RuntimeError("Unexpected end of chain execution")


# Create a chain with the OpenAI model, language rule, and language critic
chain = DebugChain(
    model=generator_model,  # Generator model for responses
    rules=[language_rule],  # Language detection rule
    critic=language_critic,  # Language correction critic
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

        print("\n=== SUCCESS ===")
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
        print(f"\n=== FAILURE ===\nChain validation failed: {e}")


if __name__ == "__main__":
    run_conversation_example()
