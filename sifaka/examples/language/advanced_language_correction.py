"""
Advanced Language Correction Example

This example demonstrates a more advanced system where:
1. A custom language validator is created to enforce English language rules
2. An OpenAI chatbot starts responding in Spanish
3. The custom validator identifies the language is wrong
4. A critic helps correct the language to English
5. The conversation continues in English with proper validation
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

# Load environment variables from .env file (containing OPENAI_API_KEY)
load_dotenv()

from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.classifiers.language import LanguageClassifier, ClassifierConfig
from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.base import BaseValidator
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import Chain


# Create a custom configuration for language validation
@dataclass
class LanguageValidationConfig:
    """Configuration for language validation."""

    required_language: str = "en"  # The language that must be used (ISO code)
    threshold: float = 0.7  # Confidence threshold for language detection
    min_confidence: float = 0.6  # Minimum confidence required for detection
    fallback_lang: str = "en"  # Fallback language if detection fails
    language_indicators: Dict[str, List[str]] = None  # Common words/patterns for languages

    def __post_init__(self):
        # Default language indicators if none provided
        if self.language_indicators is None:
            self.language_indicators = {
                "en": ["the", "and", "is", "in", "to", "you", "of", "for", "that", "with"],
                "es": ["el", "la", "que", "y", "en", "de", "por", "cómo", "está", "para"],
                "fr": ["le", "la", "que", "et", "en", "des", "pour", "je", "ce", "qui"],
                "de": ["der", "die", "und", "ist", "das", "in", "den", "mit", "zu", "für"],
            }


# Create a custom validator for language rules
class LanguageValidator(BaseValidator[str]):
    """Custom validator that ensures text is in the required language."""

    def __init__(self, config: LanguageValidationConfig) -> None:
        """Initialize with configuration."""
        self._config = config
        # Initialize the language classifier with factory method
        self._language_classifier = LanguageClassifier.create(
            name="internal_language_detector",
            description="Detects text language",
            labels=list(LanguageClassifier.LANGUAGE_NAMES.keys()),
            params={
                "min_confidence": config.min_confidence,
                "fallback_lang": config.fallback_lang,
                "fallback_confidence": 0.0,
            },
        )

    def _get_word_match_score(self, text: str) -> Dict[str, float]:
        """Calculate a simple language score based on common word matches."""
        text_lower = text.lower()
        scores = {}

        # Count matched words for each language
        for lang, indicators in self._config.language_indicators.items():
            # Create regex patterns to match whole words
            patterns = [
                re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE) for word in indicators
            ]

            # Count matches
            match_count = sum(1 for pattern in patterns if pattern.search(text_lower))

            # Calculate score (0 to 1)
            scores[lang] = match_count / len(indicators) if indicators else 0

        return scores

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate that text is in the required language."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # First use the classifier for language detection
        classification = self._language_classifier.classify(text)
        detected_lang = classification.label
        confidence = classification.confidence

        # Also check common word patterns as a backup/additional signal
        word_scores = self._get_word_match_score(text)

        # Combine signals for the final decision
        is_required_language = detected_lang == self._config.required_language

        # If confidence is high enough, trust the classifier
        if confidence >= self._config.threshold:
            if is_required_language:
                return RuleResult(
                    passed=True,
                    message=f"Text is in the required language ({self._config.required_language})",
                    metadata={
                        "language": detected_lang,
                        "confidence": confidence,
                        "word_scores": word_scores,
                    },
                )
            else:
                return RuleResult(
                    passed=False,
                    message=f"Text is in {detected_lang}, not the required language ({self._config.required_language})",
                    metadata={
                        "language": detected_lang,
                        "required_language": self._config.required_language,
                        "confidence": confidence,
                        "word_scores": word_scores,
                    },
                )

        # If confidence is low, use word scores as backup
        required_lang_score = word_scores.get(self._config.required_language, 0)
        highest_lang = max(word_scores.items(), key=lambda x: x[1]) if word_scores else (None, 0)

        if (
            highest_lang[0] == self._config.required_language
            and highest_lang[1] >= self._config.threshold
        ):
            return RuleResult(
                passed=True,
                message=f"Text appears to be in the required language ({self._config.required_language})",
                metadata={
                    "method": "word_patterns",
                    "language": highest_lang[0],
                    "score": highest_lang[1],
                    "classifier_lang": detected_lang,
                    "classifier_confidence": confidence,
                },
            )

        return RuleResult(
            passed=False,
            message=f"Unable to confidently determine if text is in the required language",
            metadata={
                "classifier_lang": detected_lang,
                "classifier_confidence": confidence,
                "word_scores": word_scores,
                "required_language": self._config.required_language,
            },
        )


# Define a custom language rule class
class EnglishLanguageRule(Rule):
    """Rule that ensures text is in English."""

    def __init__(
        self,
        name: str = "english_language_rule",
        description: str = "Ensures text is in English",
        config: Optional[RuleConfig] = None,
        validator: Optional[LanguageValidator] = None,
    ) -> None:
        """Initialize the English language rule."""
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> LanguageValidator:
        """Create a default validator from config."""
        lang_config = LanguageValidationConfig(**self._rule_params)
        return LanguageValidator(lang_config)


# Main example code
def run_advanced_language_example():
    """Run the advanced language correction example."""

    # Configure OpenAI model
    model = OpenAIProvider(
        model_name="gpt-3.5-turbo",
        config=ModelConfig(
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=1000,
        ),
    )

    # Create the custom English language rule
    language_rule = EnglishLanguageRule(
        config=RuleConfig(
            params={
                "required_language": "en",
                "threshold": 0.6,
                "min_confidence": 0.5,
            }
        )
    )

    # Create a language critic to correct non-English responses
    language_critic = PromptCritic(
        llm_provider=model,
        config=PromptCriticConfig(
            name="language_critic",
            description="Corrects non-English text to English",
            system_prompt=(
                "You are a language critic specializing in correcting non-English text to English. "
                "When you receive text that is not in English, identify the language and provide "
                "a corrected English version that conveys the same meaning. "
                "Explain what was wrong with the original text and why the correction is better. "
                "Your goal is to help produce natural, fluent English text."
            ),
        ),
    )

    # Create the chain with the model, custom rule, and critic
    chain = Chain(
        model=model,
        rules=[language_rule],
        critic=language_critic,
        max_attempts=3,
    )

    # Start with a prompt that will likely generate Spanish
    initial_prompt = (
        "Responde en español: ¿Cómo estás hoy? Cuéntame sobre el clima en tu ubicación."
    )

    try:
        # The first attempt should generate Spanish, fail validation, and then be corrected
        result = chain.run(initial_prompt)

        print("Final validated response (in English):")
        print(result.output)
        print("\nCritique details:")
        print(result.critique_details)

        # Continue the conversation in English
        follow_up_prompt = "Can you tell me more about language detection in AI systems?"
        follow_up_result = chain.run(follow_up_prompt)

        print("\nFollow-up response (should remain in English):")
        print(follow_up_result.output)

    except ValueError as e:
        print(f"Chain validation failed: {e}")


if __name__ == "__main__":
    run_advanced_language_example()
