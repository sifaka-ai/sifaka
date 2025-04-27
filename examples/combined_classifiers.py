"""
Example of using multiple classifiers with Sifaka.

This example demonstrates:
1. Creating custom classifiers for content analysis
2. Using critics for content reflection and improvement
3. Using pattern rules for structural analysis
4. Combining multiple validation strategies
"""

import logging
import os
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import Field

from sifaka.models.base import ModelConfig
from sifaka.models.anthropic import AnthropicProvider
from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.rules.base import Rule, RuleResult, RuleConfig, RuleValidator, RulePriority
from sifaka.rules.pattern_rules import SymmetryRule, RepetitionRule
from sifaka.rules.adapters import ClassifierRuleAdapter
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import Chain
from sifaka.models.openai import OpenAIProvider
from sifaka.rules.classifier_rule import ClassifierRule, ClassifierRuleConfig, ClassifierProtocol

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReadabilityClassifier(BaseClassifier):
    """Classifier for text readability."""

    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"easy": 0.7, "medium": 0.4, "hard": 0.0}
    )

    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        """Initialize the classifier."""
        super().__init__(
            name="ReadabilityClassifier",
            description="Classifies text based on readability",
            config=config
            or ClassifierConfig(
                labels=["easy", "medium", "hard"],
                min_confidence=0.5,
            ),
        )

    def _classify_impl(self, text: str) -> ClassificationResult:
        """Classify text based on readability."""
        # Simple readability metric: average word length
        words = text.split()
        if not words:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={"reason": "no_words"},
            )

        avg_word_length = sum(len(word) for word in words) / len(words)
        normalized_length = min(1.0, avg_word_length / 10.0)  # Normalize to 0-1

        # Determine label based on thresholds
        for label, threshold in self.thresholds.items():
            if normalized_length >= threshold:
                return ClassificationResult(
                    label=label,
                    confidence=normalized_length,
                    metadata={"avg_word_length": avg_word_length},
                )

        return ClassificationResult(
            label="unknown",
            confidence=0.0,
            metadata={"reason": "no_match"},
        )

    @classmethod
    def create(cls) -> "ReadabilityClassifier":
        """Create a new instance with default configuration."""
        return cls()


class ToneClassifier(BaseClassifier):
    """Classifier for text tone."""

    tone_indicators: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "positive": ["great", "excellent", "good", "amazing", "wonderful"],
            "negative": ["bad", "poor", "terrible", "awful", "horrible"],
            "neutral": ["average", "typical", "standard", "normal", "regular"],
        }
    )

    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        """Initialize the classifier."""
        super().__init__(
            name="ToneClassifier",
            description="Classifies text based on tone",
            config=config
            or ClassifierConfig(
                labels=["positive", "negative", "neutral"],
                min_confidence=0.5,
            ),
        )

    def _classify_impl(self, text: str) -> ClassificationResult:
        """Classify text based on tone."""
        text = text.lower()
        word_count = len(text.split())
        if not word_count:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={"reason": "no_words"},
            )

        # Count indicators for each tone
        tone_counts = {tone: 0 for tone in self.tone_indicators}
        for tone, indicators in self.tone_indicators.items():
            for indicator in indicators:
                tone_counts[tone] += text.count(indicator)

        # Find dominant tone
        max_count = max(tone_counts.values())
        if max_count == 0:
            return ClassificationResult(
                label="neutral",
                confidence=0.5,
                metadata={"reason": "no_indicators"},
            )

        dominant_tone = max(tone_counts.items(), key=lambda x: x[1])[0]
        confidence = min(1.0, max_count / word_count)

        return ClassificationResult(
            label=dominant_tone,
            confidence=confidence,
            metadata={"tone_counts": tone_counts},
        )

    @classmethod
    def create(cls) -> "ToneClassifier":
        """Create a new instance with default configuration."""
        return cls()


@dataclass
class ClassifierValidator(RuleValidator[str]):
    """Validator for classifier rules."""

    classifier: BaseClassifier
    required_label: Optional[str] = None
    min_confidence: float = 0.5

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate text using the classifier."""
        result = self.classifier.classify(output)

        # If no required label, just report the classification
        if not self.required_label:
            return RuleResult(
                passed=True,
                message=f"Classified as {result.label} with confidence {result.confidence:.2f}",
                metadata=result.metadata,
            )

        # Check if classification matches requirements
        passed = result.label == self.required_label and result.confidence >= self.min_confidence

        return RuleResult(
            passed=passed,
            message=(
                f"Classification {result.label} matches required {self.required_label}"
                if passed
                else f"Expected {self.required_label}, got {result.label}"
            ),
            metadata=result.metadata,
        )

    def can_validate(self, output: str) -> bool:
        """Check if the input can be validated."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> Type[str]:
        """Get the type of input this validator accepts."""
        return str


class StringValidator(RuleValidator[str]):
    """Base validator for string inputs."""

    def can_validate(self, output: str) -> bool:
        """Check if the input can be validated."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> Type[str]:
        """Get the type of input this validator accepts."""
        return str

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate the input."""
        if not isinstance(output, str):
            return RuleResult(
                passed=False,
                message="Input must be a string",
                metadata={"error": "invalid_type"},
            )
        return RuleResult(
            passed=True,
            message="Input is a valid string",
            metadata={},
        )


def main():
    # Initialize model provider using environment variables
    model_provider = OpenAIProvider()

    # Create classifiers
    readability_classifier = ReadabilityClassifier.create()
    tone_classifier = ToneClassifier.create()

    # Create validators
    readability_validator = ClassifierValidator(readability_classifier)
    tone_validator = ClassifierValidator(tone_classifier)
    string_validator = StringValidator()

    # Create rules
    rules = [
        readability_validator,
        tone_validator,
        string_validator,
    ]

    # Create critic config
    critic_config = PromptCriticConfig(
        name="combined_critic",
        description="Improves text based on readability and tone",
        min_confidence=0.5,
    )

    # Create critic
    critic = PromptCritic(config=critic_config, model=model_provider)

    # Create chain
    chain = Chain(model=model_provider, rules=rules, critic=critic, max_attempts=3)

    # Example prompts
    prompts = [
        "Explain the role of quantum computing in cryptography",
        "Describe the impact of artificial intelligence on healthcare",
        "Discuss the principles of sustainable urban development",
    ]

    # Process each prompt
    for prompt in prompts:
        print(f"\nProcessing prompt: {prompt}")
        try:
            result = chain.run(prompt)
            print(f"Generated text: {result}")
        except ValueError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
