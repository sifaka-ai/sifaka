"""
Example of using multiple classifiers together with Sifaka.

This example demonstrates how to:
1. Combine multiple classifiers
2. Use them with rules
3. Make decisions based on multiple classification results
4. Handle errors and edge cases
"""

import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from sifaka import Reflector
from sifaka.models import AnthropicProvider
from sifaka.rules import ClassifierRule
from sifaka.classifiers.base import Classifier, ClassificationResult
from sifaka.critique import PromptCritique

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentClassifier(Classifier):
    """A simple sentiment classifier."""

    def __init__(self, name: str = "sentiment", description: str = "Sentiment analysis"):
        super().__init__(name=name, description=description)
        self.positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
        self.negative_words = ["bad", "terrible", "awful", "horrible", "poor"]

    def classify(self, text: str) -> ClassificationResult:
        """Classify text sentiment."""
        text = text.lower()
        pos_count = sum(1 for word in self.positive_words if word in text)
        neg_count = sum(1 for word in self.negative_words if word in text)

        total = pos_count + neg_count
        if total == 0:
            return ClassificationResult(label="neutral", confidence=1.0)

        if pos_count > neg_count:
            return ClassificationResult(label="positive", confidence=pos_count / total)
        elif neg_count > pos_count:
            return ClassificationResult(label="negative", confidence=neg_count / total)
        else:
            return ClassificationResult(label="neutral", confidence=0.5)


class ComplexityClassifier(Classifier):
    """A simple text complexity classifier."""

    def __init__(self, name: str = "complexity", description: str = "Text complexity analysis"):
        super().__init__(name=name, description=description)

    def classify(self, text: str) -> ClassificationResult:
        """Classify text complexity."""
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        if avg_word_length > 7:
            return ClassificationResult(label="complex", confidence=0.8)
        elif avg_word_length > 5:
            return ClassificationResult(label="moderate", confidence=0.7)
        else:
            return ClassificationResult(label="simple", confidence=0.9)


def main():
    """Example usage of multiple classifiers with Sifaka."""
    # Load environment variables
    load_dotenv()

    # Initialize the model provider
    model = AnthropicProvider(model_name="claude-3-haiku-20240307")

    # Create classifiers
    sentiment_classifier = SentimentClassifier()
    complexity_classifier = ComplexityClassifier()

    # Create classifier rules
    sentiment_rule = ClassifierRule(
        name="sentiment_check",
        description="Checks for appropriate sentiment",
        classifier=sentiment_classifier,
        threshold=0.6,
        valid_labels=["positive", "neutral"],
    )

    complexity_rule = ClassifierRule(
        name="complexity_check",
        description="Checks text complexity",
        classifier=complexity_classifier,
        threshold=0.7,
        valid_labels=["simple", "moderate"],
    )

    # Create a critic for improving outputs that fail validation
    critic = PromptCritique(model=model)

    # Create a reflector with both rules
    reflector = Reflector(
        name="content_validator",
        model=model,
        rules=[sentiment_rule, complexity_rule],
        critique=True,
        critic=critic,
    )

    # Example prompts
    prompts = [
        "Write a simple and positive message about learning to code.",
        "Explain quantum computing in technical terms.",
        "Write a negative review of a bad experience.",
    ]

    # Process each prompt
    for prompt in prompts:
        logger.info("\nProcessing prompt: %s", prompt)
        result = reflector.reflect(prompt)

        logger.info("Original output:")
        logger.info(result.original_output)

        if result.rule_violations:
            logger.info("\nRule violations:")
            for violation in result.rule_violations:
                logger.info("- %s: %s", violation.rule_name, violation.message)

        logger.info("\nFinal output:")
        logger.info(result.final_output)

        if result.trace:
            logger.info("\nTrace data:")
            for event in result.trace:
                logger.info("- %s: %s", event.stage, event.message)


if __name__ == "__main__":
    main()
