#!/usr/bin/env python3
"""
Basic Sifaka Usage Example.

This example demonstrates:
1. Sentiment analysis of text
2. Readability classification
3. Content improvement using OpenAI

Usage:
    python usage.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
    - Sifaka readability extras: pip install sifaka[readability]
    - OpenAI API key in OPENAI_API_KEY environment variable
"""

import os
import sys

# Add parent directory to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing dotenv package. Install with: pip install python-dotenv")
    sys.exit(1)

from sifaka.classifiers.readability import ReadabilityClassifier
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.models import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.sentiment import create_emotional_content_rule, create_sentiment_rule
from sifaka.utils.logging import get_logger

# Initialize logger from Sifaka
logger = get_logger(__name__)


def main():
    """Run the basic usage example."""
    # Load environment variables
    load_dotenv()

    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    logger.info("Starting basic usage example...")

    # 1. Set up OpenAI model for critic
    model = OpenAIProvider(
        model_name="gpt-3.5-turbo",
        config=ModelConfig(
            api_key=openai_api_key,
            temperature=0.7,
            max_tokens=1000,
        ),
    )

    # 2. Create basic rules

    # Sentiment rule for positive/negative detection
    sentiment_rule = create_sentiment_rule(
        name="sentiment_analysis",
        description="Analyzes text sentiment (positive/negative)",
        config={
            "threshold": 0.6,
            "positive_words": ["good", "great", "excellent", "happy", "enjoy", "love"],
            "negative_words": ["bad", "poor", "terrible", "sad", "dislike", "hate"],
        },
    )

    # Emotional content rule
    emotion_rule = create_emotional_content_rule(
        name="emotional_content",
        description="Analyzes emotional content",
        config={
            "categories": {
                "joy": ["happy", "delighted", "excited", "joyful", "cheerful"],
                "sadness": ["sad", "unhappy", "gloomy", "miserable", "depressed"],
                "anger": ["angry", "furious", "irritated", "mad", "annoyed"],
            },
            "min_emotion_score": 0.3,
            "max_emotion_score": 0.8,
        },
    )

    # 3. Create readability classifier
    readability_classifier = ReadabilityClassifier(
        name="readability_analyzer", description="Analyzes text complexity"
    )

    # 4. Create content critic
    critic = PromptCritic(
        model=model,
        config=PromptCriticConfig(
            name="content_improver",
            description="Improves content based on feedback",
            system_prompt="You are an expert editor that improves text based on feedback.",
            temperature=0.7,
            max_tokens=1000,
        ),
    )

    # Example texts with different characteristics
    examples = [
        {
            "title": "Positive College-Level Text",
            "text": "The innovative product exceeded all expectations with its intuitive interface and remarkable performance. The robust feature set establishes a new standard for excellence in this competitive market.",
        },
        {
            "title": "Negative Elementary-Level Text",
            "text": "I didn't like this game. It was boring and too hard to play. The graphics were bad and the story made no sense. What a waste of money!",
        },
    ]

    # Process each example
    for i, example in enumerate(examples, 1):
        logger.info(f"\n===== Example {i}: {example['title']} =====")
        text = example["text"]
        logger.info(f"Text: '{text}'")

        # Analyze sentiment
        sentiment_result = sentiment_rule.validate(text)
        sentiment_score = sentiment_result.metadata.get("sentiment_score", 0.0)
        positive_terms = sentiment_result.metadata.get("positive_matches", 0)
        negative_terms = sentiment_result.metadata.get("negative_matches", 0)

        logger.info(f"\nSentiment Analysis:")
        logger.info(f"Is Positive: {sentiment_result.passed}")
        logger.info(f"Score: {sentiment_score:.2f}")
        logger.info(f"Positive terms: {positive_terms}, Negative terms: {negative_terms}")

        # Analyze emotional content
        emotion_result = emotion_rule.validate(text)
        logger.info(f"\nEmotional Content Analysis:")
        logger.info(f"Is Balanced: {emotion_result.passed}")

        if "emotion_scores" in emotion_result.metadata:
            scores = emotion_result.metadata["emotion_scores"]
            logger.info("Detected emotions:")
            for emotion, score in scores.items():
                if score > 0:
                    logger.info(f"{emotion}: {score:.2f}")

        # Analyze readability
        readability_result = readability_classifier.classify(text)
        logger.info(f"\nReadability Analysis:")
        logger.info(f"Reading Level: {readability_result.label}")
        logger.info(f"Confidence: {readability_result.confidence:.2f}")

        if "metrics" in readability_result.metadata:
            metrics = readability_result.metadata["metrics"]
            logger.info(f"Flesch Reading Ease: {metrics.get('flesch_reading_ease', 0):.1f}")
            logger.info(f"Grade Level: {metrics.get('flesch_kincaid_grade', 0):.1f}")

        # Demonstrate content improvement
        target_sentiment = "negative" if sentiment_result.passed else "positive"
        target_level = "middle"

        logger.info(
            f"\nImproving to {target_sentiment.upper()} sentiment, {target_level.upper()} reading level..."
        )

        # Create violations based on targets
        violations = [
            {
                "rule": "sentiment",
                "message": f"Change sentiment to {target_sentiment}",
                "metadata": {
                    "current": "positive" if sentiment_result.passed else "negative",
                    "target": target_sentiment,
                },
            },
            {
                "rule": "readability",
                "message": f"Adjust to {target_level} reading level",
                "metadata": {"current": readability_result.label, "target": target_level},
            },
        ]

        # Use critic to improve content
        try:
            improved_text = critic.improve(text, violations)
            logger.info(f"Improved Text: '{improved_text}'")

            # Analyze improved text
            improved_sentiment = sentiment_rule.validate(improved_text).passed
            improved_readability = readability_classifier.classify(improved_text).label

            logger.info(f"New Sentiment: {'positive' if improved_sentiment else 'negative'}")
            logger.info(f"New Reading Level: {improved_readability}")

        except Exception as e:
            logger.error(f"Error improving content: {e}")

    logger.info("\nBasic usage example completed.")


if __name__ == "__main__":
    main()
