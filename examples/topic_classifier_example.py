#!/usr/bin/env python3
"""
Topic Classification Example using Sifaka.

This example demonstrates:
1. Creating and configuring a topic classifier
2. Training the model on a labeled corpus of documents
3. Classifying new documents into topics
4. Using proper topic configuration and validation

Usage:
    python topic_classifier_example.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
"""

import os
import sys
from typing import Dict, List, Tuple

# Add parent directory to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dotenv import load_dotenv

from sifaka.classifiers import ClassificationResult, ClassifierConfig
from sifaka.classifiers.topic import TopicClassifier, TopicConfig
from sifaka.utils.logging import get_logger

# Initialize logger from Sifaka
logger = get_logger(__name__)

# Define topic categories and their training data
TOPIC_DATA: Dict[str, List[str]] = {
    "technology": [
        "Machine learning algorithms are used to process and analyze data. Neural networks and deep learning are revolutionizing AI applications.",
        "Cloud computing provides scalable infrastructure and services. AWS, Azure, and Google Cloud are major providers in this space.",
        "Blockchain technology enables secure and transparent transactions without intermediaries. Cryptocurrencies like Bitcoin use blockchain.",
    ],
    "health": [
        "Proper nutrition is essential for health, including balanced intake of proteins, carbohydrates, vitamins, and minerals.",
        "Regular exercise improves cardiovascular health, reduces stress, and helps maintain healthy weight and muscle mass.",
        "Mental health is as important as physical health, encompassing emotional, psychological, and social well-being.",
    ],
    "finance": [
        "Investment strategies include diversification, asset allocation, and risk management to optimize portfolio returns.",
        "Stock markets allow investors to buy and sell shares of publicly traded companies based on performance expectations.",
        "Banking systems facilitate transactions, savings, loans, and other financial services for individuals and businesses.",
    ],
    "education": [
        "Educational methodologies include student-centered learning, project-based instruction, and flipped classroom models.",
        "Higher education institutions offer undergraduate and graduate degrees, conducting research and advancing knowledge.",
        "Educational technology enhances learning experiences through interactive content, adaptive learning, and analytics.",
    ],
}

# Test documents for classification
TEST_DOCUMENTS = [
    "Deep learning and neural networks are transforming how we process large datasets in cloud environments.",
    "A balanced diet with proper vitamins and regular exercise is key to maintaining good physical and mental health.",
    "Diversifying your investment portfolio across different asset classes can help manage risk in volatile markets.",
    "Modern educational platforms combine AI-driven personalization with collaborative learning tools.",
]


def prepare_training_data() -> Tuple[List[str], List[str]]:
    """Prepare training data and labels from the topic data dictionary."""
    documents = []
    labels = []
    for topic, texts in TOPIC_DATA.items():
        documents.extend(texts)
        labels.extend([topic] * len(texts))
    return documents, labels


def create_topic_classifier() -> TopicClassifier:
    """Create and configure the topic classifier with proper topic configuration."""
    try:
        # Create topic-specific configuration
        topic_config = TopicConfig(
            num_topics=5,  # Number of topics to extract
            min_confidence=0.2,  # Minimum confidence threshold
            max_features=1000,  # Max features for vectorization
            random_state=42,  # For reproducibility
            top_words_per_topic=10,  # Number of top words per topic
        )

        # Create base classifier config
        base_config = ClassifierConfig(
            labels=[f"topic_{i}" for i in range(topic_config.num_topics)],
            cost=1.0,
            min_confidence=topic_config.min_confidence,
            params={
                "num_topics": topic_config.num_topics,
                "max_features": topic_config.max_features,
                "random_state": topic_config.random_state,
                "top_words_per_topic": topic_config.top_words_per_topic,
            },
        )

        # Create classifier instance
        classifier = TopicClassifier(
            name="document_topic_classifier",
            description="Classifies documents into subject topics",
            config=base_config,
        )

        # Set topic config directly
        classifier._topic_config = topic_config

        # Prepare and validate training data
        documents, labels = prepare_training_data()
        if not documents or not labels:
            raise ValueError("No training data available")

        # Train classifier
        classifier.fit(documents)  # TopicClassifier.fit only takes texts, not labels
        return classifier

    except Exception as e:
        logger.error(f"Error creating topic classifier: {str(e)}")
        raise


def process_classification_result(result: ClassificationResult, text: str) -> None:
    """Process and display classification result with proper formatting."""
    logger.info(f"\nText: {text[:100]}...")
    logger.info(f"Detected topic: {result.label}")
    logger.info(f"Confidence: {result.confidence:.2f}")

    if result.metadata:
        if "top_features" in result.metadata:
            logger.info("\nTop features:")
            for feature, score in result.metadata["top_features"].items():
                logger.info(f"  {feature}: {score:.3f}")

        if "topic_scores" in result.metadata:
            logger.info("\nTopic scores:")
            for topic, score in result.metadata["topic_scores"].items():
                logger.info(f"  {topic}: {score:.3f}")


def main():
    """Run the topic classification example with proper error handling."""
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Starting topic classification example...")

        # Create and train topic classifier
        logger.info("Creating and training topic classifier...")
        topic_classifier = create_topic_classifier()

        # Print classifier information
        logger.info("\nClassifier configuration:")
        logger.info(f"Available topics: {', '.join(topic_classifier.config.labels)}")
        logger.info(f"Minimum confidence: {topic_classifier.config.min_confidence}")

        # Test classifier on examples
        logger.info("\nTesting classifier on examples...")
        for i, text in enumerate(TEST_DOCUMENTS, 1):
            logger.info(f"\nExample {i}:")
            result = topic_classifier.classify(text)
            process_classification_result(result, text)

        logger.info("\nTopic classification example completed successfully.")

    except Exception as e:
        logger.error(f"Error in topic classification example: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
