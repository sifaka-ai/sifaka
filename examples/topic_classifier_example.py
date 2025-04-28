#!/usr/bin/env python3
"""
Topic Classification Example using Sifaka.

This example demonstrates:
1. Creating and configuring a topic classifier
2. Training the model on a corpus of documents
3. Classifying new documents into topics
4. Using a classifier rule adapter

Usage:
    python topic_classifier_example.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
"""

import os
import sys

# Add parent directory to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dotenv import load_dotenv

from sifaka.classifiers import ClassifierConfig
from sifaka.classifiers.topic import TopicClassifier, TopicConfig
from sifaka.utils.logging import get_logger

# Initialize logger from Sifaka
logger = get_logger(__name__)

# Sample corpus for training (reduced set)
SAMPLE_CORPUS = [
    # Technology documents
    "Machine learning algorithms are used to process and analyze data. Neural networks and deep learning are revolutionizing AI applications.",
    "Cloud computing provides scalable infrastructure and services. AWS, Azure, and Google Cloud are major providers in this space.",
    "Blockchain technology enables secure and transparent transactions without intermediaries. Cryptocurrencies like Bitcoin use blockchain.",
    # Health documents
    "Proper nutrition is essential for health, including balanced intake of proteins, carbohydrates, vitamins, and minerals.",
    "Regular exercise improves cardiovascular health, reduces stress, and helps maintain healthy weight and muscle mass.",
    "Mental health is as important as physical health, encompassing emotional, psychological, and social well-being.",
    # Finance documents
    "Investment strategies include diversification, asset allocation, and risk management to optimize portfolio returns.",
    "Stock markets allow investors to buy and sell shares of publicly traded companies based on performance expectations.",
    "Banking systems facilitate transactions, savings, loans, and other financial services for individuals and businesses.",
    # Education documents
    "Educational methodologies include student-centered learning, project-based instruction, and flipped classroom models.",
    "Higher education institutions offer undergraduate and graduate degrees, conducting research and advancing knowledge.",
    "Educational technology enhances learning experiences through interactive content, adaptive learning, and analytics.",
]

# Test documents for classification
TEST_DOCUMENTS = [
    "Deep learning and neural networks are transforming how we process large datasets in cloud environments.",
    "A balanced diet with proper vitamins and regular exercise is key to maintaining good physical and mental health.",
    "Diversifying your investment portfolio across different asset classes can help manage risk in volatile markets.",
]

def create_topic_classifier():
    """Create and configure the topic classifier."""
    classifier = TopicClassifier(
        name="document_topic_classifier",
        description="Classifies documents into subject topics",
        topic_config=TopicConfig(
            num_topics=4,
            min_confidence=0.2,
            max_features=500,
            top_words_per_topic=5,
        ),
        config=ClassifierConfig(
            labels=["technology", "politics", "sports", "general"],
            cost=2.0,
            min_confidence=0.2,
        ),
    )

    # Sample training data
    training_data = [
        # Technology examples
        "The new AI model demonstrates remarkable performance in natural language processing tasks.",
        "Cloud computing services have revolutionized how businesses store and process data.",
        "The latest smartphone features include advanced facial recognition and 5G connectivity.",
        # Politics examples
        "The Senate passed a new bill addressing climate change and renewable energy.",
        "Local elections saw record turnout as voters expressed concerns about education policy.",
        "International diplomacy efforts focus on maintaining regional stability.",
        # Sports examples
        "The championship game went into overtime with a dramatic finish.",
        "Athletes prepare for the upcoming international competition with intensive training.",
        "The team's strategy paid off with a decisive victory in the finals.",
        # General examples
        "Weather forecasts predict mild temperatures and clear skies for the weekend.",
        "Community events bring together people from diverse backgrounds.",
        "Recent studies show changing trends in consumer behavior.",
    ]

    # Train the classifier
    classifier.fit(training_data)
    return classifier

def main():
    """Run the topic classification example."""
    # Load environment variables
    load_dotenv()

    # Set up logging
    logger = get_logger(__name__)
    logger.info("Starting topic classification example...")

    # Create and train topic classifier
    logger.info("Creating and training topic classifier...")
    topic_classifier = create_topic_classifier()

    # Print discovered topics
    logger.info("Discovered topics:")
    for i, topic_label in enumerate(topic_classifier.config.labels):
        logger.info(f"Topic {i + 1}: {topic_label}")

    # Example texts to classify
    example_texts = [
        # Technology
        """The latest smartphone features a revolutionary AI chip that enhances
        photo quality and battery life. The neural processing unit can handle
        complex machine learning tasks locally.""",
        # Politics
        """The Senate will vote on the new infrastructure bill next week.
        Opposition leaders have criticized the proposed budget allocation
        and environmental impact assessments.""",
        # Sports
        """In a thrilling match, the underdog team scored in the final minutes
        to secure their place in the championship. The crowd erupted as the
        winning goal found the back of the net.""",
        # Mixed content
        """While technology companies debate AI regulation, athletes prepare
        for the upcoming tournament. Meanwhile, lawmakers discuss climate
        change policies.""",
    ]

    # Test classifier on examples
    logger.info("\nTesting classifier on examples...")
    for i, text in enumerate(example_texts, 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"Text: {text[:100]}...")

        # Classify text
        result = topic_classifier.classify(text)

        # Print results
        logger.info(f"Detected topic: {result.label}")
        logger.info(f"Confidence: {result.confidence:.2f}")

        # Print top features if available
        if result.metadata and "top_features" in result.metadata:
            logger.info("\nTop features:")
            for feature, score in result.metadata["top_features"].items():
                logger.info(f"  {feature}: {score:.3f}")

    logger.info("\nTopic classification example completed.")

if __name__ == "__main__":
    main()
