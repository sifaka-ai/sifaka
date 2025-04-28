#!/usr/bin/env python3
"""
Advanced Classification Example using Sifaka and scikit-learn.

This example demonstrates:
1. Creating and configuring genre and bias classifiers
2. Training the models on sample data
3. Classifying texts with both classifiers
4. Using the classifiers with rule adapters

Usage:
    python advanced_classifiers_example.py

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

from sifaka.classifiers.bias import BiasDetector
from sifaka.classifiers.genre import GenreClassifier
from sifaka.rules.adapters import ClassifierRuleAdapter
from sifaka.rules.base import RuleConfig, RulePriority
from sifaka.utils.logging import get_logger

# Initialize logger from Sifaka
logger = get_logger(__name__)

# Sample data for genre classification (reduced set)
GENRE_SAMPLES = {
    "news": [
        "Breaking news: A major earthquake struck the coast yesterday, causing significant damage to infrastructure.",
        "The president announced a new economic plan aimed at reducing inflation and creating jobs.",
        "Scientists have discovered a new species of deep-sea marine life, according to a report published in the Journal.",
    ],
    "fiction": [
        "The moon cast an eerie glow on the abandoned mansion as Sarah cautiously approached the creaking front door.",
        "The dragon unfurled its massive wings and roared, sending a tremor through the ancient forest.",
        "Time seemed to slow as he leaned in for the kiss, the years of longing finally fading away.",
    ],
    "academic": [
        "The study examines the correlation between socioeconomic factors and educational outcomes across diverse demographic groups.",
        "This paper presents a theoretical framework for understanding quantum entanglement in multi-particle systems.",
        "The researchers conducted a meta-analysis of 42 peer-reviewed studies, revealing statistically significant patterns.",
    ],
}

# Sample data for bias detection (reduced set)
BIAS_SAMPLES = {
    "gender": [
        "Women are naturally better at nurturing roles while men excel at leadership positions.",
        "Female employees tend to be more emotional in workplace conflicts than their male counterparts.",
        "The company needs strong, decisive men who can make tough decisions without getting emotional.",
    ],
    "racial": [
        "That neighborhood has become dangerous since more immigrants moved in.",
        "Certain racial groups are naturally more athletic while others are more academically inclined.",
        "You can't trust people from that part of the world; they have different values.",
    ],
    "neutral": [
        "The research study included participants from diverse backgrounds and controlled for various demographic factors.",
        "The company's hiring policy states that candidates are evaluated based on their qualifications and experience.",
        "Both approaches have advantages and disadvantages that should be considered when making a decision.",
    ],
}

# Test documents for classification
TEST_DOCUMENTS = [
    "The latest research paper in the Journal of Quantum Physics explores theoretical implications of string theory.",
    "Once upon a time, in a faraway kingdom, a brave princess set out on a journey to rescue the prince.",
    "According to experts, women generally avoid technical fields because they prefer working with people.",
    "That immigrant neighborhood is driving down property values and increasing crime rates across the city.",
]


def prepare_data(samples_dict):
    """Prepare training data from samples dictionary."""
    texts = []
    labels = []

    for label, samples in samples_dict.items():
        texts.extend(samples)
        labels.extend([label] * len(samples))

    return texts, labels


def main():
    """Run the advanced classifiers example."""
    logger.info("Starting advanced classifiers example...")

    # Prepare genre classification data
    genre_texts, genre_labels = prepare_data(GENRE_SAMPLES)

    # Prepare bias detection data
    bias_texts, bias_labels = prepare_data(BIAS_SAMPLES)

    # Create and train genre classifier
    logger.info("Creating and training genre classifier...")
    genre_classifier = GenreClassifier.create(
        name="custom_genre_classifier",
        description="Custom genre classifier for demo",
        labels=list(GENRE_SAMPLES.keys()),
        min_confidence=0.6,
        cost=2.0,
        params={
            "max_features": 1000,
            "use_ngrams": True,
        },
    )
    genre_classifier.fit(genre_texts, genre_labels)

    # Create and train bias detector
    logger.info("Creating and training bias detector...")
    bias_detector = BiasDetector.create(
        name="custom_bias_detector",
        description="Custom bias detector for demo",
        labels=list(BIAS_SAMPLES.keys()),
        min_confidence=0.7,
        cost=2.5,
    )
    bias_detector.fit(bias_texts, bias_labels)

    # Test the classifiers
    logger.info("\nTesting classifiers...")
    for i, doc in enumerate(TEST_DOCUMENTS):
        logger.info(f"\nDocument {i+1}: '{doc[:50]}...'")

        # Test genre classifier
        genre_result = genre_classifier.classify(doc)
        logger.info(f"Genre: {genre_result.label} (confidence: {genre_result.confidence:.2f})")

        # Test bias detector
        bias_result = bias_detector.classify(doc)
        logger.info(f"Bias: {bias_result.label} (confidence: {bias_result.confidence:.2f})")

    # Create rule adapters
    logger.info("\nTesting classifier rule adapters...")

    # Create genre rule adapter
    genre_rule = ClassifierRuleAdapter(
        classifier=genre_classifier,
        rule_config=RuleConfig(priority=RulePriority.MEDIUM, cost=2.0),
    )

    # Create bias rule adapter
    bias_rule = ClassifierRuleAdapter(
        classifier=bias_detector,
        rule_config=RuleConfig(priority=RulePriority.HIGH, cost=2.5),
    )

    # Test rule adapters
    for i, doc in enumerate(TEST_DOCUMENTS):
        logger.info(f"\nDocument {i+1}: '{doc[:50]}...'")

        # Test genre rule
        genre_result = genre_rule.validate(doc)
        logger.info(f"Genre Rule - Passed: {genre_result.passed}, Message: {genre_result.message}")
        if "classification_result" in genre_result.metadata:
            cls_result = genre_result.metadata["classification_result"]
            logger.info(f"Genre: {cls_result.label} (confidence: {cls_result.confidence:.2f})")

        # Test bias rule
        bias_result = bias_rule.validate(doc)
        logger.info(f"Bias Rule - Passed: {bias_result.passed}, Message: {bias_result.message}")
        if "classification_result" in bias_result.metadata:
            cls_result = bias_result.metadata["classification_result"]
            logger.info(f"Bias: {cls_result.label} (confidence: {cls_result.confidence:.2f})")

    logger.info("\nAdvanced classifiers example completed.")


if __name__ == "__main__":
    main()
