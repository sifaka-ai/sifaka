#!/usr/bin/env python3
"""
Spam Classification Example using Sifaka.

This example demonstrates:
1. Creating and configuring a spam classifier
2. Training on sample ham/spam data
3. Classifying new messages
4. Using ClassifierRuleAdapter

Usage:
    python spam_classifier_example.py

Requirements:
    - Python environment with Sifaka installed (use pyenv environment "sifaka")
"""

import os
import sys
import tempfile
from pathlib import Path
import logging

# Add parent directory to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sifaka.rules.adapters import ClassifierRuleAdapter
from sifaka.rules.base import RuleConfig, RulePriority
from sifaka.classifiers.spam import SpamClassifier, SpamConfig
from sifaka.utils.logging import get_logger, log_operation, LogConfig, configure_logging

# Configure logging with colors and structured output
configure_logging(
    LogConfig(
        name="spam_example",
        use_colors=True,
        structured=True,
        log_to_file=True,
        log_dir=Path("logs"),
    )
)

# Initialize logger
logger = get_logger(__name__)

# Sample data for training (reduced)
SAMPLE_HAM = [
    "Hey, are we still meeting for coffee tomorrow?",
    "Please submit your report by Friday.",
    "Thanks for sending the documents, I'll review them today.",
    "The conference call is scheduled for 3 PM.",
    "Could you please share the presentation slides?",
]

SAMPLE_SPAM = [
    "URGENT: You have won $5,000,000 in our lottery! Claim now!",
    "CONGRATULATIONS! You've been selected for a FREE iPhone. Click here!",
    "Limited time offer: 90% OFF all luxury watches!",
    "Your bank account has been suspended. Verify your details immediately!",
    "URGENT: Your inheritance of $3.5M is waiting for you to claim!",
]

# Test messages for classification
TEST_MESSAGES = [
    "Can we reschedule our meeting to Thursday instead?",
    "ACT NOW: Get your FREE trial of our premium service!!!",
    "I have attached the requested report for your review.",
    "CONGRATULATIONS! You've won a luxury cruise vacation!",
]


@log_operation()
def create_classifier(model_path: str, is_training: bool = True) -> SpamClassifier:
    """Create a spam classifier with the specified configuration.

    Args:
        model_path: Path to save/load the model
        is_training: Whether the classifier is being created for training

    Returns:
        Configured SpamClassifier instance
    """
    name = "email_spam_classifier" if is_training else "loaded_spam_classifier"
    description = (
        "Classifies emails as spam or ham" if is_training else "Loaded pre-trained spam classifier"
    )

    spam_config = SpamConfig(
        min_confidence=0.7,
        max_features=500,
        use_bigrams=True,
        model_path=model_path,
    )

    classifier = SpamClassifier(
        name=name,
        description=description,
        spam_config=spam_config,
    )

    logger.structured(
        logging.INFO,
        "Classifier created",
        name=name,
        description=description,
        config=spam_config.__dict__,
    )

    return classifier


@log_operation()
def classify_messages(classifier: SpamClassifier, messages: list[str]) -> None:
    """Classify a list of messages and log the results.

    Args:
        classifier: Trained SpamClassifier instance
        messages: List of messages to classify
    """
    for i, message in enumerate(messages, 1):
        try:
            with logger.operation_context(f"Classifying message {i}"):
                result = classifier.classify(message)
                logger.structured(
                    logging.INFO,
                    "Message classification result",
                    message_id=i,
                    message_preview=message[:50],
                    classification=result.label,
                    confidence=round(result.confidence, 2),
                    probabilities=result.metadata.get("probabilities", {}),
                )
        except Exception as e:
            logger.error(f"Error classifying message {i}: {str(e)}")


@log_operation()
def test_rule_adapter(classifier: SpamClassifier, messages: list[str]) -> None:
    """Test the classifier using a rule adapter.

    Args:
        classifier: Trained SpamClassifier instance
        messages: List of messages to test
    """
    spam_rule = ClassifierRuleAdapter(
        classifier=classifier,
        rule_config=RuleConfig(priority=RulePriority.HIGH, cost=1.5),
    )

    for i, text in enumerate(messages, 1):
        try:
            with logger.operation_context(f"Testing rule adapter on message {i}"):
                result = spam_rule._validate_impl(text)
                logger.structured(
                    logging.INFO,
                    "Rule validation result",
                    message_id=i,
                    text_preview=text[:50],
                    passed=result.passed,
                    validation_message=result.message,
                )

                if "classification_result" in result.metadata:
                    cls_result = result.metadata["classification_result"]
                    logger.structured(
                        logging.INFO,
                        "Classification details",
                        message_id=i,
                        label=cls_result.label,
                        confidence=round(cls_result.confidence, 2),
                    )
        except Exception as e:
            logger.error(f"Error validating message {i} with rule adapter: {str(e)}")


def main():
    """Run the spam classification example."""
    with logger.operation_context("Spam Classification Example"):
        # Prepare training data
        texts = SAMPLE_HAM + SAMPLE_SPAM
        labels = ["ham"] * len(SAMPLE_HAM) + ["spam"] * len(
            SAMPLE_SPAM
        )  # "ham" for legitimate, "spam" for spam

        # Create temporary file for model storage
        temp_dir = Path(tempfile.gettempdir())
        model_path = str(temp_dir / "spam_classifier_model.pkl")

        try:
            # Configure and train spam classifier
            with logger.operation_context("Training classifier"):
                classifier = create_classifier(model_path, is_training=True)
                classifier.fit(texts, labels)
                logger.success("Classifier training completed")

            # Classify test messages
            classify_messages(classifier, TEST_MESSAGES)

            # Demonstrate loading saved model
            with logger.operation_context("Loading saved model"):
                loaded_classifier = create_classifier(model_path, is_training=False)
                loaded_classifier.warm_up()
                logger.success("Model loaded successfully")

            # Test with rule adapter
            test_rule_adapter(classifier, TEST_MESSAGES)

        except Exception as e:
            logger.error(f"An error occurred during example execution: {str(e)}")
        finally:
            # Clean up temporary model file
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
                    logger.success(f"Cleaned up temporary model file: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary model file: {str(e)}")


if __name__ == "__main__":
    main()
