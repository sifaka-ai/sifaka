#!/usr/bin/env python3
"""
Spam Classification Example using Sifaka.

This example demonstrates:
1. Creating and configuring a spam classifier with proper configuration
2. Training on sample ham/spam data
3. Classifying new messages
4. Using ClassifierRuleAdapter for rule-based validation
5. Model persistence and loading

Usage:
    python spam_classifier_example.py

Requirements:
    - Python environment with Sifaka installed
    - scikit-learn (automatically handled by SpamClassifier)
"""

import logging
import os
from pathlib import Path
import tempfile

from sifaka.classifiers.spam import SpamClassifier, SpamConfig
from sifaka.rules.adapters import ClassifierRuleAdapter
from sifaka.rules.base import RuleConfig, RulePriority
from sifaka.utils.logging import LogConfig, configure_logging, get_logger, log_operation

# Configure logging
configure_logging(
    LogConfig(
        name="spam_example",
        use_colors=True,
        structured=True,
        log_to_file=True,
        log_dir=Path("logs"),
    )
)

logger = get_logger(__name__)

# Sample data for training
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
def create_classifier(model_path: str) -> SpamClassifier:
    """Create a properly configured spam classifier.

    Args:
        model_path: Path to save/load the model

    Returns:
        Configured SpamClassifier instance
    """
    # Create spam configuration with optimal settings
    spam_config = SpamConfig(
        min_confidence=0.8,  # Higher threshold for more reliable predictions
        max_features=2000,  # Increased feature set for better accuracy
        use_bigrams=True,  # Enable bigrams for better context understanding
        model_path=model_path,
        random_state=42,  # For reproducibility
    )

    # Create classifier with configuration
    classifier = SpamClassifier(
        name="email_spam_classifier",
        description="Classifies emails as spam or ham using Naive Bayes",
        spam_config=spam_config,
    )

    logger.structured(logging.INFO, "Classifier created", config=spam_config.__dict__)

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
    # Create rule adapter with high priority for spam detection
    spam_rule = ClassifierRuleAdapter(
        classifier=classifier,
        rule_config=RuleConfig(
            priority=RulePriority.HIGH, cost=1.5  # Higher cost for spam violations
        ),
    )

    for i, text in enumerate(messages, 1):
        try:
            with logger.operation_context(f"Testing rule adapter on message {i}"):
                result = spam_rule.validate(text)  # Use public validate method
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
        labels = ["ham"] * len(SAMPLE_HAM) + ["spam"] * len(SAMPLE_SPAM)

        # Create temporary file for model storage
        temp_dir = Path(tempfile.gettempdir())
        model_path = str(temp_dir / "spam_classifier_model.pkl")

        try:
            # Create and train classifier
            with logger.operation_context("Training classifier"):
                classifier = create_classifier(model_path)
                classifier.warm_up()  # Initialize the classifier
                classifier.fit(texts, labels)
                logger.success("Classifier training completed")

            # Classify test messages
            classify_messages(classifier, TEST_MESSAGES)

            # Test rule-based validation
            test_rule_adapter(classifier, TEST_MESSAGES)

            # Demonstrate loading saved model
            with logger.operation_context("Loading saved model"):
                loaded_classifier = SpamClassifier.create_pretrained(
                    texts=texts,
                    labels=labels,
                    name="pretrained_spam_classifier",
                    description="Pre-trained spam classifier",
                    spam_config=SpamConfig(model_path=model_path),
                )
                loaded_classifier.warm_up()  # Initialize the loaded classifier
                logger.success("Pre-trained classifier loaded successfully")

                # Verify loaded model works
                classify_messages(loaded_classifier, TEST_MESSAGES[:2])

        except Exception as e:
            logger.error(f"Error in spam classification example: {str(e)}")
        finally:
            # Clean up temporary model file
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Cleaned up temporary model file: {model_path}")


if __name__ == "__main__":
    main()
