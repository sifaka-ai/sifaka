"""
Integration tests for the interaction between adapters, rules, and classifiers.

These tests ensure that adapters work correctly with rules and classifiers
in realistic usage scenarios.
"""

import pytest
from typing import Dict, Any, List

from sifaka.adapters.rules.base import BaseAdapter, create_adapter
from sifaka.adapters.rules.classifier import (
    ClassifierAdapter,
    create_classifier_adapter,
    create_classifier_rule,
    ClassifierRule
)
from sifaka.rules.base import Rule, RuleResult
from sifaka.classifiers.base import ClassificationResult


class SimpleClassifier:
    """A simple classifier for testing integration."""

    def __init__(self):
        self._config = {"labels": ["positive", "negative", "neutral"]}

    @property
    def name(self) -> str:
        return "simple_classifier"

    @property
    def description(self) -> str:
        return "A simple classifier for testing"

    @property
    def config(self) -> Any:
        return self._config

    def classify(self, text: str) -> ClassificationResult:
        """Classify text based on simple keyword matching."""
        text_lower = text.lower()

        # Detect sentiment based on keywords
        positive_words = ["good", "great", "excellent", "positive", "wonderful", "happy"]
        negative_words = ["bad", "terrible", "awful", "negative", "horrible", "sad"]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            label = "positive"
            confidence = min(0.5 + (pos_count - neg_count) * 0.1, 0.95)
        elif neg_count > pos_count:
            label = "negative"
            confidence = min(0.5 + (neg_count - pos_count) * 0.1, 0.95)
        else:
            label = "neutral"
            confidence = 0.7

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={"pos_words": pos_count, "neg_words": neg_count}
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Batch classify texts."""
        return [self.classify(text) for text in texts]


class LengthClassifier:
    """A classifier that classifies text based on length."""

    def __init__(self):
        self._config = {"labels": ["short", "medium", "long"]}

    @property
    def name(self) -> str:
        return "length_classifier"

    @property
    def description(self) -> str:
        return "Classifies text based on length"

    @property
    def config(self) -> Any:
        return self._config

    def classify(self, text: str) -> ClassificationResult:
        """Classify text based on length."""
        length = len(text)

        if length < 20:
            label = "short"
            confidence = 0.9
        elif length < 100:
            label = "medium"
            confidence = 0.9
        else:
            label = "long"
            confidence = 0.9

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={"length": length}
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Batch classify texts."""
        return [self.classify(text) for text in texts]


class TestAdapterWithRules:
    """Tests for using adapters with rules."""

    def test_classifier_adapter_with_rule(self):
        """Test using ClassifierAdapter with a Rule."""
        # Create classifier
        classifier = SimpleClassifier()

        # Create adapter directly
        adapter = ClassifierAdapter(classifier, valid_labels=["positive", "neutral"])

        # Validate positive text
        result = adapter.validate("This is a great example of a good text.")
        assert result.passed
        assert result.metadata["label"] == "positive"

        # Validate negative text
        result = adapter.validate("This is a terrible example of a bad text.")
        assert not result.passed
        assert result.metadata["label"] == "negative"

    def test_rule_created_from_adapter(self):
        """Test rules created from adapters."""
        # Create classifier
        classifier = SimpleClassifier()

        # Create rule from classifier
        rule = create_classifier_rule(
            classifier=classifier,
            valid_labels=["positive", "neutral"],
            threshold=0.7,
            name="sentiment_rule",
            description="Checks text sentiment"
        )

        # Validate positive text
        result = rule.validate("This is a great example of a good text.")
        assert result.passed
        assert result.metadata["label"] == "positive"

        # Validate negative text
        result = rule.validate("This is a terrible example of a bad text.")
        assert not result.passed
        assert result.metadata["label"] == "negative"

    def test_chain_of_rules(self):
        """Test multiple rules in a chain."""
        # Create classifiers
        sentiment_classifier = SimpleClassifier()
        length_classifier = LengthClassifier()

        # Create rules from classifiers
        sentiment_rule = create_classifier_rule(
            classifier=sentiment_classifier,
            valid_labels=["positive", "neutral"],
            name="sentiment_rule"
        )

        length_rule = create_classifier_rule(
            classifier=length_classifier,
            valid_labels=["medium", "long"],
            name="length_rule"
        )

        # Define a function to run rules in sequence
        def validate_with_rules(text: str, rules: List[Rule]) -> List[RuleResult]:
            results = []
            for rule in rules:
                result = rule.validate(text)
                results.append(result)
                if not result.passed:
                    break
            return results

        # Test with text that passes both rules
        text = "This is a good text that is of medium length and has positive sentiment."
        results = validate_with_rules(text, [sentiment_rule, length_rule])
        assert all(result.passed for result in results)
        assert len(results) == 2

        # Test with text that fails sentiment rule
        text = "This is a bad text that is of medium length but has negative sentiment."
        results = validate_with_rules(text, [sentiment_rule, length_rule])
        assert not results[0].passed
        assert len(results) == 1  # Should stop after first failure

        # Test with text that passes sentiment but fails length
        text = "Good."
        results = validate_with_rules(text, [sentiment_rule, length_rule])
        assert results[0].passed
        assert not results[1].passed
        assert len(results) == 2


class TestE2EScenarios:
    """End-to-end scenarios with adapters, rules, and classifiers."""

    def test_content_moderation_scenario(self):
        """Test a content moderation scenario."""
        # Create classifiers
        sentiment_classifier = SimpleClassifier()
        length_classifier = LengthClassifier()

        # Create rules from classifiers
        sentiment_rule = create_classifier_rule(
            classifier=sentiment_classifier,
            valid_labels=["positive", "neutral"],
            name="sentiment_rule",
            description="Ensures content has positive or neutral sentiment"
        )

        length_rule = create_classifier_rule(
            classifier=length_classifier,
            valid_labels=["medium", "long"],
            name="length_rule",
            description="Ensures content is not too short"
        )

        # Define a moderation function
        def moderate_content(text: str) -> Dict[str, Any]:
            """Moderate content using rules."""
            sentiment_result = sentiment_rule.validate(text)

            if not sentiment_result.passed:
                return {
                    "approved": False,
                    "reason": "Content contains negative sentiment",
                    "details": sentiment_result.metadata
                }

            length_result = length_rule.validate(text)

            if not length_result.passed:
                return {
                    "approved": False,
                    "reason": f"Content is too {length_result.metadata['label']}",
                    "details": length_result.metadata
                }

            return {
                "approved": True,
                "reason": "Content meets all criteria"
            }

        # Test with approved content
        good_content = "This is a good example of content that is long enough and has positive sentiment."
        result = moderate_content(good_content)
        assert result["approved"]

        # Test with negative content
        negative_content = "This is a terrible example of bad content."
        result = moderate_content(negative_content)
        assert not result["approved"]
        assert "negative sentiment" in result["reason"]

        # Test with short content
        short_content = "Good."
        result = moderate_content(short_content)
        assert not result["approved"]
        assert "too short" in result["reason"]

    def test_combining_multiple_classifiers(self):
        """Test combining multiple classifiers for content validation."""
        # Create classifiers
        sentiment_classifier = SimpleClassifier()
        length_classifier = LengthClassifier()

        # Create a composite adapter that combines results
        class CompositeValidator(BaseAdapter[str, SimpleClassifier]):
            """An adapter that combines results from multiple classifiers."""

            def __init__(self,
                         sentiment_classifier: SimpleClassifier,
                         length_classifier: LengthClassifier,
                         min_confidence: float = 0.7):
                super().__init__(sentiment_classifier)
                self.sentiment_classifier = sentiment_classifier
                self.length_classifier = length_classifier
                self.min_confidence = min_confidence

            def validate(self, text: str, **kwargs) -> RuleResult:
                # Handle empty text
                empty_result = self.handle_empty_text(text)
                if empty_result:
                    return empty_result

                try:
                    # Get sentiment classification
                    sentiment_result = self.sentiment_classifier.classify(text)

                    # Get length classification
                    length_result = self.length_classifier.classify(text)

                    # Check sentiment
                    positive_sentiment = sentiment_result.label == "positive"
                    sentiment_confident = sentiment_result.confidence >= self.min_confidence

                    # Check length
                    good_length = length_result.label in ["medium", "long"]
                    length_confident = length_result.confidence >= self.min_confidence

                    # Combined validation
                    passed = (positive_sentiment and sentiment_confident and
                              good_length and length_confident)

                    # Create validation message
                    message_parts = []
                    if not positive_sentiment:
                        message_parts.append("negative sentiment")
                    if not sentiment_confident:
                        message_parts.append("low sentiment confidence")
                    if not good_length:
                        message_parts.append("too short")
                    if not length_confident:
                        message_parts.append("low length confidence")

                    message = "Content is valid" if passed else f"Content has issues: {', '.join(message_parts)}"

                    return RuleResult(
                        passed=passed,
                        message=message,
                        metadata={
                            "sentiment": sentiment_result.label,
                            "sentiment_confidence": sentiment_result.confidence,
                            "length": length_result.label,
                            "length_confidence": length_result.confidence
                        }
                    )

                except Exception as e:
                    return RuleResult(
                        passed=False,
                        message=f"Validation error: {str(e)}",
                        metadata={"error_type": type(e).__name__}
                    )

        # Create the composite validator
        validator = CompositeValidator(
            sentiment_classifier=SimpleClassifier(),
            length_classifier=LengthClassifier()
        )

        # Test with valid content
        good_content = "This is a good example of content that is long enough and has positive sentiment."
        result = validator.validate(good_content)
        assert result.passed
        assert "Content is valid" in result.message
        assert result.metadata["sentiment"] == "positive"
        assert result.metadata["length"] in ["medium", "long"]

        # Test with negative content
        negative_content = "This is a terrible example of bad content that is still long enough."
        result = validator.validate(negative_content)
        assert not result.passed
        assert "negative sentiment" in result.message
        assert result.metadata["sentiment"] == "negative"

        # Test with short content
        short_content = "Good."
        result = validator.validate(short_content)
        assert not result.passed
        assert "too short" in result.message
        assert result.metadata["length"] == "short"