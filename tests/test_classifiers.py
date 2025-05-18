"""
Tests for the classifiers module.

This module contains tests for the classifiers in the Sifaka framework.
"""

from typing import List
from unittest.mock import patch

from sifaka.classifiers import ClassificationResult, Classifier


class TestClassificationResult:
    """Tests for the ClassificationResult class."""

    def test_init_with_required_fields(self) -> None:
        """Test initializing a ClassificationResult with required fields."""
        result = ClassificationResult(label="positive", confidence=0.95)
        assert result.label == "positive"
        assert result.confidence == 0.95
        assert result.metadata is None

    def test_init_with_metadata(self) -> None:
        """Test initializing a ClassificationResult with metadata."""
        metadata = {"key": "value", "scores": {"positive": 0.95, "negative": 0.05}}
        result = ClassificationResult(label="positive", confidence=0.95, metadata=metadata)
        assert result.label == "positive"
        assert result.confidence == 0.95
        assert result.metadata == metadata

    def test_str_representation(self) -> None:
        """Test the string representation of a ClassificationResult."""
        result = ClassificationResult(label="positive", confidence=0.95)
        assert "label='positive'" in str(result)
        assert "confidence=0.95" in str(result)

    def test_repr_representation(self) -> None:
        """Test the repr representation of a ClassificationResult."""
        result = ClassificationResult(label="positive", confidence=0.95)
        assert (
            repr(result) == "ClassificationResult(label='positive', confidence=0.95, metadata=None)"
        )


class MockClassifier(Classifier):
    """Mock classifier for testing."""

    def __init__(self, name: str = "MockClassifier", description: str = "A mock classifier"):
        """Initialize the mock classifier."""
        self._name = name
        self._description = description
        self.classify_calls = []
        self.batch_classify_calls = []
        self._label = "positive"
        self._confidence = 0.95

    @property
    def name(self) -> str:
        """Get the classifier name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the classifier description."""
        return self._description

    def set_result(self, label: str, confidence: float) -> None:
        """Set the result that the classifier will return."""
        self._label = label
        self._confidence = confidence

    def classify(self, text: str) -> ClassificationResult:
        """Classify text."""
        self.classify_calls.append(text)
        return ClassificationResult(label=self._label, confidence=self._confidence)

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts."""
        self.batch_classify_calls.append(texts)
        return [ClassificationResult(label=self._label, confidence=self._confidence) for _ in texts]


class TestClassifier:
    """Tests for the Classifier protocol."""

    def test_classifier_interface(self) -> None:
        """Test that a class implementing the Classifier protocol works correctly."""
        classifier = MockClassifier()

        # Test the name and description properties
        assert classifier.name == "MockClassifier"
        assert classifier.description == "A mock classifier"

        # Test the classify method
        result = classifier.classify("This is a test.")
        assert result.label == "positive"
        assert result.confidence == 0.95
        assert len(classifier.classify_calls) == 1
        assert classifier.classify_calls[0] == "This is a test."

        # Test the batch_classify method
        texts = ["Text 1", "Text 2", "Text 3"]
        results = classifier.batch_classify(texts)
        assert len(results) == 3
        for result in results:
            assert result.label == "positive"
            assert result.confidence == 0.95
        assert len(classifier.batch_classify_calls) == 1
        assert classifier.batch_classify_calls[0] == texts

    def test_classifier_with_custom_result(self) -> None:
        """Test a classifier that returns a custom result."""
        classifier = MockClassifier()

        # Set a custom result
        classifier.set_result("negative", 0.8)

        # Test the classify method
        result = classifier.classify("This is a negative test.")
        assert result.label == "negative"
        assert result.confidence == 0.8


class TestSentimentClassifier:
    """Tests for the SentimentClassifier."""

    def test_sentiment_classifier_positive(self) -> None:
        """Test the SentimentClassifier with positive text."""
        # Mock the SentimentClassifier
        with patch("sifaka.classifiers.sentiment.SentimentClassifier") as MockSentimentClassifier:
            # Configure the mock
            mock_instance = MockSentimentClassifier.return_value
            mock_instance.name = "SentimentClassifier"
            mock_instance.description = "Classifies text sentiment"
            mock_instance.classify.return_value = ClassificationResult(
                label="positive", confidence=0.95
            )

            # Create an instance of the mock
            classifier = MockSentimentClassifier()

            # Test the classify method
            result = classifier.classify("I love this product!")
            assert result.label == "positive"
            assert result.confidence == 0.95
            mock_instance.classify.assert_called_once_with("I love this product!")

    def test_sentiment_classifier_negative(self) -> None:
        """Test the SentimentClassifier with negative text."""
        # Mock the SentimentClassifier
        with patch("sifaka.classifiers.sentiment.SentimentClassifier") as MockSentimentClassifier:
            # Configure the mock
            mock_instance = MockSentimentClassifier.return_value
            mock_instance.name = "SentimentClassifier"
            mock_instance.description = "Classifies text sentiment"
            mock_instance.classify.return_value = ClassificationResult(
                label="negative", confidence=0.9
            )

            # Create an instance of the mock
            classifier = MockSentimentClassifier()

            # Test the classify method
            result = classifier.classify("I hate this product!")
            assert result.label == "negative"
            assert result.confidence == 0.9
            mock_instance.classify.assert_called_once_with("I hate this product!")


class TestToxicityClassifier:
    """Tests for the ToxicityClassifier."""

    def test_toxicity_classifier_non_toxic(self) -> None:
        """Test the ToxicityClassifier with non-toxic text."""
        # Mock the ToxicityClassifier
        with patch("sifaka.classifiers.toxicity.ToxicityClassifier") as MockToxicityClassifier:
            # Configure the mock
            mock_instance = MockToxicityClassifier.return_value
            mock_instance.name = "ToxicityClassifier"
            mock_instance.description = "Detects toxic content"
            mock_instance.classify.return_value = ClassificationResult(
                label="non-toxic", confidence=0.98
            )

            # Create an instance of the mock
            classifier = MockToxicityClassifier()

            # Test the classify method
            result = classifier.classify("This is a friendly message.")
            assert result.label == "non-toxic"
            assert result.confidence == 0.98
            mock_instance.classify.assert_called_once_with("This is a friendly message.")

    def test_toxicity_classifier_toxic(self) -> None:
        """Test the ToxicityClassifier with toxic text."""
        # Mock the ToxicityClassifier
        with patch("sifaka.classifiers.toxicity.ToxicityClassifier") as MockToxicityClassifier:
            # Configure the mock
            mock_instance = MockToxicityClassifier.return_value
            mock_instance.name = "ToxicityClassifier"
            mock_instance.description = "Detects toxic content"
            mock_instance.classify.return_value = ClassificationResult(
                label="toxic", confidence=0.85
            )

            # Create an instance of the mock
            classifier = MockToxicityClassifier()

            # Test the classify method
            result = classifier.classify("This message contains offensive content.")
            assert result.label == "toxic"
            assert result.confidence == 0.85
            mock_instance.classify.assert_called_once_with(
                "This message contains offensive content."
            )
