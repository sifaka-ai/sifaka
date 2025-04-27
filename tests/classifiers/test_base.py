"""Tests for the base classifier."""

import pytest
from typing import Dict, Any, List, Union
from pydantic import ValidationError

from sifaka.classifiers.base import Classifier, ClassificationResult


class TestClassifier(Classifier):
    """Test implementation of Classifier."""

    def __init__(self, **data):
        data.setdefault("name", "test")
        data.setdefault("description", "test classifier")
        data.setdefault("labels", ["positive", "negative"])
        super().__init__(**data)

    def classify(self, text: str) -> ClassificationResult:
        """Return mock classification result."""
        return ClassificationResult(label="positive", confidence=0.8, metadata={"test": True})

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Return mock batch classification results."""
        return [self.classify(text) for text in texts]


def test_classification_result_initialization():
    """Test ClassificationResult initialization."""
    # Test basic initialization
    result = ClassificationResult(label="test", confidence=0.8, metadata={"key": "value"})
    assert result.label == "test"
    assert result.confidence == 0.8
    assert result.metadata == {"key": "value"}

    # Test different label types
    label_types = ["string_label", 123, 1.23, True]
    for label in label_types:
        result = ClassificationResult(label=label, confidence=0.8)
        assert result.label == label

    # Test confidence bounds
    result = ClassificationResult(label="test", confidence=0.0)
    assert result.confidence == 0.0
    result = ClassificationResult(label="test", confidence=1.0)
    assert result.confidence == 1.0

    # Test validation errors
    with pytest.raises(ValidationError):
        ClassificationResult(label="test", confidence=1.5)
    with pytest.raises(ValidationError):
        ClassificationResult(label="test", confidence=-0.5)


def test_classifier_initialization():
    """Test Classifier initialization."""
    # Test basic initialization
    classifier = TestClassifier()
    assert classifier.name == "test"
    assert classifier.description == "test classifier"
    assert classifier.labels == ["positive", "negative"]
    assert classifier.config == {}
    assert classifier.cache_size == 0
    assert classifier.cost == 1

    # Test custom initialization
    custom_classifier = TestClassifier(
        name="custom",
        description="custom classifier",
        labels=["label1", "label2"],
        config={"param": "value"},
        cache_size=100,
        cost=5,
    )
    assert custom_classifier.name == "custom"
    assert custom_classifier.description == "custom classifier"
    assert custom_classifier.labels == ["label1", "label2"]
    assert custom_classifier.config == {"param": "value"}
    assert custom_classifier.cache_size == 100
    assert custom_classifier.cost == 5

    # Test validation errors
    with pytest.raises(ValidationError):
        TestClassifier(cache_size=-1)
    with pytest.raises(ValidationError):
        TestClassifier(cost=-1)


def test_classifier_classify():
    """Test Classifier classify method."""
    classifier = TestClassifier()

    # Test single classification
    result = classifier.classify("test text")
    assert isinstance(result, ClassificationResult)
    assert result.label == "positive"
    assert result.confidence == 0.8
    assert result.metadata == {"test": True}

    # Test with empty text
    result = classifier.classify("")
    assert isinstance(result, ClassificationResult)

    # Test with special characters
    result = classifier.classify("!@#$%^&*()")
    assert isinstance(result, ClassificationResult)

    # Test with unicode
    result = classifier.classify("Hello 世界")
    assert isinstance(result, ClassificationResult)


def test_classifier_batch_classify():
    """Test Classifier batch_classify method."""
    classifier = TestClassifier()

    # Test batch classification
    texts = ["text1", "text2", "text3"]
    results = classifier.batch_classify(texts)
    assert isinstance(results, list)
    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, ClassificationResult)
        assert result.label == "positive"
        assert result.confidence == 0.8
        assert result.metadata == {"test": True}

    # Test empty batch
    results = classifier.batch_classify([])
    assert isinstance(results, list)
    assert len(results) == 0

    # Test batch with various text types
    mixed_texts = [
        "",  # Empty text
        "normal text",  # Normal text
        "!@#$%^&*()",  # Special characters
        "Hello 世界",  # Unicode
        "\n\t\r",  # Whitespace
    ]
    results = classifier.batch_classify(mixed_texts)
    assert len(results) == len(mixed_texts)
    for result in results:
        assert isinstance(result, ClassificationResult)


def test_classifier_warm_up():
    """Test Classifier warm_up method."""
    classifier = TestClassifier()

    # Test that warm_up can be called without error
    classifier.warm_up()


def test_edge_cases():
    """Test edge cases."""
    classifier = TestClassifier()
    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "unicode": "Hello 世界",
        "newlines": "Line 1\nLine 2\nLine 3",
        "numbers_only": "123 456 789",
        "very_long": "a" * 10000,  # Test with long input
    }

    for case_name, text in edge_cases.items():
        # Test single classification
        result = classifier.classify(text)
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.label, (str, int, float, bool))
        assert 0 <= result.confidence <= 1
        assert isinstance(result.metadata, dict)

        # Test batch classification
        results = classifier.batch_classify([text])
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], ClassificationResult)


def test_error_handling():
    """Test error handling."""
    classifier = TestClassifier()
    invalid_inputs = [None, 123, [], {}]

    for invalid_input in invalid_inputs:
        # Test single classification
        with pytest.raises(Exception):
            classifier.classify(invalid_input)

        # Test batch classification
        with pytest.raises(Exception):
            classifier.batch_classify([invalid_input])


def test_consistent_results():
    """Test consistency of classification results."""
    classifier = TestClassifier()
    test_text = "This is a test text that should give consistent results."

    # Test single classification consistency
    results = [classifier.classify(test_text) for _ in range(3)]
    first_result = results[0]
    for result in results[1:]:
        assert result.label == first_result.label
        assert result.confidence == first_result.confidence
        assert result.metadata == first_result.metadata

    # Test batch classification consistency
    batch_results = [classifier.batch_classify([test_text]) for _ in range(3)]
    first_batch = batch_results[0]
    for batch in batch_results[1:]:
        assert len(batch) == len(first_batch)
        for r1, r2 in zip(batch, first_batch):
            assert r1.label == r2.label
            assert r1.confidence == r2.confidence
            assert r1.metadata == r2.metadata
