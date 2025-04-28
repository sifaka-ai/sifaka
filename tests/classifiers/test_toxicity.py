from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from sifaka.classifiers.base import ClassificationResult
from sifaka.classifiers.toxicity import ToxicityClassifier


class MockDetoxify:
    """Mock Detoxify model for testing."""

    def __init__(self, model_type: str = "original"):
        self.model_type = model_type

    def predict(self, text: str | List[str]) -> Dict[str, Any]:
        if isinstance(text, list):
            return self._batch_predict(text)

        # Simulate toxicity detection based on keywords
        toxic_words = ["hate", "stupid", "idiot", "kill"]
        severe_words = ["kill", "die"]
        obscene_words = ["damn"]
        threat_words = ["kill", "hurt"]
        insult_words = ["stupid", "idiot"]
        identity_words = ["hate"]

        text = text.lower()
        scores = {
            "toxic": 0.1,
            "severe_toxic": 0.1,
            "obscene": 0.1,
            "threat": 0.1,
            "insult": 0.1,
            "identity_hate": 0.1,
        }

        # Increase scores based on toxic words
        for word in text.split():
            if word in toxic_words:
                scores["toxic"] = 0.9
            if word in severe_words:
                scores["severe_toxic"] = 0.9
            if word in obscene_words:
                scores["obscene"] = 0.9
            if word in threat_words:
                scores["threat"] = 0.9
            if word in insult_words:
                scores["insult"] = 0.9
            if word in identity_words:
                scores["identity_hate"] = 0.9

        return scores

    def _batch_predict(self, texts: List[str]) -> Dict[str, List[float]]:
        results = []
        for text in texts:
            results.append(self.predict(text))

        # Convert to batch format
        batch_results = {
            label: [result[label] for result in results] for label in results[0].keys()
        }
        return batch_results


@pytest.fixture
def mock_detoxify():
    return MockDetoxify()


@pytest.fixture
def toxicity_classifier():
    with patch("sifaka.classifiers.toxicity.importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_module.Detoxify = MockDetoxify
        mock_import.return_value = mock_module

        classifier = ToxicityClassifier()
        classifier.warm_up()
        return classifier


def test_initialization():
    """Test classifier initialization."""
    classifier = ToxicityClassifier(
        name="custom_toxicity", description="Custom description", model_name="unbiased"
    )
    assert classifier.name == "custom_toxicity"
    assert classifier.description == "Custom description"
    assert classifier.model_name == "unbiased"
    assert classifier.labels == [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    assert classifier.cost == 2


def test_warm_up(toxicity_classifier):
    """Test warm-up functionality."""
    assert toxicity_classifier._model is not None
    assert isinstance(toxicity_classifier._model, MockDetoxify)

    # Test warm-up with missing package
    with patch("sifaka.classifiers.toxicity.importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError()
        classifier = ToxicityClassifier()
        with pytest.raises(ImportError):
            classifier.warm_up()


def test_classification(toxicity_classifier):
    """Test classification of different text types."""
    # Test non-toxic text
    result = toxicity_classifier.classify("This is a friendly message")
    assert result.label in toxicity_classifier.labels
    assert result.confidence <= 0.2
    assert "all_scores" in result.metadata

    # Test toxic text
    result = toxicity_classifier.classify("I hate you stupid idiot")
    assert result.label in ["toxic", "insult", "identity_hate"]
    assert result.confidence >= 0.8

    # Test severe toxic text
    result = toxicity_classifier.classify("I will kill you")
    assert result.label in ["severe_toxic", "threat"]
    assert result.confidence >= 0.8


def test_batch_classification(toxicity_classifier):
    """Test batch classification."""
    texts = ["Hello friend", "I hate you", "Nice weather today"]
    results = toxicity_classifier.batch_classify(texts)

    assert len(results) == 3
    assert all(isinstance(r, ClassificationResult) for r in results)
    assert results[0].confidence <= 0.2  # non-toxic
    assert results[1].confidence >= 0.8  # toxic
    assert results[2].confidence <= 0.2  # non-toxic


def test_edge_cases(toxicity_classifier):
    """Test edge cases."""
    # Empty string
    result = toxicity_classifier.classify("")
    assert isinstance(result, ClassificationResult)

    # Whitespace
    result = toxicity_classifier.classify("   ")
    assert isinstance(result, ClassificationResult)

    # Special characters
    result = toxicity_classifier.classify("!@#$%^&*()")
    assert isinstance(result, ClassificationResult)

    # Very long text
    long_text = "hello " * 1000
    result = toxicity_classifier.classify(long_text)
    assert isinstance(result, ClassificationResult)


def test_error_handling(toxicity_classifier):
    """Test error handling."""
    # Test classification error
    with patch.object(toxicity_classifier._model, "predict") as mock_predict:
        mock_predict.side_effect = Exception("Model error")
        result = toxicity_classifier.classify("test")
        assert result.label == "unknown"
        assert result.confidence == 0.0
        assert "error" in result.metadata

    # Test batch classification error
    with patch.object(toxicity_classifier._model, "predict") as mock_predict:
        mock_predict.side_effect = Exception("Model error")
        results = toxicity_classifier.batch_classify(["test1", "test2"])
        assert all(r.label == "unknown" for r in results)
        assert all(r.confidence == 0.0 for r in results)
        assert all("error" in r.metadata for r in results)


def test_consistent_results(toxicity_classifier):
    """Test consistency of classification results."""
    text = "I hate you"
    result1 = toxicity_classifier.classify(text)
    result2 = toxicity_classifier.classify(text)

    assert result1.label == result2.label
    assert result1.confidence == result2.confidence
    assert result1.metadata == result2.metadata


def test_model_types():
    """Test different model types."""
    model_types = ["original", "unbiased", "multilingual"]
    for model_type in model_types:
        with patch("sifaka.classifiers.toxicity.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.Detoxify = MockDetoxify
            mock_import.return_value = mock_module

            classifier = ToxicityClassifier(model_name=model_type)
            classifier.warm_up()
            assert classifier.model_name == model_type
            assert isinstance(classifier._model, MockDetoxify)
            assert classifier._model.model_type == model_type
