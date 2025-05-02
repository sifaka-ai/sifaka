"""Test module for the spam classifier."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from typing import Dict, Any, List

# Import from module under test - these imports will be handled by the patched import
from sifaka.classifiers.spam import (
    SpamClassifier,
    create_spam_classifier,
)
from sifaka.classifiers.base import ClassificationResult


class MockSpamModel:
    """Mock implementation of Naive Bayes model for testing."""

    def __init__(self):
        """Initialize with model type."""
        self.call_count = 0

    def predict_proba(self, texts):
        """Mock prediction method that returns predefined probabilities."""
        self.call_count += 1

        if isinstance(texts, list) and len(texts) > 1:
            # Handle batch prediction
            batch_size = len(texts)
            results = []
            for text in texts:
                if "spam" in text.lower() or "offer" in text.lower() or "free" in text.lower():
                    results.append([0.2, 0.8])  # Likely spam
                else:
                    results.append([0.9, 0.1])  # Likely ham
            return np.array(results)
        else:
            # Single text prediction
            text = texts[0] if isinstance(texts, list) else texts
            if "spam" in text.lower() or "offer" in text.lower() or "free" in text.lower():
                return np.array([[0.2, 0.8]])  # Likely spam
            else:
                return np.array([[0.9, 0.1]])  # Likely ham


@pytest.fixture
def mock_pipeline():
    """Fixture for a mock pipeline."""
    pipeline = MagicMock()
    pipeline.predict_proba.side_effect = MockSpamModel().predict_proba
    return pipeline


@patch("importlib.import_module")
def test_init(mock_import):
    """Test initialization of spam classifier."""
    classifier = SpamClassifier()

    assert classifier.name == "spam_classifier"
    assert classifier.description == "Detects spam content in text"
    assert hasattr(classifier.config, "labels")
    assert hasattr(classifier.config, "cost")

    # Check default labels and cost
    assert classifier.config.labels == ["ham", "spam"]
    assert classifier.config.cost == 1.5


@patch("importlib.import_module")
def test_custom_init(mock_import):
    """Test initialization with custom parameters."""
    classifier = SpamClassifier(
        name="custom_spam",
        description="Custom spam detector",
        config=None,
        max_features=2000,
        use_bigrams=False,
        model_path="custom_model.pkl"
    )

    assert classifier.name == "custom_spam"
    assert classifier.description == "Custom spam detector"
    assert classifier.config.params["max_features"] == 2000
    assert classifier.config.params["use_bigrams"] is False
    assert classifier.config.params["model_path"] == "custom_model.pkl"


@patch("importlib.import_module")
@patch("os.path.exists", return_value=False)
def test_warm_up(mock_exists, mock_import):
    """Test warm_up method initializes the model."""
    # Create mocks for scikit-learn dependencies
    mock_tfidf = MagicMock()
    mock_naive_bayes = MagicMock()
    mock_pipeline = MagicMock()

    # Create mock modules
    mock_feature_extraction = MagicMock()
    mock_feature_extraction.TfidfVectorizer = mock_tfidf

    mock_nb_module = MagicMock()
    mock_nb_module.MultinomialNB = mock_naive_bayes

    mock_pipeline_module = MagicMock()
    mock_pipeline_module.Pipeline = mock_pipeline

    # Set up import side effects
    mock_import.side_effect = lambda name: {
        "sklearn.feature_extraction.text": mock_feature_extraction,
        "sklearn.naive_bayes": mock_nb_module,
        "sklearn.pipeline": mock_pipeline_module
    }.get(name, MagicMock())

    # Create classifier
    classifier = SpamClassifier(use_bigrams=True, max_features=1000)

    # Initialize model
    classifier.warm_up()

    # Check if dependencies are loaded
    assert classifier._initialized is True
    assert classifier._sklearn_feature_extraction_text is not None
    assert classifier._sklearn_naive_bayes is not None
    assert classifier._sklearn_pipeline is not None

    # Check if vectorizer and model are created
    mock_tfidf.assert_called_once()
    mock_naive_bayes.assert_called_once()
    mock_pipeline.assert_called_once()


@patch("importlib.import_module", side_effect=ImportError("No scikit-learn"))
def test_load_dependencies_import_error(mock_import):
    """Test _load_dependencies raises ImportError when scikit-learn is not available."""
    classifier = SpamClassifier()
    with pytest.raises(ImportError) as context:
        classifier._load_dependencies()
    assert "scikit-learn is required" in str(context.value)


def test_classify_with_mock_pipeline(mock_pipeline):
    """Test classification with mock pipeline."""
    classifier = SpamClassifier()
    classifier._initialized = True
    classifier._pipeline = mock_pipeline

    # Test ham text
    result = classifier._classify_impl_uncached("Hello, let's meet for coffee.")
    assert result.label == "ham"
    assert result.confidence > 0.8
    assert "probabilities" in result.metadata

    # Test spam text
    result = classifier._classify_impl_uncached("FREE OFFER! Limited time only!")
    assert result.label == "spam"
    assert result.confidence > 0.7
    assert "probabilities" in result.metadata


def test_batch_classify_with_mock_pipeline(mock_pipeline):
    """Test batch classification with mock pipeline."""
    classifier = SpamClassifier()
    classifier._initialized = True
    classifier._pipeline = mock_pipeline

    texts = [
        "Hello, let's meet for coffee.",
        "FREE OFFER! Limited time only!",
        "Please review the document I sent yesterday."
    ]

    results = classifier.batch_classify(texts)

    assert len(results) == 3
    assert results[0].label == "ham"
    assert results[1].label == "spam"
    assert results[2].label == "ham"

    assert all(isinstance(r, ClassificationResult) for r in results)
    assert all("probabilities" in r.metadata for r in results)


@patch("importlib.import_module")
@patch("os.path.exists", return_value=False)
def test_create_spam_classifier_factory(mock_exists, mock_import):
    """Test the factory function for creating spam classifier."""
    classifier = create_spam_classifier(
        name="factory_spam",
        description="Factory created classifier",
        model_path="model.pkl",
        max_features=2000,
        use_bigrams=False,
        cache_size=100,
        cost=2.0
    )

    assert classifier.name == "factory_spam"
    assert classifier.description == "Factory created classifier"
    assert classifier.config.cache_size == 100
    assert classifier.config.cost == 2.0
    assert classifier.config.params["model_path"] == "model.pkl"
    assert classifier.config.params["max_features"] == 2000
    assert classifier.config.params["use_bigrams"] is False


@patch("importlib.import_module")
def test_fit(mock_import):
    """Test fit method trains the model."""
    # Create mocks
    mock_pipeline = MagicMock()

    # Create classifier
    classifier = SpamClassifier()
    classifier._initialized = True
    classifier._pipeline = mock_pipeline
    classifier.config.params["model_path"] = None

    # Training data
    texts = [
        "Hello, this is a normal email.",
        "CLICK HERE for a FREE OFFER!",
        "Let's meet for coffee tomorrow.",
        "You've won a million dollars! Claim now!"
    ]
    labels = ["ham", "spam", "ham", "spam"]

    # Train the model
    result = classifier.fit(texts, labels)

    # Check return value
    assert result is classifier

    # Check that fit was called on the pipeline
    mock_pipeline.fit.assert_called_once()

    # Check label conversion
    args = mock_pipeline.fit.call_args[0]
    assert args[0] == texts
    assert args[1] == [0, 1, 0, 1]  # Converted to indices


@patch("importlib.import_module")
def test_fit_with_invalid_data(mock_import):
    """Test fit method raises ValueError with invalid data."""
    classifier = SpamClassifier()

    # Mismatched lengths
    with pytest.raises(ValueError):
        classifier.fit(["text1", "text2"], ["ham"])

    # Invalid labels
    with pytest.raises(ValueError):
        classifier.fit(["text1", "text2"], ["ham", "invalid"])


@patch("importlib.import_module")
def test_create_pretrained(mock_import):
    """Test create_pretrained class method."""
    # Create a spy on the fit method using a custom subclass
    original_fit = SpamClassifier.fit
    fit_called = [False]
    fit_args = [None, None]

    def spy_fit(self, texts, labels):
        fit_called[0] = True
        fit_args[0] = texts
        fit_args[1] = labels
        return original_fit(self, texts, labels)

    # Replace the method temporarily
    SpamClassifier.fit = spy_fit

    try:
        # Training data
        texts = [
            "Hello, this is a normal email.",
            "CLICK HERE for a FREE OFFER!",
            "Let's meet for coffee tomorrow.",
            "You've won a million dollars! Claim now!"
        ]
        labels = ["ham", "spam", "ham", "spam"]

        # Create pretrained classifier
        classifier = SpamClassifier.create_pretrained(
            texts=texts,
            labels=labels,
            name="pretrained",
            description="Pretrained classifier"
        )

        # Check that the classifier was created
        assert isinstance(classifier, SpamClassifier)
        assert classifier.name == "pretrained"  # Now this should pass
        assert classifier.description == "Pretrained classifier"

        # Check that fit was called with the correct arguments
        assert fit_called[0] is True
        assert fit_args[0] == texts
        assert fit_args[1] == labels
    finally:
        # Restore the original method
        SpamClassifier.fit = original_fit


@patch("importlib.import_module")
@patch("pickle.dump")
def test_save_model(mock_dump, mock_import):
    """Test _save_model method."""
    classifier = SpamClassifier()
    classifier._initialized = True
    classifier._pipeline = MagicMock()

    classifier._save_model("model.pkl")

    # Check that pickle.dump was called
    mock_dump.assert_called_once()
    assert mock_dump.call_args[0][0] is classifier._pipeline


@patch("importlib.import_module")
@patch("pickle.load", return_value=MagicMock())
@patch("builtins.open", new_callable=MagicMock)
def test_load_model(mock_open, mock_load, mock_import):
    """Test _load_model method."""
    # Create a mock pipeline with named_steps
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {
        "vectorizer": MagicMock(),
        "classifier": MagicMock()
    }
    mock_load.return_value = mock_pipeline

    classifier = SpamClassifier()
    classifier._load_model("model.pkl")

    # Check that the model was loaded
    assert classifier._pipeline is mock_pipeline
    assert classifier._vectorizer is mock_pipeline.named_steps["vectorizer"]
    assert classifier._model is mock_pipeline.named_steps["classifier"]


@patch("importlib.import_module")
@patch("pickle.load", side_effect=Exception("Error loading model"))
@patch("builtins.open", new_callable=MagicMock)
def test_load_model_error(mock_open, mock_load, mock_import):
    """Test _load_model method with error."""
    classifier = SpamClassifier()

    with pytest.raises(RuntimeError):
        classifier._load_model("model.pkl")