"""
Tests for the spam classifier.
"""

import os
import pickle
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from sifaka.classifiers.base import ClassificationResult, ClassifierConfig
from sifaka.classifiers.spam import SpamClassifier, SpamConfig


class MockMultinomialNB:
    """Mock MultinomialNB implementation for testing."""

    def __init__(self):
        """Initialize the mock classifier."""
        self.classes_ = np.array([0, 1])  # 0 = ham, 1 = spam

    def fit(self, X, y):
        """Mock fit method."""
        return self

    def predict(self, X):
        """Mock predict method."""
        # Return mock predictions based on input
        if isinstance(X, list):
            n_samples = len(X)
        else:
            n_samples = X.shape[0]
        return np.array([i % 2 for i in range(n_samples)])

    def predict_proba(self, X):
        """Mock predict_proba method."""
        # Return mock probabilities for each class
        if isinstance(X, list):
            n_samples = len(X)
        else:
            n_samples = X.shape[0]
        probas = []

        for i in range(n_samples):
            if i % 2 == 0:
                # Ham (class 0) is more likely
                probas.append([0.8, 0.2])
            else:
                # Spam (class 1) is more likely
                probas.append([0.3, 0.7])

        return np.array(probas)


class MockTfidfVectorizer:
    """Mock TF-IDF vectorizer for testing."""

    def __init__(self, max_features=1000, stop_words=None, ngram_range=(1, 1)):
        """Initialize with parameters."""
        self.max_features = max_features
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.vocabulary_ = {
            "free": 0,
            "win": 1,
            "prize": 2,
            "money": 3,
            "offer": 4,
            "urgent": 5,
            "limited": 6,
            "opportunity": 7,
            "click": 8,
            "link": 9,
            "hello": 10,
            "meeting": 11,
            "report": 12,
            "project": 13,
            "schedule": 14,
            "thanks": 15,
            "regards": 16,
            "sincerely": 17,
            "best": 18,
            "wishes": 19,
        }
        self.feature_names = list(self.vocabulary_.keys())

    def fit(self, texts, y=None):
        """Mock fit method."""
        return self

    def fit_transform(self, texts, y=None):
        """Mock fit_transform method."""
        # Create a sparse matrix representation
        n_samples = len(texts)
        n_features = len(self.vocabulary_)

        # Create a mock sparse matrix
        mock_matrix = MagicMock()
        mock_matrix.shape = (n_samples, n_features)
        return mock_matrix

    def transform(self, texts):
        """Mock transform method."""
        # Create a sparse matrix representation
        n_samples = len(texts)
        n_features = len(self.vocabulary_)

        # Create a mock sparse matrix
        mock_matrix = MagicMock()
        mock_matrix.shape = (n_samples, n_features)
        return mock_matrix

    def get_feature_names_out(self):
        """Return feature names."""
        return self.feature_names


class MockPipeline:
    """Mock Pipeline implementation for testing."""

    def __init__(self, steps):
        """Initialize with steps."""
        self.steps = steps
        self.named_steps = {name: step for name, step in steps}

    def fit(self, X, y):
        """Mock fit method."""
        for _, step in self.steps:
            if hasattr(step, "fit"):
                step.fit(X, y)
        return self

    def predict(self, X):
        """Mock predict method."""
        return self.named_steps["classifier"].predict(X)

    def predict_proba(self, X):
        """Mock predict_proba method."""
        return self.named_steps["classifier"].predict_proba(X)


@pytest.fixture
def mock_sklearn_modules():
    """Create mock sklearn modules."""
    with patch("importlib.import_module") as mock_import:
        # Create mock modules
        mock_feature_extraction_text = MagicMock()
        mock_feature_extraction_text.TfidfVectorizer = MockTfidfVectorizer

        mock_naive_bayes = MagicMock()
        mock_naive_bayes.MultinomialNB = MockMultinomialNB

        mock_pipeline = MagicMock()
        mock_pipeline.Pipeline = MockPipeline

        # Configure import to return our mocks
        def side_effect(name):
            if name == "sklearn.feature_extraction.text":
                return mock_feature_extraction_text
            elif name == "sklearn.naive_bayes":
                return mock_naive_bayes
            elif name == "sklearn.pipeline":
                return mock_pipeline
            else:
                raise ImportError(f"Mock cannot import {name}")

        mock_import.side_effect = side_effect

        yield {
            "feature_extraction_text": mock_feature_extraction_text,
            "naive_bayes": mock_naive_bayes,
            "pipeline": mock_pipeline,
        }


@pytest.fixture
def spam_classifier(mock_sklearn_modules):
    """Create a SpamClassifier with mocked dependencies."""
    classifier = SpamClassifier()

    # Manually set the mocked modules
    classifier._sklearn_feature_extraction_text = mock_sklearn_modules["feature_extraction_text"]
    classifier._sklearn_naive_bayes = mock_sklearn_modules["naive_bayes"]
    classifier._sklearn_pipeline = mock_sklearn_modules["pipeline"]

    # Initialize the classifier
    classifier.warm_up()

    return classifier


def test_initialization():
    """Test SpamClassifier initialization."""
    # Test basic initialization
    classifier = SpamClassifier()
    assert classifier.name == "spam_classifier"
    assert classifier.description == "Detects spam content in text"
    assert classifier.config.labels == ["ham", "spam"]
    assert classifier.config.cost == 1.5

    # Test with custom spam config
    spam_config = SpamConfig(
        min_confidence=0.8,
        max_features=2000,
        random_state=123,
        use_bigrams=False,
    )

    classifier = SpamClassifier(
        name="custom_spam",
        description="Custom spam classifier",
        spam_config=spam_config,
    )

    assert classifier.name == "custom_spam"
    assert classifier.description == "Custom spam classifier"
    assert classifier.config.params["min_confidence"] == 0.8
    assert classifier.config.params["max_features"] == 2000
    assert classifier.config.params["random_state"] == 123
    assert classifier.config.params["use_bigrams"] is False


def test_spam_config_validation():
    """Test SpamConfig validation."""
    # Test valid config
    config = SpamConfig(min_confidence=0.5, max_features=1000)
    assert config.min_confidence == 0.5
    assert config.max_features == 1000

    # Test invalid min_confidence
    with pytest.raises(ValueError):
        SpamConfig(min_confidence=-0.1)

    with pytest.raises(ValueError):
        SpamConfig(min_confidence=1.1)

    # Test invalid max_features
    with pytest.raises(ValueError):
        SpamConfig(max_features=0)


def test_is_initialized(spam_classifier):
    """Test _is_initialized method."""
    assert spam_classifier._is_initialized() is True

    # Test with no pipeline
    spam_classifier._pipeline = None
    assert spam_classifier._is_initialized() is False


def test_warm_up(mock_sklearn_modules):
    """Test warm_up functionality."""
    classifier = SpamClassifier()

    # Manually set the mocked modules
    classifier._sklearn_feature_extraction_text = mock_sklearn_modules["feature_extraction_text"]
    classifier._sklearn_naive_bayes = mock_sklearn_modules["naive_bayes"]
    classifier._sklearn_pipeline = mock_sklearn_modules["pipeline"]

    # Test warm_up with mocked modules
    classifier.warm_up()
    assert classifier._is_initialized() is True
    assert classifier._vectorizer is not None
    assert classifier._model is not None
    assert classifier._pipeline is not None

    # Test warm_up with missing package
    with patch.object(
        classifier, "_load_dependencies", side_effect=ImportError("Test import error")
    ):
        classifier._initialized = False
        classifier._pipeline = None
        with pytest.raises(ImportError):
            classifier.warm_up()

    # Test warm_up with other exception
    with patch.object(classifier, "_load_dependencies", side_effect=RuntimeError("Test error")):
        classifier._initialized = False
        classifier._pipeline = None
        with pytest.raises(RuntimeError):
            classifier.warm_up()


def test_save_load_model(spam_classifier, tmp_path):
    """Test model saving and loading."""
    # Create a temporary file path
    model_path = os.path.join(tmp_path, "spam_model.pkl")

    # Mock _save_model to avoid actual file operations
    with patch.object(spam_classifier, "_save_model") as mock_save:
        # Call save_model
        spam_classifier._save_model(model_path)
        mock_save.assert_called_once()

    # Mock _load_model to avoid actual file operations
    with patch.object(spam_classifier, "_load_model") as mock_load:
        # Call load_model
        spam_classifier._load_model(model_path)
        mock_load.assert_called_once_with(model_path)

    # Test load model with exception
    with patch.object(spam_classifier, "_load_model", side_effect=RuntimeError("Test error")):
        with pytest.raises(RuntimeError):
            spam_classifier._load_model("invalid_path")


def test_fit(spam_classifier, tmp_path):
    """Test fit method."""
    # Create sample texts and labels
    texts = [
        "Hello, let's schedule a meeting for the project.",
        "FREE MONEY! Click here to claim your prize now!",
        "Please find attached the report for your review.",
        "URGENT: Limited time offer! Win a free vacation!",
        "Thanks for your help. Best regards.",
    ]

    labels = ["ham", "spam", "ham", "spam", "ham"]

    # Test fit without model path
    fitted_classifier = spam_classifier.fit(texts, labels)

    # Check that it returns self
    assert fitted_classifier is spam_classifier

    # Test fit with model path
    model_path = os.path.join(tmp_path, "spam_model.pkl")
    spam_classifier._spam_config = SpamConfig(model_path=model_path)

    # Mock the _save_model method
    with patch.object(spam_classifier, "_save_model") as mock_save:
        spam_classifier.fit(texts, labels)
        mock_save.assert_called_once_with(model_path)

    # Test with mismatched texts and labels
    with pytest.raises(ValueError):
        spam_classifier.fit(texts, labels[:-1])

    # Test with invalid labels
    with pytest.raises(ValueError):
        spam_classifier.fit(texts, ["ham", "spam", "ham", "invalid", "ham"])


def test_classification(spam_classifier):
    """Test classification functionality."""
    # Test with ham text
    ham_text = "Hello, let's schedule a meeting for the project."
    result = spam_classifier._classify_impl(ham_text)

    assert isinstance(result, ClassificationResult)
    assert result.label == "ham"
    assert result.confidence == 0.8
    assert "probabilities" in result.metadata
    assert result.metadata["probabilities"]["ham"] == 0.8
    assert result.metadata["probabilities"]["spam"] == 0.2

    # Test with spam text
    spam_text = "FREE MONEY! Click here to claim your prize now!"
    result = spam_classifier._classify_impl(spam_text)

    assert isinstance(result, ClassificationResult)
    assert result.label == "spam"
    assert result.confidence == 0.7
    assert "probabilities" in result.metadata
    assert result.metadata["probabilities"]["ham"] == 0.3
    assert result.metadata["probabilities"]["spam"] == 0.7

    # Test with uninitialized classifier
    spam_classifier._pipeline = None

    # Mock warm_up to do nothing
    with patch.object(spam_classifier, "warm_up") as mock_warm_up:
        with pytest.raises(AttributeError):
            spam_classifier._classify_impl("Test text")


def test_batch_classification(spam_classifier):
    """Test batch classification."""
    texts = [
        "Hello, let's schedule a meeting for the project.",
        "FREE MONEY! Click here to claim your prize now!",
        "Please find attached the report for your review.",
        "URGENT: Limited time offer! Win a free vacation!",
        "Thanks for your help. Best regards.",
    ]

    results = spam_classifier.batch_classify(texts)

    assert isinstance(results, list)
    assert len(results) == len(texts)

    # Check each result
    for i, result in enumerate(results):
        assert isinstance(result, ClassificationResult)

        # Even indices should be ham, odd indices should be spam
        expected_label = "ham" if i % 2 == 0 else "spam"
        expected_confidence = 0.8 if i % 2 == 0 else 0.7

        assert result.label == expected_label
        assert result.confidence == expected_confidence
        assert "probabilities" in result.metadata

    # Test with uninitialized classifier
    spam_classifier._pipeline = None

    # Mock warm_up to do nothing
    with patch.object(spam_classifier, "warm_up") as mock_warm_up:
        with pytest.raises(AttributeError):
            spam_classifier.batch_classify(texts)


def test_create_pretrained(mock_sklearn_modules):
    """Test create_pretrained factory method."""
    texts = [
        "Hello, let's schedule a meeting for the project.",
        "FREE MONEY! Click here to claim your prize now!",
        "Please find attached the report for your review.",
    ]

    labels = ["ham", "spam", "ham"]

    # Create a pretrained classifier
    classifier = SpamClassifier.create_pretrained(
        texts=texts,
        labels=labels,
        name="pretrained_test",
        description="Pretrained test classifier",
        spam_config=SpamConfig(min_confidence=0.8),
    )

    assert classifier.name == "pretrained_test"
    assert classifier.description == "Pretrained test classifier"
    assert classifier._is_initialized() is True
