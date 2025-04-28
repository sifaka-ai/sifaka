"""
Tests for the genre classifier.
"""

import os
import pickle
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from sifaka.classifiers.base import ClassificationResult, ClassifierConfig
from sifaka.classifiers.genre import GenreClassifier, GenreConfig


class MockRandomForestClassifier:
    """Mock RandomForest implementation for testing."""

    def __init__(self, n_estimators=100, random_state=42):
        """Initialize with parameters."""
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def fit(self, X, y):
        """Mock fit method."""
        return self

    def predict(self, X):
        """Mock predict method."""
        # Return mock predictions based on input
        n_samples = X.shape[0]
        return np.array([i % len(self.classes_) for i in range(n_samples)])

    def predict_proba(self, X):
        """Mock predict_proba method."""
        # Return mock probabilities for each class
        n_samples = X.shape[0]
        probas = []

        for i in range(n_samples):
            # Create different probability distributions based on input index
            proba = np.zeros(len(self.classes_))
            dominant_class = i % len(self.classes_)
            proba[dominant_class] = 0.7  # Dominant class

            # Distribute remaining probability
            remaining = 0.3
            for j in range(len(self.classes_)):
                if j != dominant_class:
                    proba[j] = remaining / (len(self.classes_) - 1)

            probas.append(proba)

        return np.array(probas)


class MockTfidfVectorizer:
    """Mock TF-IDF vectorizer for testing."""

    def __init__(self, max_features=2000, stop_words=None, ngram_range=(1, 1)):
        """Initialize with parameters."""
        self.max_features = max_features
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.vocabulary_ = {
            "news": 0,
            "article": 1,
            "headline": 2,
            "report": 3,
            "journalist": 4,
            "fiction": 5,
            "novel": 6,
            "story": 7,
            "character": 8,
            "plot": 9,
            "academic": 10,
            "research": 11,
            "study": 12,
            "theory": 13,
            "hypothesis": 14,
            "technical": 15,
            "manual": 16,
            "guide": 17,
            "instruction": 18,
            "specification": 19,
            "blog": 20,
            "post": 21,
            "blogger": 22,
            "personal": 23,
            "opinion": 24,
            "social": 25,
            "media": 26,
            "tweet": 27,
            "post": 28,
            "share": 29,
            "email": 30,
            "message": 31,
            "sender": 32,
            "recipient": 33,
            "subject": 34,
            "marketing": 35,
            "promotion": 36,
            "advertisement": 37,
            "brand": 38,
            "product": 39,
            "legal": 40,
            "contract": 41,
            "agreement": 42,
            "law": 43,
            "clause": 44,
            "creative": 45,
            "poem": 46,
            "song": 47,
            "artistic": 48,
            "expression": 49,
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

        mock_ensemble = MagicMock()
        mock_ensemble.RandomForestClassifier = MockRandomForestClassifier

        mock_pipeline = MagicMock()
        mock_pipeline.Pipeline = MockPipeline

        # Configure import to return our mocks
        def side_effect(name):
            if name == "sklearn.feature_extraction.text":
                return mock_feature_extraction_text
            elif name == "sklearn.ensemble":
                return mock_ensemble
            elif name == "sklearn.pipeline":
                return mock_pipeline
            else:
                raise ImportError(f"Mock cannot import {name}")

        mock_import.side_effect = side_effect

        yield {
            "feature_extraction_text": mock_feature_extraction_text,
            "ensemble": mock_ensemble,
            "pipeline": mock_pipeline,
        }


@pytest.fixture
def genre_classifier(mock_sklearn_modules):
    """Create a GenreClassifier with mocked dependencies."""
    # Create a genre config
    genre_config = GenreConfig(
        min_confidence=0.6, max_features=2000, random_state=42, use_ngrams=True, n_estimators=100
    )

    # Create the classifier with the config
    config = ClassifierConfig(
        labels=genre_config.default_genres,
        cost=2.0,
        min_confidence=genre_config.min_confidence,
        params={
            "min_confidence": genre_config.min_confidence,
            "max_features": genre_config.max_features,
            "random_state": genre_config.random_state,
            "use_ngrams": genre_config.use_ngrams,
            "n_estimators": genre_config.n_estimators,
        },
    )

    classifier = GenreClassifier(
        name="genre_classifier",
        description="Categorizes text into different genres",
        genre_config=genre_config,
        config=config,
    )

    # Manually set the mocked modules
    classifier.sklearn_feature_extraction_text = mock_sklearn_modules["feature_extraction_text"]
    classifier.sklearn_ensemble = mock_sklearn_modules["ensemble"]
    classifier.sklearn_pipeline = mock_sklearn_modules["pipeline"]

    # Initialize the classifier
    classifier.warm_up()

    # Create mock feature importances
    classifier.feature_importances = {
        "news": {"headline": 0.8, "article": 0.7, "report": 0.6},
        "fiction": {"novel": 0.8, "story": 0.7, "character": 0.6},
        "academic": {"research": 0.8, "study": 0.7, "theory": 0.6},
        "technical": {"manual": 0.8, "guide": 0.7, "instruction": 0.6},
        "blog": {"post": 0.8, "blogger": 0.7, "personal": 0.6},
        "social_media": {"tweet": 0.8, "share": 0.7, "social": 0.6},
        "email": {"message": 0.8, "sender": 0.7, "recipient": 0.6},
        "marketing": {"promotion": 0.8, "advertisement": 0.7, "brand": 0.6},
        "legal": {"contract": 0.8, "agreement": 0.7, "law": 0.6},
        "creative": {"poem": 0.8, "song": 0.7, "artistic": 0.6},
    }

    return classifier


def test_initialization():
    """Test GenreClassifier initialization."""
    # Test basic initialization
    classifier = GenreClassifier()
    assert classifier.name == "genre_classifier"
    assert classifier.description == "Categorizes text into different genres"
    assert len(classifier.config.labels) == 10  # Default genres
    assert classifier.config.cost == 2.0

    # Test with custom genre config
    genre_config = GenreConfig(
        min_confidence=0.8,
        max_features=3000,
        random_state=123,
        use_ngrams=False,
        n_estimators=200,
    )

    classifier = GenreClassifier(
        name="custom_genre",
        description="Custom genre classifier",
        genre_config=genre_config,
    )

    assert classifier.name == "custom_genre"
    assert classifier.description == "Custom genre classifier"
    assert classifier.genre_config.min_confidence == 0.8
    assert classifier.genre_config.max_features == 3000
    assert classifier.genre_config.random_state == 123
    assert classifier.genre_config.use_ngrams is False
    assert classifier.genre_config.n_estimators == 200


def test_genre_config_validation():
    """Test GenreConfig validation."""
    # Test valid config
    config = GenreConfig(min_confidence=0.5, max_features=1000)
    assert config.min_confidence == 0.5
    assert config.max_features == 1000

    # Test invalid min_confidence
    with pytest.raises(ValueError):
        GenreConfig(min_confidence=-0.1)

    with pytest.raises(ValueError):
        GenreConfig(min_confidence=1.1)

    # Test invalid max_features
    with pytest.raises(ValueError):
        GenreConfig(max_features=0)

    # Test invalid n_estimators
    with pytest.raises(ValueError):
        GenreConfig(n_estimators=0)

    # Test invalid default_genres
    with pytest.raises(ValueError):
        GenreConfig(default_genres=[])


def test_warm_up(mock_sklearn_modules):
    """Test warm_up functionality."""
    classifier = GenreClassifier()

    # Test warm_up with mocked modules
    classifier.warm_up()
    assert classifier.initialized is True
    assert classifier.vectorizer is not None
    assert classifier.model is not None
    assert classifier.pipeline is not None

    # Test warm_up with missing package
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError()
        classifier = GenreClassifier()
        with pytest.raises(ImportError):
            classifier.warm_up()

    # Test warm_up with other exception
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = RuntimeError("Test error")
        classifier = GenreClassifier()
        with pytest.raises(RuntimeError):
            classifier.warm_up()


def test_fit(genre_classifier, tmp_path):
    """Test fit method."""
    # Create sample texts and labels
    texts = [
        "This is a news article about current events",
        "A fictional story with interesting characters",
        "An academic research paper on a scientific topic",
        "Technical manual for operating machinery",
        "Blog post about personal experiences",
        "Social media update about daily life",
        "Email message regarding business matters",
        "Marketing promotion for a new product",
        "Legal contract between two parties",
        "Creative poem expressing emotions",
    ]

    labels = [
        "news",
        "fiction",
        "academic",
        "technical",
        "blog",
        "social_media",
        "email",
        "marketing",
        "legal",
        "creative",
    ]

    # Test fit without model path
    fitted_classifier = genre_classifier.fit(texts, labels)

    # Check that it returns self
    assert fitted_classifier is genre_classifier

    # Check that feature importances are extracted
    assert genre_classifier.feature_importances is not None

    # Test fit with model path
    model_path = os.path.join(tmp_path, "genre_model.pkl")
    genre_classifier.genre_config = GenreConfig(model_path=model_path)

    # Mock the _save_model method
    with patch.object(genre_classifier, "_save_model") as mock_save:
        genre_classifier.fit(texts, labels)
        mock_save.assert_called_once_with(model_path)

    # Test with mismatched texts and labels
    with pytest.raises(ValueError):
        genre_classifier.fit(texts, labels[:-1])


def test_save_load_model(genre_classifier, tmp_path):
    """Test model saving and loading."""
    # Create a temporary file path
    model_path = os.path.join(tmp_path, "genre_model.pkl")

    # Mock pickle.dump to avoid actual file operations
    with patch("pickle.dump") as mock_dump:
        genre_classifier._save_model(model_path)
        mock_dump.assert_called_once()

    # Mock pickle.load to return mock data
    mock_data = {
        "pipeline": genre_classifier.pipeline,
        "labels": ["news", "fiction"],
        "feature_importances": {"news": {"headline": 0.8}},
    }

    with patch("pickle.load", return_value=mock_data):
        with patch("builtins.open", create=True):
            genre_classifier._load_model(model_path)
            assert genre_classifier.custom_labels == ["news", "fiction"]
            assert genre_classifier.feature_importances == {"news": {"headline": 0.8}}

    # Test load model with exception
    with patch("builtins.open", side_effect=Exception("Test error")):
        with pytest.raises(RuntimeError):
            genre_classifier._load_model("invalid_path")


def test_classification(genre_classifier):
    """Test classification functionality."""
    # Test with a news text
    result = genre_classifier._classify_impl("This is a news article with headlines")

    assert isinstance(result, ClassificationResult)
    assert result.label in genre_classifier.config.labels
    assert 0 <= result.confidence <= 1
    assert "probabilities" in result.metadata
    assert "threshold" in result.metadata
    assert "is_confident" in result.metadata
    assert "top_features" in result.metadata

    # Check metadata structure
    assert len(result.metadata["probabilities"]) == len(genre_classifier.config.labels)
    for genre, probability in result.metadata["probabilities"].items():
        assert genre in genre_classifier.config.labels
        assert 0 <= probability <= 1


def test_batch_classification(genre_classifier):
    """Test batch classification."""
    texts = [
        "News article about current events",
        "Fiction story with characters",
        "Academic research paper",
        "Technical manual for machinery",
        "Blog post about experiences",
        "Social media update",
        "Email message",
        "Marketing promotion",
        "Legal contract",
        "Creative poem",
    ]

    results = genre_classifier.batch_classify(texts)

    assert isinstance(results, list)
    assert len(results) == len(texts)

    # Check each result
    for i, result in enumerate(results):
        assert isinstance(result, ClassificationResult)
        assert result.label in genre_classifier.config.labels
        assert 0 <= result.confidence <= 1
        assert "probabilities" in result.metadata
        assert "threshold" in result.metadata
        assert "is_confident" in result.metadata


def test_error_handling(genre_classifier):
    """Test error handling."""
    # Test with invalid input types
    invalid_inputs = [None, 123, [], {}]

    for invalid_input in invalid_inputs:
        with pytest.raises(Exception):
            genre_classifier.classify(invalid_input)

        with pytest.raises(Exception):
            genre_classifier.batch_classify([invalid_input])

    # Test classification without initialization
    uninit_classifier = GenreClassifier()
    uninit_classifier.initialized = False
    uninit_classifier.pipeline = None

    with pytest.raises(RuntimeError):
        uninit_classifier.classify("This should fail")

    with pytest.raises(RuntimeError):
        uninit_classifier.batch_classify(["This should fail"])


def test_extract_feature_importances(genre_classifier):
    """Test feature importance extraction."""
    # Test with valid model
    importances = genre_classifier._extract_feature_importances()
    assert isinstance(importances, dict)

    # Test with no model
    genre_classifier.model = None
    importances = genre_classifier._extract_feature_importances()
    assert importances == {}

    # Test with exception
    genre_classifier.model = MagicMock()
    genre_classifier.model.feature_importances_ = np.array([0.1, 0.2])
    genre_classifier.vectorizer.get_feature_names_out = MagicMock(
        side_effect=Exception("Test error")
    )
    importances = genre_classifier._extract_feature_importances()
    assert importances == {}


def test_create_pretrained(mock_sklearn_modules):
    """Test create_pretrained factory method."""
    texts = [
        "News article about current events",
        "Fiction story with characters",
        "Academic research paper",
    ]

    labels = ["news", "fiction", "academic"]

    # Create a pretrained classifier
    classifier = GenreClassifier.create_pretrained(
        texts=texts,
        labels=labels,
        name="pretrained_test",
        description="Pretrained test classifier",
        genre_config=GenreConfig(min_confidence=0.8),
    )

    assert classifier.name == "pretrained_test"
    assert classifier.description == "Pretrained test classifier"
    assert classifier.genre_config.min_confidence == 0.8
    assert classifier.initialized is True
