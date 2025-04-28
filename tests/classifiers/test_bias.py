"""
Tests for the bias detector.
"""

import os
import pickle
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from sifaka.classifiers.base import ClassificationResult, ClassifierConfig
from sifaka.classifiers.bias import BiasDetector, BiasConfig


class MockSVC:
    """Mock SVC implementation for testing."""

    def __init__(self, kernel="linear", probability=True, random_state=42, **kwargs):
        """Initialize with parameters."""
        self.kernel = kernel
        self.probability = probability
        self.random_state = random_state
        self.coef_ = np.array(
            [
                [0.1, 0.5, 0.2, 0.8, 0.3, 0.1, 0.9, 0.4, 0.2, 0.1],  # Gender bias
                [0.8, 0.1, 0.1, 0.2, 0.7, 0.9, 0.1, 0.3, 0.5, 0.6],  # Racial bias
                [0.3, 0.2, 0.8, 0.1, 0.1, 0.2, 0.3, 0.7, 0.6, 0.4],  # Political bias
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],  # Age bias
                [0.5, 0.6, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.4, 0.8],  # Socioeconomic bias
                [0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4],  # Religious bias
                [0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # Cultural bias
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3, 0.4, 0.5],  # Educational bias
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2],  # Geographical bias
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Neutral
            ]
        )

    def fit(self, X, y):
        """Mock fit method."""
        return self

    def predict(self, X):
        """Mock predict method."""
        # Return mock predictions based on input
        n_samples = X.shape[0]
        return np.array([i % 10 for i in range(n_samples)])

    def predict_proba(self, X):
        """Mock predict_proba method."""
        # Return mock probabilities for each class
        n_samples = X.shape[0]
        probas = []

        for i in range(n_samples):
            # Create different probability distributions based on input index
            proba = np.zeros(10)  # 10 bias types
            dominant_class = i % 10
            proba[dominant_class] = 0.7  # Dominant class

            # Distribute remaining probability
            remaining = 0.3
            for j in range(10):
                if j != dominant_class:
                    proba[j] = remaining / 9

            probas.append(proba)

        return np.array(probas)


class MockLinearSVC:
    """Mock LinearSVC implementation for testing."""

    def __init__(self, random_state=42, class_weight=None, max_iter=1000):
        """Initialize with parameters."""
        self.random_state = random_state
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.coef_ = np.array(
            [
                [0.1, 0.5, 0.2, 0.8, 0.3, 0.1, 0.9, 0.4, 0.2, 0.1],  # Gender bias
                [0.8, 0.1, 0.1, 0.2, 0.7, 0.9, 0.1, 0.3, 0.5, 0.6],  # Racial bias
                [0.3, 0.2, 0.8, 0.1, 0.1, 0.2, 0.3, 0.7, 0.6, 0.4],  # Political bias
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],  # Age bias
                [0.5, 0.6, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.4, 0.8],  # Socioeconomic bias
                [0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4],  # Religious bias
                [0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # Cultural bias
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3, 0.4, 0.5],  # Educational bias
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2],  # Geographical bias
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Neutral
            ]
        )

    def fit(self, X, y):
        """Mock fit method."""
        return self

    def predict(self, X):
        """Mock predict method."""
        # Return mock predictions based on input
        n_samples = X.shape[0]
        return np.array([i % 10 for i in range(n_samples)])


class MockCalibratedClassifierCV:
    """Mock CalibratedClassifierCV implementation for testing."""

    def __init__(self, estimator, cv=3):
        """Initialize with parameters."""
        self.estimator = estimator
        self.cv = cv
        self.base_estimator = estimator

    def fit(self, X, y):
        """Mock fit method."""
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        """Mock predict method."""
        return self.estimator.predict(X)

    def predict_proba(self, X):
        """Mock predict_proba method."""
        # Return mock probabilities for each class
        n_samples = X.shape[0]
        probas = []

        for i in range(n_samples):
            # Create different probability distributions based on input index
            proba = np.zeros(10)  # 10 bias types
            dominant_class = i % 10
            proba[dominant_class] = 0.7  # Dominant class

            # Distribute remaining probability
            remaining = 0.3
            for j in range(10):
                if j != dominant_class:
                    proba[j] = remaining / 9

            probas.append(proba)

        return np.array(probas)


class MockTfidfVectorizer:
    """Mock TF-IDF vectorizer for testing."""

    def __init__(self, max_features=3000, stop_words=None, ngram_range=(1, 2)):
        """Initialize with parameters."""
        self.max_features = max_features
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.vocabulary_ = {
            "man": 0,
            "woman": 1,
            "male": 2,
            "female": 3,
            "gender": 4,
            "race": 5,
            "ethnic": 6,
            "minority": 7,
            "white": 8,
            "black": 9,
            "conservative": 10,
            "liberal": 11,
            "right": 12,
            "left": 13,
            "political": 14,
            "young": 15,
            "old": 16,
            "elderly": 17,
            "teen": 18,
            "age": 19,
            "rich": 20,
            "poor": 21,
            "wealth": 22,
            "poverty": 23,
            "class": 24,
            "christian": 25,
            "muslim": 26,
            "jewish": 27,
            "hindu": 28,
            "religion": 29,
            "western": 30,
            "eastern": 31,
            "traditional": 32,
            "modern": 33,
            "culture": 34,
            "educated": 35,
            "uneducated": 36,
            "academic": 37,
            "school": 38,
            "education": 39,
            "urban": 40,
            "rural": 41,
            "city": 42,
            "country": 43,
            "region": 44,
            "neutral": 45,
            "unbiased": 46,
            "fair": 47,
            "balanced": 48,
            "objective": 49,
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

        mock_svm = MagicMock()
        mock_svm.SVC = MockSVC
        mock_svm.LinearSVC = MockLinearSVC

        mock_pipeline = MagicMock()
        mock_pipeline.Pipeline = MockPipeline

        mock_calibration = MagicMock()
        mock_calibration.CalibratedClassifierCV = MockCalibratedClassifierCV

        # Configure import to return our mocks
        def side_effect(name):
            if name == "sklearn.feature_extraction.text":
                return mock_feature_extraction_text
            elif name == "sklearn.svm":
                return mock_svm
            elif name == "sklearn.pipeline":
                return mock_pipeline
            elif name == "sklearn.calibration":
                return mock_calibration
            else:
                raise ImportError(f"Mock cannot import {name}")

        mock_import.side_effect = side_effect

        yield {
            "feature_extraction_text": mock_feature_extraction_text,
            "svm": mock_svm,
            "pipeline": mock_pipeline,
            "calibration": mock_calibration,
        }


@pytest.fixture
def bias_detector(mock_sklearn_modules):
    """Create a BiasDetector with mocked dependencies."""
    # Create a config with the required labels
    config = ClassifierConfig(
        labels=[
            "gender",
            "racial",
            "political",
            "age",
            "socioeconomic",
            "religious",
            "cultural",
            "educational",
            "geographical",
            "neutral",
        ],
        cost=2.0,
        min_confidence=0.6,
    )

    detector = BiasDetector(
        name="bias_detector", description="Detects various forms of bias in text", config=config
    )

    # Manually set the mocked modules
    detector._sklearn_feature_extraction_text = mock_sklearn_modules["feature_extraction_text"]
    detector._sklearn_svm = mock_sklearn_modules["svm"]
    detector._sklearn_pipeline = mock_sklearn_modules["pipeline"]
    detector._sklearn_calibration = mock_sklearn_modules["calibration"]

    # Initialize the detector
    detector.warm_up()

    # Create mock explanations
    detector._explanations = {
        "gender": {
            "positive": {"man": 0.8, "woman": 0.7, "male": 0.6},
            "negative": {"neutral": -0.8, "fair": -0.7, "balanced": -0.6},
        },
        "racial": {
            "positive": {"race": 0.8, "ethnic": 0.7, "minority": 0.6},
            "negative": {"neutral": -0.8, "fair": -0.7, "balanced": -0.6},
        },
        "political": {
            "positive": {"conservative": 0.8, "liberal": 0.7, "political": 0.6},
            "negative": {"neutral": -0.8, "fair": -0.7, "balanced": -0.6},
        },
        "neutral": {
            "positive": {"neutral": 0.8, "fair": 0.7, "balanced": 0.6},
            "negative": {"bias": -0.8, "prejudice": -0.7, "stereotype": -0.6},
        },
    }

    # Set initialized flag
    detector._initialized = True

    return detector


def test_initialization():
    """Test BiasDetector initialization."""
    # Test basic initialization
    detector = BiasDetector()
    assert detector.name == "bias_detector"
    assert detector.description == "Detects various forms of bias in text"
    assert detector.config.labels == [
        "gender",
        "racial",
        "political",
        "age",
        "socioeconomic",
        "religious",
        "cultural",
        "educational",
        "geographical",
        "neutral",
    ]

    # Test with custom config
    config = ClassifierConfig(
        labels=["gender", "racial", "neutral"],
        min_confidence=0.8,
        cost=3.0,
    )

    detector = BiasDetector(
        name="custom_bias",
        description="Custom bias detector",
        config=config,
    )

    assert detector.name == "custom_bias"
    assert detector.description == "Custom bias detector"
    assert detector.config.labels == ["gender", "racial", "neutral"]
    assert detector.config.min_confidence == 0.8
    assert detector.config.cost == 3.0


def test_bias_config_validation():
    """Test BiasConfig validation."""
    # Test valid config
    config = BiasConfig(min_confidence=0.5, max_features=2000)
    assert config.min_confidence == 0.5
    assert config.max_features == 2000

    # Test invalid min_confidence
    with pytest.raises(ValueError):
        BiasConfig(min_confidence=-0.1)

    with pytest.raises(ValueError):
        BiasConfig(min_confidence=1.1)

    # Test invalid max_features
    with pytest.raises(ValueError):
        BiasConfig(max_features=0)

    # Test invalid bias_types
    with pytest.raises(ValueError):
        BiasConfig(bias_types=[])

    # Test missing neutral in bias_types
    with pytest.raises(ValueError):
        BiasConfig(bias_types=["gender", "racial"])


def test_warm_up(mock_sklearn_modules):
    """Test warm_up functionality."""
    detector = BiasDetector()

    # Test warm_up with mocked modules
    detector.warm_up()
    assert detector._initialized is True
    assert detector._vectorizer is not None
    assert detector._model is not None
    assert detector._pipeline is not None

    # Test warm_up with missing package
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError()
        detector = BiasDetector()
        with pytest.raises(ImportError):
            detector.warm_up()

    # Test warm_up with other exception
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = RuntimeError("Test error")
        detector = BiasDetector()
        with pytest.raises(RuntimeError):
            detector.warm_up()


def test_extract_bias_features(bias_detector):
    """Test bias feature extraction."""
    # Test with gender-biased text
    text = "Men are better at math than women."
    features = bias_detector._extract_bias_features(text)

    assert isinstance(features, dict)
    assert "bias_gender" in features
    assert features["bias_gender"] > 0

    # Test with racial-biased text
    text = "Ethnic minorities face discrimination in the workplace."
    features = bias_detector._extract_bias_features(text)

    assert "bias_racial" in features
    assert features["bias_racial"] > 0

    # Test with neutral text
    text = "The sky is blue and the grass is green."
    features = bias_detector._extract_bias_features(text)

    # All bias features should be low
    assert all(value < 0.5 for value in features.values())


def test_save_load_model(bias_detector, tmp_path):
    """Test model saving and loading."""
    # Create a temporary file path
    model_path = os.path.join(tmp_path, "bias_model.pkl")

    # Mock pickle.dump to avoid actual file operations
    with patch("pickle.dump") as mock_dump:
        bias_detector._save_model(model_path)
        mock_dump.assert_called_once()

    # Test save model with uninitialized model
    bias_detector._initialized = False
    with pytest.raises(RuntimeError):
        bias_detector._save_model(model_path)
    bias_detector._initialized = True

    # Mock pickle.load to return mock data
    mock_data = {
        "vectorizer": bias_detector._vectorizer,
        "model": bias_detector._model,
        "pipeline": bias_detector._pipeline,
        "bias_config": bias_detector._bias_config,
    }

    with patch("pickle.load", return_value=mock_data):
        with patch("builtins.open", create=True):
            bias_detector._load_model(model_path)
            assert bias_detector._initialized is True

    # Test load model with exception
    with patch("builtins.open", side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            bias_detector._load_model("invalid_path")


def test_fit(bias_detector, tmp_path):
    """Test fit method."""
    # Create sample texts and labels
    texts = [
        "Men are better at math than women.",
        "Ethnic minorities face discrimination in the workplace.",
        "Conservatives are more patriotic than liberals.",
        "Young people are lazy and entitled.",
        "Rich people work harder than poor people.",
        "Christians are more moral than atheists.",
        "Western culture is superior to Eastern culture.",
        "College-educated people are smarter than those without degrees.",
        "Urban areas are more cultured than rural areas.",
        "This is a neutral and unbiased statement about facts.",
    ]

    labels = [
        "gender",
        "racial",
        "political",
        "age",
        "socioeconomic",
        "religious",
        "cultural",
        "educational",
        "geographical",
        "neutral",
    ]

    # Test fit without model path
    fitted_detector = bias_detector.fit(texts, labels)

    # Check that it returns self
    assert fitted_detector is bias_detector

    # Test fit with model path
    model_path = os.path.join(tmp_path, "bias_model.pkl")
    bias_detector._bias_config = BiasConfig(model_path=model_path)

    # Mock the _save_model method
    with patch.object(bias_detector, "_save_model") as mock_save:
        bias_detector.fit(texts, labels)
        mock_save.assert_called_once_with(model_path)

    # Test with empty training data
    with pytest.raises(ValueError):
        bias_detector.fit([], [])


def test_extract_explanations(bias_detector):
    """Test explanation extraction."""
    # Test with valid model
    bias_detector._extract_explanations()
    assert isinstance(bias_detector._explanations, dict)

    # Test with no base_estimator attribute
    bias_detector._model = MagicMock(spec=[])
    bias_detector._extract_explanations()

    # Test with exception
    bias_detector._model = MagicMock()
    bias_detector._model.base_estimator = MagicMock()
    bias_detector._model.base_estimator.coef_ = "invalid"
    bias_detector._extract_explanations()


def test_classification(bias_detector):
    """Test classification functionality."""
    # Test with gender-biased text
    result = bias_detector._classify_impl("Men are better at math than women.")

    assert isinstance(result, ClassificationResult)
    assert result.label in bias_detector.config.labels
    assert 0 <= result.confidence <= 1
    assert "features" in result.metadata
    assert "explanations" in result.metadata

    # Test with uninitialized model
    bias_detector._initialized = False
    with pytest.raises(RuntimeError):
        bias_detector._classify_impl("Test text")
    bias_detector._initialized = True


def test_batch_classification(bias_detector):
    """Test batch classification."""
    texts = [
        "Men are better at math than women.",
        "Ethnic minorities face discrimination in the workplace.",
        "Conservatives are more patriotic than liberals.",
        "Young people are lazy and entitled.",
        "Rich people work harder than poor people.",
        "This is a neutral and unbiased statement about facts.",
    ]

    results = bias_detector.batch_classify(texts)

    assert isinstance(results, list)
    assert len(results) == len(texts)

    # Check each result
    for result in results:
        assert isinstance(result, ClassificationResult)
        assert result.label in bias_detector.config.labels
        assert 0 <= result.confidence <= 1
        assert "probabilities" in result.metadata
        assert "threshold" in result.metadata
        assert "is_confident" in result.metadata
        assert "bias_features" in result.metadata

    # Test with uninitialized pipeline
    bias_detector._pipeline = None
    with pytest.raises(RuntimeError):
        bias_detector.batch_classify(texts)


def test_get_bias_explanation(bias_detector):
    """Test bias explanation functionality."""
    # Test with valid bias type
    explanation = bias_detector.get_bias_explanation("gender", "Men are better at math than women.")

    assert isinstance(explanation, dict)
    assert explanation["bias_type"] == "gender"
    assert "probability" in explanation
    assert "confidence" in explanation
    assert "is_primary_bias" in explanation
    assert "contributing_features" in explanation
    assert "countering_features" in explanation
    assert "bias_specific_features" in explanation
    assert "examples" in explanation

    # Test with invalid bias type
    with pytest.raises(ValueError):
        bias_detector.get_bias_explanation("invalid_type", "Test text")


def test_create_pretrained(mock_sklearn_modules):
    """Test create_pretrained factory method."""
    texts = [
        "Men are better at math than women.",
        "Ethnic minorities face discrimination in the workplace.",
        "This is a neutral and unbiased statement about facts.",
    ]

    labels = ["gender", "racial", "neutral"]

    # Create a pretrained detector
    detector = BiasDetector.create_pretrained(
        texts=texts,
        labels=labels,
        name="pretrained_test",
        description="Pretrained test detector",
    )

    assert detector.name == "pretrained_test"
    assert detector.description == "Pretrained test detector"
    assert detector._initialized is True
