"""Tests for the topic classifier."""

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sifaka.classifiers.base import ClassificationResult, ClassifierConfig
from sifaka.classifiers.topic import TopicClassifier, TopicConfig


class MockLatentDirichletAllocation:
    """Mock LDA implementation for testing."""

    def __init__(self, n_components=5, random_state=42):
        """Initialize with parameters."""
        self.n_components = n_components
        self.random_state = random_state
        self.components_ = np.array(
            [
                [0.1, 0.5, 0.2, 0.8, 0.3, 0.1, 0.9, 0.4, 0.2, 0.1],  # Topic 0
                [0.8, 0.1, 0.1, 0.2, 0.7, 0.9, 0.1, 0.3, 0.5, 0.6],  # Topic 1
                [0.3, 0.2, 0.8, 0.1, 0.1, 0.2, 0.3, 0.7, 0.6, 0.4],  # Topic 2
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],  # Topic 3
                [0.5, 0.6, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.4, 0.8],  # Topic 4
            ]
        )

    def fit(self, X):
        """Mock fit method."""
        return self

    def transform(self, X):
        """Mock transform method."""
        # Return mock topic distributions for each document
        n_samples = X.shape[0]
        distributions = []

        # Generate different distributions based on input
        for i in range(n_samples):
            if i % 5 == 0:
                # First topic dominant
                distributions.append([0.7, 0.1, 0.1, 0.05, 0.05])
            elif i % 5 == 1:
                # Second topic dominant
                distributions.append([0.1, 0.7, 0.1, 0.05, 0.05])
            elif i % 5 == 2:
                # Third topic dominant
                distributions.append([0.1, 0.1, 0.7, 0.05, 0.05])
            elif i % 5 == 3:
                # Fourth topic dominant
                distributions.append([0.05, 0.05, 0.1, 0.7, 0.1])
            else:
                # Fifth topic dominant
                distributions.append([0.05, 0.05, 0.1, 0.1, 0.7])

        return np.array(distributions)


class MockTfidfVectorizer:
    """Mock TF-IDF vectorizer for testing."""

    def __init__(self, max_features=1000, stop_words=None):
        """Initialize with parameters."""
        self.max_features = max_features
        self.stop_words = stop_words
        self.vocabulary_ = {
            "science": 0,
            "technology": 1,
            "research": 2,
            "data": 3,
            "analysis": 4,
            "politics": 5,
            "government": 6,
            "policy": 7,
            "election": 8,
            "vote": 9,
        }

    def fit_transform(self, texts):
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
        return list(self.vocabulary_.keys())


@pytest.fixture
def mock_sklearn_modules():
    """Create mock sklearn modules."""
    with patch("importlib.import_module") as mock_import:
        # Create mock modules
        mock_feature_extraction_text = MagicMock()
        mock_feature_extraction_text.TfidfVectorizer = MockTfidfVectorizer

        mock_decomposition = MagicMock()
        mock_decomposition.LatentDirichletAllocation = MockLatentDirichletAllocation

        # Configure import to return our mocks
        def side_effect(name):
            if name == "sklearn.feature_extraction.text":
                return mock_feature_extraction_text
            elif name == "sklearn.decomposition":
                return mock_decomposition
            else:
                raise ImportError(f"Mock cannot import {name}")

        mock_import.side_effect = side_effect

        yield {
            "feature_extraction_text": mock_feature_extraction_text,
            "decomposition": mock_decomposition,
        }


@pytest.fixture
def topic_classifier(mock_sklearn_modules):
    """Create a TopicClassifier with mocked dependencies."""
    # Create classifier with default config
    classifier = TopicClassifier()

    # Patch the _load_dependencies method to avoid actual imports
    with patch.object(classifier, "_load_dependencies", return_value=True):
        # Manually set up the classifier for testing
        classifier._initialized = True
        classifier._sklearn_feature_extraction_text = mock_sklearn_modules[
            "feature_extraction_text"
        ]
        classifier._sklearn_decomposition = mock_sklearn_modules["decomposition"]
        classifier._vectorizer = MockTfidfVectorizer()
        classifier._model = MockLatentDirichletAllocation()
        classifier._feature_names = list(classifier._vectorizer.vocabulary_.keys())

        # Create mock topic words
        classifier._topic_words = [
            ["science", "technology", "research"],  # Topic 0
            ["politics", "government", "policy"],  # Topic 1
            ["data", "analysis", "research"],  # Topic 2
            ["election", "vote", "government"],  # Topic 3
            ["technology", "science", "data"],  # Topic 4
        ]

        # Update labels with meaningful names
        classifier.config = ClassifierConfig(
            labels=[
                f"topic_0_{'+'.join(classifier._topic_words[0][:3])}",
                f"topic_1_{'+'.join(classifier._topic_words[1][:3])}",
                f"topic_2_{'+'.join(classifier._topic_words[2][:3])}",
                f"topic_3_{'+'.join(classifier._topic_words[3][:3])}",
                f"topic_4_{'+'.join(classifier._topic_words[4][:3])}",
            ],
            cost=2.0,
        )

        return classifier


def test_initialization():
    """Test TopicClassifier initialization."""
    # Test basic initialization
    classifier = TopicClassifier()
    assert classifier.name == "topic_classifier"
    assert classifier.description == "Classifies text into topics using LDA"
    assert len(classifier.config.labels) == 5  # Default num_topics
    assert classifier.config.cost == 2.0

    # Test with custom topic config
    topic_config = TopicConfig(
        num_topics=10,
        min_confidence=0.2,
        max_features=2000,
        random_state=123,
        top_words_per_topic=15,
    )

    # Create a custom config
    config = ClassifierConfig(
        labels=[f"topic_{i}" for i in range(10)],
        cost=3.0,
        params={
            "num_topics": 10,
            "min_confidence": 0.2,
            "max_features": 2000,
            "random_state": 123,
            "top_words_per_topic": 15,
        },
    )

    classifier = TopicClassifier(
        name="custom_topic",
        description="Custom topic classifier",
        topic_config=topic_config,
        config=config,
    )

    assert classifier.name == "custom_topic"
    assert classifier.description == "Custom topic classifier"
    assert len(classifier.config.labels) == 10
    assert classifier.config.cost == 3.0


def test_topic_config_validation():
    """Test TopicConfig validation."""
    # Test valid config
    config = TopicConfig(num_topics=5, min_confidence=0.5)
    assert config.num_topics == 5
    assert config.min_confidence == 0.5

    # Test invalid num_topics
    with pytest.raises(ValueError):
        TopicConfig(num_topics=0)

    with pytest.raises(ValueError):
        TopicConfig(num_topics=-1)

    # Test invalid min_confidence
    with pytest.raises(ValueError):
        TopicConfig(min_confidence=-0.1)

    with pytest.raises(ValueError):
        TopicConfig(min_confidence=1.1)

    # Test invalid max_features
    with pytest.raises(ValueError):
        TopicConfig(max_features=0)

    # Test invalid top_words_per_topic
    with pytest.raises(ValueError):
        TopicConfig(top_words_per_topic=0)


def test_warm_up(mock_sklearn_modules):
    """Test warm_up functionality."""
    # Test successful warm_up
    classifier = TopicClassifier()

    # Patch _load_dependencies to return our mocks
    with patch.object(classifier, "_load_dependencies", return_value=True):
        # Manually set the sklearn modules
        classifier._sklearn_feature_extraction_text = mock_sklearn_modules[
            "feature_extraction_text"
        ]
        classifier._sklearn_decomposition = mock_sklearn_modules["decomposition"]

        # Call warm_up
        classifier.warm_up()

        # Verify results
        assert classifier._initialized is True
        assert classifier._vectorizer is not None
        assert classifier._model is not None

    # Test warm_up with missing package
    classifier = TopicClassifier()
    with patch.object(
        classifier, "_load_dependencies", side_effect=ImportError("Test import error")
    ):
        with pytest.raises(ImportError):
            classifier.warm_up()

    # Test warm_up with other exception
    classifier = TopicClassifier()
    with patch.object(
        classifier, "_load_dependencies", side_effect=RuntimeError("Test runtime error")
    ):
        with pytest.raises(RuntimeError):
            classifier.warm_up()


def test_fit(topic_classifier):
    """Test fit method."""
    # Create sample texts
    texts = [
        "Science and technology research with data analysis",
        "Politics and government policy for the election",
        "Sports game with team players and scores",
        "Health and medical doctors treating patients",
        "Business finance and market investment companies",
    ]

    # Test fit method
    with patch.object(topic_classifier, "_vectorizer") as mock_vectorizer:
        with patch.object(topic_classifier, "_model") as mock_model:
            # Configure mocks
            mock_matrix = MagicMock()
            mock_vectorizer.fit_transform.return_value = mock_matrix
            mock_model.fit.return_value = mock_model
            mock_vectorizer.get_feature_names_out.return_value = topic_classifier._feature_names

            # Call fit
            result = topic_classifier.fit(texts)

            # Verify method calls
            mock_vectorizer.fit_transform.assert_called_once_with(texts)
            mock_model.fit.assert_called_once_with(mock_matrix)

            # Verify result
            assert result is topic_classifier
            assert topic_classifier._topic_words is not None
            assert len(topic_classifier.config.labels) == 5


def test_classification(topic_classifier):
    """Test classification functionality."""
    # Test with a science text
    with patch.object(topic_classifier, "_vectorizer") as mock_vectorizer:
        with patch.object(topic_classifier, "_model") as mock_model:
            # Configure mocks
            mock_matrix = MagicMock()
            mock_vectorizer.transform.return_value = mock_matrix

            # Configure mock_model.transform to return a distribution with first topic dominant
            mock_model.transform.return_value = np.array([[0.7, 0.1, 0.1, 0.05, 0.05]])

            # Call classify
            result = topic_classifier._classify_impl("Science and technology research")

            # Verify method calls
            mock_vectorizer.transform.assert_called_once_with(["Science and technology research"])
            mock_model.transform.assert_called_once_with(mock_matrix)

            # Verify result
            assert isinstance(result, ClassificationResult)
            assert result.label == topic_classifier.config.labels[0]
            assert result.confidence == 0.7
            assert "all_topics" in result.metadata
            assert "topic_words" in result.metadata
            assert "topic_distribution" in result.metadata

            # Check metadata structure
            assert len(result.metadata["all_topics"]) == 5
            for topic_label, topic_data in result.metadata["all_topics"].items():
                assert "probability" in topic_data
                assert "words" in topic_data
                assert isinstance(topic_data["probability"], float)
                assert isinstance(topic_data["words"], list)


def test_batch_classification(topic_classifier):
    """Test batch classification."""
    texts = [
        "Science and technology research",
        "Politics and government policy",
        "Data analysis research",
        "Election vote government",
        "Technology science data",
    ]

    with patch.object(topic_classifier, "_vectorizer") as mock_vectorizer:
        with patch.object(topic_classifier, "_model") as mock_model:
            # Configure mocks
            mock_matrix = MagicMock()
            mock_vectorizer.transform.return_value = mock_matrix

            # Configure mock_model.transform to return distributions with different dominant topics
            mock_model.transform.return_value = np.array(
                [
                    [0.7, 0.1, 0.1, 0.05, 0.05],  # First topic dominant
                    [0.1, 0.7, 0.1, 0.05, 0.05],  # Second topic dominant
                    [0.1, 0.1, 0.7, 0.05, 0.05],  # Third topic dominant
                    [0.05, 0.05, 0.1, 0.7, 0.1],  # Fourth topic dominant
                    [0.05, 0.05, 0.1, 0.1, 0.7],  # Fifth topic dominant
                ]
            )

            # Call batch_classify
            results = topic_classifier.batch_classify(texts)

            # Verify method calls
            mock_vectorizer.transform.assert_called_once_with(texts)
            mock_model.transform.assert_called_once_with(mock_matrix)

            # Verify results
            assert isinstance(results, list)
            assert len(results) == len(texts)

            # Check each result
            for i, result in enumerate(results):
                assert isinstance(result, ClassificationResult)
                assert result.label == topic_classifier.config.labels[i]
                assert result.confidence == 0.7
                assert "all_topics" in result.metadata
                assert "topic_words" in result.metadata
                assert "topic_distribution" in result.metadata


def test_error_handling(topic_classifier):
    """Test error handling."""
    # Test with invalid input types
    invalid_inputs = [None, 123, [], {}]

    for invalid_input in invalid_inputs:
        with pytest.raises(Exception):
            topic_classifier.classify(invalid_input)

        with pytest.raises(Exception):
            topic_classifier.batch_classify([invalid_input])

    # Test classification without initialization
    uninit_classifier = TopicClassifier()
    uninit_classifier._initialized = False
    uninit_classifier._model = None
    uninit_classifier._vectorizer = None

    with pytest.raises(RuntimeError):
        uninit_classifier._classify_impl("This should fail")

    with pytest.raises(RuntimeError):
        uninit_classifier.batch_classify(["This should fail"])


def test_create_pretrained(mock_sklearn_modules):
    """Test create_pretrained factory method."""
    texts = [
        "Science and technology research with data analysis",
        "Politics and government policy for the election",
        "Sports game with team players and scores",
        "Health and medical doctors treating patients",
        "Business finance and market investment companies",
    ]

    # Create a mock classifier to return from fit
    mock_classifier = TopicClassifier(
        name="pretrained_test",
        description="Pretrained test classifier",
    )

    # Patch the TopicClassifier constructor to avoid actual initialization
    with patch.object(TopicClassifier, "__new__", return_value=mock_classifier):
        # Patch the fit method to return our mock classifier
        with patch.object(mock_classifier, "fit", return_value=mock_classifier):
            # Create a pretrained classifier
            classifier = TopicClassifier.create_pretrained(
                corpus=texts,
                name="pretrained_test",
                description="Pretrained test classifier",
                topic_config=TopicConfig(num_topics=3),
            )

            # Verify method calls
            mock_classifier.fit.assert_called_once_with(texts)

            # Verify result
            assert classifier.name == "pretrained_test"
            assert classifier.description == "Pretrained test classifier"
