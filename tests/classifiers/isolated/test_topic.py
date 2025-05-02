"""Test module for the topic classifier."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from typing import Dict, Any, List

# Import from module under test - these imports will be handled by the patched import
from sifaka.classifiers.topic import TopicClassifier
from sifaka.classifiers.base import ClassificationResult


# Create a concrete subclass for testing
class TestableTopicClassifier(TopicClassifier):
    """Concrete implementation of TopicClassifier for testing."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implement the abstract method for testing."""
        # Simple implementation for testing
        topic_idx = hash(text) % self.num_topics
        label = self.config.labels[topic_idx]
        confidence = 0.7

        metadata = {
            "topic_words": ["word1", "word2", "word3"],
            "topic_distribution": [0.1] * self.num_topics,
        }
        metadata["topic_distribution"][topic_idx] = confidence

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata=metadata
        )


class MockTopicModel:
    """Mock implementation of LDA model for testing."""

    def __init__(self, n_components=5):
        """Initialize with number of topics."""
        self.n_components = n_components
        self.call_count = 0
        # Generate mock components (word-topic matrix)
        self.components_ = np.array([
            [0.1, 0.2, 0.5, 0.1, 0.1],  # Topic 0: word 2 has highest weight
            [0.5, 0.1, 0.1, 0.2, 0.1],  # Topic 1: word 0 has highest weight
            [0.1, 0.5, 0.1, 0.1, 0.2],  # Topic 2: word 1 has highest weight
            [0.2, 0.1, 0.1, 0.5, 0.1],  # Topic 3: word 3 has highest weight
            [0.1, 0.1, 0.2, 0.1, 0.5],  # Topic 4: word 4 has highest weight
        ])

    def fit(self, X):
        """Mock fit method."""
        self.call_count += 1
        return self

    def transform(self, X):
        """Mock transform method that returns topic distributions."""
        self.call_count += 1

        if X.shape[0] == 1:
            # Single document mode
            return np.array([[0.05, 0.1, 0.7, 0.05, 0.1]])  # Dominant topic is index 2
        else:
            # Batch mode - create different distributions for each document
            result = []
            for i in range(X.shape[0]):
                if i % 5 == 0:
                    result.append([0.7, 0.1, 0.1, 0.05, 0.05])  # Dominant topic 0
                elif i % 5 == 1:
                    result.append([0.1, 0.7, 0.1, 0.05, 0.05])  # Dominant topic 1
                elif i % 5 == 2:
                    result.append([0.1, 0.1, 0.7, 0.05, 0.05])  # Dominant topic 2
                elif i % 5 == 3:
                    result.append([0.1, 0.05, 0.05, 0.7, 0.1])  # Dominant topic 3
                else:
                    result.append([0.1, 0.05, 0.05, 0.1, 0.7])  # Dominant topic 4
            return np.array(result)


class MockVectorizer:
    """Mock implementation of TfidfVectorizer."""

    def __init__(self, max_features=1000, stop_words="english"):
        """Initialize vectorizer."""
        self.max_features = max_features
        self.stop_words = stop_words
        self.vocabulary = ["technology", "science", "politics", "sports", "entertainment"]

    def fit_transform(self, texts):
        """Mock fit_transform method."""
        # Return a sparse matrix of shape (len(texts), 5)
        # The shape doesn't matter since we're mocking the model
        return MagicMock(shape=(len(texts), 5))

    def transform(self, texts):
        """Mock transform method."""
        # Return a sparse matrix of shape (len(texts), 5)
        return MagicMock(shape=(len(texts), 5))

    def get_feature_names_out(self):
        """Mock get_feature_names_out method."""
        return np.array(self.vocabulary)


@pytest.fixture
def mock_modules():
    """Fixture for mock scikit-learn modules."""
    feature_extraction = MagicMock()
    feature_extraction.TfidfVectorizer.return_value = MockVectorizer()

    decomposition = MagicMock()
    decomposition.LatentDirichletAllocation.return_value = MockTopicModel()

    return {
        "sklearn.feature_extraction.text": feature_extraction,
        "sklearn.decomposition": decomposition
    }


@patch("importlib.import_module")
def test_init(mock_import):
    """Test initialization of topic classifier."""
    classifier = TestableTopicClassifier()

    assert classifier.name == "topic_classifier"
    assert classifier.description == "Classifies text into topics using LDA"
    assert hasattr(classifier.config, "labels")
    assert len(classifier.config.labels) == 5  # Default number of topics
    assert classifier.config.cost == 2.0  # Default cost


@patch("importlib.import_module")
def test_custom_init(mock_import):
    """Test initialization with custom parameters."""
    classifier = TestableTopicClassifier(
        name="custom_topic",
        description="Custom topic detector",
        config=None,
        params={
            "num_topics": 10,
            "max_features": 2000,
            "top_words_per_topic": 5
        }
    )

    assert classifier.name == "custom_topic"
    assert classifier.description == "Custom topic detector"
    assert len(classifier.config.labels) == 10  # Custom number of topics
    assert classifier.config.params["max_features"] == 2000
    assert classifier.config.params["top_words_per_topic"] == 5
    assert classifier.num_topics == 10
    assert classifier.top_words_per_topic == 5


@patch("importlib.import_module")
def test_property_getters(mock_import):
    """Test property getters."""
    classifier = TestableTopicClassifier(
        params={
            "num_topics": 7,
            "top_words_per_topic": 15
        }
    )

    assert classifier.num_topics == 7
    assert classifier.top_words_per_topic == 15

    # Test default values
    classifier = TestableTopicClassifier()
    assert classifier.num_topics == 5  # Default
    assert classifier.top_words_per_topic == 10  # Default


def test_load_dependencies_with_mock_modules(mock_modules):
    """Test _load_dependencies method with mocked modules."""
    classifier = TestableTopicClassifier()

    with patch("importlib.import_module", side_effect=lambda name: mock_modules.get(name, MagicMock())):
        result = classifier._load_dependencies()

        assert result is True
        assert classifier._sklearn_feature_extraction_text is mock_modules["sklearn.feature_extraction.text"]
        assert classifier._sklearn_decomposition is mock_modules["sklearn.decomposition"]


@patch("importlib.import_module", side_effect=ImportError("No scikit-learn"))
def test_load_dependencies_import_error(mock_import):
    """Test _load_dependencies raises ImportError when scikit-learn is not available."""
    classifier = TestableTopicClassifier()

    with pytest.raises(ImportError) as context:
        classifier._load_dependencies()

    assert "scikit-learn is required" in str(context.value)


@patch("importlib.import_module", side_effect=Exception("Unknown error"))
def test_load_dependencies_other_error(mock_import):
    """Test _load_dependencies raises RuntimeError for other errors."""
    classifier = TestableTopicClassifier()

    with pytest.raises(RuntimeError) as context:
        classifier._load_dependencies()

    assert "Failed to load scikit-learn modules" in str(context.value)


def test_warm_up(mock_modules):
    """Test warm_up method initializes the model."""
    classifier = TestableTopicClassifier(
        params={
            "max_features": 2000,
            "random_state": 123
        }
    )

    with patch("importlib.import_module", side_effect=lambda name: mock_modules.get(name, MagicMock())):
        classifier.warm_up()

        assert classifier._initialized is True
        assert classifier._vectorizer is not None
        assert classifier._model is not None

        # Check that the vectorizer was initialized with correct parameters
        mock_modules["sklearn.feature_extraction.text"].TfidfVectorizer.assert_called_once_with(
            max_features=2000,
            stop_words="english"
        )

        # Check that the LDA model was initialized with correct parameters
        mock_modules["sklearn.decomposition"].LatentDirichletAllocation.assert_called_once_with(
            n_components=5,
            random_state=123
        )


def test_fit(mock_modules):
    """Test fit method trains the model."""
    classifier = TestableTopicClassifier()

    # Set up mocks
    with patch("importlib.import_module", side_effect=lambda name: mock_modules.get(name, MagicMock())):
        # Call warm_up to initialize mocks
        classifier.warm_up()

        # Training data
        texts = [
            "Technology and science are advancing rapidly in today's world.",
            "Political decisions can have a significant impact on society.",
            "Sports events bring people together from all walks of life.",
            "Entertainment industry has been transformed by digital platforms.",
            "Scientific discoveries continue to shape our understanding of the universe."
        ]

        # Train the model
        result = classifier.fit(texts)

        # Check return value
        assert result is classifier

        # Check that the model was trained
        assert classifier._initialized is True
        assert classifier._vectorizer is not None
        assert classifier._model is not None
        assert classifier._feature_names is not None
        assert classifier._topic_words is not None

        # Check that the labels were updated with meaningful topic names
        assert all(label.startswith("topic_") for label in classifier.config.labels)
        # Should have num_topics labels
        assert len(classifier.config.labels) == classifier.num_topics


def test_classify_not_initialized():
    """Test _classify_impl raises RuntimeError when model is not initialized."""
    classifier = TestableTopicClassifier()

    with pytest.raises(RuntimeError) as context:
        classifier._classify_impl("Test text")

    assert "Model not initialized" in str(context.value)


def test_batch_classify_not_initialized():
    """Test batch_classify raises RuntimeError when model is not initialized."""
    classifier = TestableTopicClassifier()

    with pytest.raises(RuntimeError) as context:
        classifier.batch_classify(["Test text"])

    assert "Model not initialized" in str(context.value)


def test_classify(mock_modules):
    """Test classification with mock model."""
    classifier = TestableTopicClassifier()

    with patch("importlib.import_module", side_effect=lambda name: mock_modules.get(name, MagicMock())):
        # Initialize and train the model
        classifier.warm_up()

        # Set up the model's state after training
        classifier._feature_names = np.array(["technology", "science", "politics", "sports", "entertainment"])
        classifier._topic_words = [
            ["technology", "computer", "digital"],
            ["science", "research", "discovery"],
            ["politics", "government", "election"],
            ["sports", "athlete", "competition"],
            ["entertainment", "movie", "music"]
        ]

        # Create a new config with updated labels
        new_labels = [
            "topic_0_technology+computer+digital",
            "topic_1_science+research+discovery",
            "topic_2_politics+government+election",
            "topic_3_sports+athlete+competition",
            "topic_4_entertainment+movie+music"
        ]
        # Create a new config with the same properties but new labels
        classifier.config = type(classifier.config)(
            labels=new_labels,
            cost=classifier.config.cost,
            params=classifier.config.params
        )

        # Test classification
        text = "This is a test text about politics and government."
        result = classifier._classify_impl(text)

        # Based on our mock implementation, result could be any topic based on hash
        assert result.label.startswith("topic_")
        assert result.confidence > 0

        # Check metadata
        assert "topic_words" in result.metadata
        assert "topic_distribution" in result.metadata

        # Check topic distribution
        assert len(result.metadata["topic_distribution"]) == 5


def test_batch_classify(mock_modules):
    """Test batch classification with mock model."""
    classifier = TestableTopicClassifier()

    with patch("importlib.import_module", side_effect=lambda name: mock_modules.get(name, MagicMock())):
        # Initialize and train the model
        classifier.warm_up()

        # Set up the model's state after training
        classifier._feature_names = np.array(["technology", "science", "politics", "sports", "entertainment"])
        classifier._topic_words = [
            ["technology", "computer", "digital"],
            ["science", "research", "discovery"],
            ["politics", "government", "election"],
            ["sports", "athlete", "competition"],
            ["entertainment", "movie", "music"]
        ]

        # Create a new config with updated labels
        new_labels = [
            "topic_0_technology+computer+digital",
            "topic_1_science+research+discovery",
            "topic_2_politics+government+election",
            "topic_3_sports+athlete+competition",
            "topic_4_entertainment+movie+music"
        ]
        # Create a new config with the same properties but new labels
        classifier.config = type(classifier.config)(
            labels=new_labels,
            cost=classifier.config.cost,
            params=classifier.config.params
        )

        # Test batch classification
        texts = [
            "This is about technology.",
            "This is about science.",
            "This is about politics.",
            "This is about sports.",
            "This is about entertainment."
        ]
        results = classifier.batch_classify(texts)

        # Check results
        assert len(results) == 5
        assert all(isinstance(r, ClassificationResult) for r in results)

        # Each result should have the correct structure
        for result in results:
            assert result.label.startswith("topic_")
            assert result.confidence > 0
            assert "topic_words" in result.metadata
            assert "topic_distribution" in result.metadata


@patch("importlib.import_module")
def test_create_pretrained(mock_import, mock_modules):
    """Test create_pretrained class method."""
    # Setup spy on fit method
    original_fit = TestableTopicClassifier.fit
    fit_called = [False]
    fit_corpus = [None]

    def spy_fit(self, corpus):
        fit_called[0] = True
        fit_corpus[0] = corpus

        # Set up the model's state as if it had been trained
        self._feature_names = np.array(["technology", "science", "politics", "sports", "entertainment"])
        self._topic_words = [
            ["technology", "computer", "digital"],
            ["science", "research", "discovery"],
            ["politics", "government", "election"],
            ["sports", "athlete", "competition"],
            ["entertainment", "movie", "music"]
        ]

        # Create new labels
        new_labels = [
            "topic_0_technology+computer+digital",
            "topic_1_science+research+discovery",
            "topic_2_politics+government+election",
            "topic_3_sports+athlete+competition",
            "topic_4_entertainment+movie+music"
        ]

        # Create a new config with the same properties but new labels
        self.config = type(self.config)(
            labels=new_labels,
            cost=self.config.cost,
            params=self.config.params
        )

        self._initialized = True
        self._vectorizer = MockVectorizer()
        self._model = MockTopicModel()

        return self

    # Replace the method temporarily
    TestableTopicClassifier.fit = spy_fit

    try:
        # Training data
        corpus = [
            "Technology and science are advancing rapidly in today's world.",
            "Political decisions can have a significant impact on society.",
            "Sports events bring people together from all walks of life.",
            "Entertainment industry has been transformed by digital platforms.",
            "Scientific discoveries continue to shape our understanding of the universe."
        ]

        # Create pretrained classifier
        mock_import.side_effect = lambda name: mock_modules.get(name, MagicMock())

        classifier = TestableTopicClassifier.create_pretrained(
            corpus=corpus,
            name="pretrained_topics",
            description="Pretrained topic classifier",
            params={
                "num_topics": 5,
                "max_features": 1000
            }
        )

        # Check that the classifier was created
        assert isinstance(classifier, TestableTopicClassifier)
        assert classifier.name == "pretrained_topics"
        assert classifier.description == "Pretrained topic classifier"
        assert classifier.num_topics == 5
        assert classifier._initialized is True

        # Check that fit was called with the correct arguments
        assert fit_called[0] is True
        assert fit_corpus[0] == corpus

        # Check labels after training
        assert len(classifier.config.labels) == 5
        assert all("topic_" in label for label in classifier.config.labels)
    finally:
        # Restore the original method
        TestableTopicClassifier.fit = original_fit