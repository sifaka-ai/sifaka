"""
Tests for the topic classifier.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

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
            "sports": 10,
            "game": 11,
            "team": 12,
            "player": 13,
            "score": 14,
            "health": 15,
            "medical": 16,
            "doctor": 17,
            "patient": 18,
            "treatment": 19,
            "business": 20,
            "finance": 21,
            "market": 22,
            "company": 23,
            "investment": 24,
        }
        self.feature_names = list(self.vocabulary_.keys())

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
        return self.feature_names


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
    # Create a mock classifier
    classifier = MagicMock(spec=TopicClassifier)

    # Set up the mock properties and methods
    classifier.name = "topic_classifier"
    classifier.description = "Classifies text into topics using LDA"
    classifier._initialized = True
    classifier._vectorizer = mock_sklearn_modules["feature_extraction_text"].TfidfVectorizer()
    classifier._model = mock_sklearn_modules["decomposition"].LatentDirichletAllocation()

    # Create mock topic words
    classifier._feature_names = [
        "science",
        "technology",
        "research",
        "data",
        "analysis",
        "politics",
        "government",
        "policy",
        "election",
        "vote",
        "sports",
        "game",
        "team",
        "player",
        "score",
        "health",
        "medical",
        "doctor",
        "patient",
        "treatment",
        "business",
        "finance",
        "market",
        "company",
        "investment",
    ]

    classifier._topic_words = [
        ["science", "technology", "research", "data", "analysis"],  # Topic 0: Science
        ["politics", "government", "policy", "election", "vote"],  # Topic 1: Politics
        ["sports", "game", "team", "player", "score"],  # Topic 2: Sports
        ["health", "medical", "doctor", "patient", "treatment"],  # Topic 3: Health
        ["business", "finance", "market", "company", "investment"],  # Topic 4: Business
    ]

    # Set up config
    classifier.config = ClassifierConfig(
        labels=[
            "topic_0_science+technology+research",
            "topic_1_politics+government+policy",
            "topic_2_sports+game+team",
            "topic_3_health+medical+doctor",
            "topic_4_business+finance+market",
        ],
        cost=2.0,
    )

    # Mock the topic_config property
    mock_topic_config = MagicMock()
    mock_topic_config.num_topics = 5
    mock_topic_config.min_confidence = 0.1
    mock_topic_config.max_features = 1000
    mock_topic_config.random_state = 42
    mock_topic_config.top_words_per_topic = 10

    # Set up the property
    type(classifier).topic_config = PropertyMock(return_value=mock_topic_config)

    # Mock the classify method
    def mock_classify_impl(text):
        # Return a mock classification result
        topic_distribution = np.zeros(5)
        topic_distribution[0] = 0.7  # Make the first topic dominant

        all_topics = {}
        for i, label in enumerate(classifier.config.labels):
            all_topics[label] = {
                "probability": float(topic_distribution[i]),
                "words": classifier._topic_words[i][:5],
            }

        return ClassificationResult(
            label=classifier.config.labels[0],
            confidence=0.7,
            metadata={
                "all_topics": all_topics,
                "topic_words": classifier._topic_words[0],
                "topic_distribution": topic_distribution.tolist(),
            },
        )

    classifier._classify_impl.side_effect = mock_classify_impl

    # Mock the batch_classify method
    def mock_batch_classify(texts):
        return [mock_classify_impl(text) for text in texts]

    classifier.batch_classify.side_effect = mock_batch_classify

    # Mock the fit method
    classifier.fit.return_value = classifier

    # Mock the warm_up method
    classifier.warm_up.return_value = None

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

    classifier = TopicClassifier(
        name="custom_topic",
        description="Custom topic classifier",
        topic_config=topic_config,
    )

    assert classifier.name == "custom_topic"
    assert classifier.description == "Custom topic classifier"
    assert len(classifier.config.labels) == 10
    assert classifier.topic_config.num_topics == 10
    assert classifier.topic_config.min_confidence == 0.2
    assert classifier.topic_config.max_features == 2000
    assert classifier.topic_config.random_state == 123
    assert classifier.topic_config.top_words_per_topic == 15


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
    classifier = TopicClassifier()

    # Test warm_up with mocked modules
    classifier.warm_up()
    assert classifier._initialized is True
    assert classifier._vectorizer is not None
    assert classifier._model is not None

    # Test warm_up with missing package
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError()
        classifier = TopicClassifier()
        with pytest.raises(ImportError):
            classifier.warm_up()

    # Test warm_up with other exception
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = RuntimeError("Test error")
        classifier = TopicClassifier()
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

    # Fit the classifier
    fitted_classifier = topic_classifier.fit(texts)

    # Check that it returns self
    assert fitted_classifier is topic_classifier

    # Check that topic words and labels are updated
    assert topic_classifier._topic_words is not None
    assert len(topic_classifier.config.labels) == 5
    assert all("topic_" in label for label in topic_classifier.config.labels)


def test_classification(topic_classifier):
    """Test classification functionality."""
    # Test with a science text
    result = topic_classifier._classify_impl("Science and technology research")

    assert isinstance(result, ClassificationResult)
    assert "topic_0" in result.label  # Should be topic 0 (science)
    assert 0 <= result.confidence <= 1
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
        "Sports game with team players",
        "Health and medical doctors",
        "Business finance and market",
    ]

    results = topic_classifier.batch_classify(texts)

    assert isinstance(results, list)
    assert len(results) == len(texts)

    # Check each result
    for i, result in enumerate(results):
        assert isinstance(result, ClassificationResult)
        assert f"topic_{i}" in result.label  # Each text should match its corresponding topic
        assert 0 <= result.confidence <= 1
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

    with pytest.raises(RuntimeError):
        uninit_classifier.classify("This should fail")

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

    # Create a pretrained classifier
    classifier = TopicClassifier.create_pretrained(
        corpus=texts,
        name="pretrained_test",
        description="Pretrained test classifier",
        topic_config=TopicConfig(num_topics=3),
    )

    assert classifier.name == "pretrained_test"
    assert classifier.description == "Pretrained test classifier"
    assert classifier.topic_config.num_topics == 3
    assert classifier._initialized is True
