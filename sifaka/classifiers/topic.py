"""
Topic classifier using scikit-learn's LDA.

This module provides a classifier for extracting topics from text using
Latent Dirichlet Allocation (LDA) from scikit-learn.
"""

import importlib
from typing import List, Optional, Any, Dict, ClassVar, Union

from pydantic import PrivateAttr
from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.state import ClassifierState, create_classifier_state
from sifaka.utils.config import standardize_classifier_config

logger = get_logger(__name__)


class TopicClassifier(BaseClassifier):
    """
    A topic classifier using Latent Dirichlet Allocation from scikit-learn.

    This classifier extracts topics from text using LDA and returns the
    dominant topic with confidence scores for each topic.

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    # Class constants
    DEFAULT_COST: ClassVar[float] = 2.0

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)

    def __init__(
        self,
        name: str = "topic_classifier",
        description: str = "Classifies text into topics using LDA",
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the topic classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Create config if not provided
        if config is None:
            # Extract params from kwargs if present
            params = kwargs.pop("params", {})

            # Get num_topics from params or use default
            num_topics = params.get("num_topics", 5)
            min_confidence = params.get("min_confidence", 0.1)

            # Create config with remaining kwargs
            config = ClassifierConfig(
                labels=[f"topic_{i}" for i in range(num_topics)],
                cost=self.DEFAULT_COST,
                min_confidence=min_confidence,
                params=params,
                **kwargs,
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False
        state.cache = {}

    @property
    def num_topics(self) -> int:
        """Get the number of topics."""
        return self.config.params.get("num_topics", 5)

    @property
    def top_words_per_topic(self) -> int:
        """Get the number of top words per topic."""
        return self.config.params.get("top_words_per_topic", 10)

    def _load_dependencies(self) -> Dict[str, Any]:
        """
        Load required dependencies for topic classification.

        This method imports and initializes the necessary scikit-learn
        components for LDA topic modeling.

        Returns:
            Dictionary containing loaded dependencies:
            - 'CountVectorizer': For text vectorization
            - 'LatentDirichletAllocation': For topic modeling
            - 'TfidfTransformer': For TF-IDF transformation

        Raises:
            ImportError: If scikit-learn is not installed
        """
        try:
            # Import necessary scikit-learn modules
            sklearn_modules = {
                "feature_extraction_text": importlib.import_module(
                    "sklearn.feature_extraction.text"
                ),
                "decomposition": importlib.import_module("sklearn.decomposition"),
            }
            return sklearn_modules
        except ImportError:
            raise ImportError(
                "scikit-learn is required for TopicClassifier. "
                "Install it with: pip install scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn modules: {e}")

    def warm_up(self) -> None:
        """
        Warm up the classifier by initializing the LDA model.

        This method prepares the classifier for use by:
        1. Loading required dependencies
        2. Initializing the vectorizer and LDA model
        3. Setting up the topic-word matrix

        Raises:
            RuntimeError: If model initialization fails
        """
        # Get state
        state = self._state_manager.get_state()

        # Check if already initialized
        if not state.initialized:
            # Load dependencies
            sklearn = self._load_dependencies()
            state.dependencies_loaded = True

            # Get configuration from params
            max_features = self.config.params.get("max_features", 1000)
            random_state = self.config.params.get("random_state", 42)

            # Create vectorizer
            state.vectorizer = sklearn["feature_extraction_text"].TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
            )

            # Create LDA model
            state.model = sklearn["decomposition"].LatentDirichletAllocation(
                n_components=self.num_topics,
                random_state=random_state,
            )

            # Mark as initialized
            state.initialized = True

    def fit(self, texts: List[str]) -> "TopicClassifier":
        """
        Fit the topic model on a corpus of texts.

        Args:
            texts: List of texts to fit the model on

        Returns:
            self: The fitted classifier
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        self.warm_up()

        # Transform texts to TF-IDF features
        X = state.vectorizer.fit_transform(texts)

        # Fit LDA model
        state.model.fit(X)

        # Get feature names for later interpretation
        feature_names = state.vectorizer.get_feature_names_out()
        state.feature_names = feature_names

        # Extract the words that define each topic
        topic_words = []
        for _, topic in enumerate(state.model.components_):
            top_features_ind = topic.argsort()[: -self.top_words_per_topic - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_words.append(top_features)

        # Store topic words in state cache
        state.cache["topic_words"] = topic_words

        # Update labels with meaningful topic names
        labels = [f"topic_{i}_{'+'.join(words[:3])}" for i, words in enumerate(topic_words)]
        self.config = ClassifierConfig(
            labels=labels,
            cost=self.config.cost,
            min_confidence=self.config.min_confidence,
            params=self.config.params,
        )

        return self

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Classify a single text into topics.

        This method:
        1. Vectorizes the input text
        2. Applies the LDA model to get topic distributions
        3. Returns the dominant topic and confidence scores

        Args:
            text: The text to classify

        Returns:
            ClassificationResult containing:
            - dominant_topic: The most likely topic
            - confidence: Confidence score for the dominant topic
            - topic_distribution: Distribution across all topics
            - top_words: Most representative words for each topic

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If classification fails
        """
        # Get state
        state = self._state_manager.get_state()

        if not state.model or not state.vectorizer:
            raise RuntimeError(
                "Model not initialized. You must call fit() with a corpus before classification."
            )

        # Transform the input text
        X = state.vectorizer.transform([text])

        # Get topic distribution
        topic_distribution = state.model.transform(X)[0]

        # Get dominant topic
        dominant_topic_idx = topic_distribution.argmax()
        confidence = float(topic_distribution[dominant_topic_idx])

        # Get topic words from state cache
        topic_words = state.cache.get("topic_words", [])

        # Create detailed metadata
        all_topics = {}
        labels = list(self.config.labels)  # Convert FieldInfo to list
        for i, prob in enumerate(topic_distribution):
            all_topics[labels[i]] = {
                "probability": float(prob),
                "words": topic_words[i][: self.top_words_per_topic] if i < len(topic_words) else [],
            }

        metadata = {
            "all_topics": all_topics,
            "topic_words": (
                topic_words[dominant_topic_idx] if dominant_topic_idx < len(topic_words) else []
            ),
            "topic_distribution": topic_distribution.tolist(),
        }

        return ClassificationResult(
            label=labels[dominant_topic_idx],
            confidence=confidence,
            metadata=metadata,
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts into topics.

        This method efficiently processes multiple texts by:
        1. Vectorizing all texts at once
        2. Applying the LDA model in batch
        3. Returning topic distributions for each text

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults, one for each input text

        Raises:
            ValueError: If texts list is empty or contains invalid entries
            RuntimeError: If batch classification fails
        """
        # Get state
        state = self._state_manager.get_state()

        self.validate_batch_input(texts)

        if not state.model or not state.vectorizer:
            raise RuntimeError(
                "Model not initialized. You must call fit() with a corpus before classification."
            )

        # Transform all texts at once
        X = state.vectorizer.transform(texts)

        # Get topic distributions for all texts
        topic_distributions = state.model.transform(X)

        # Get topic words from state cache
        topic_words = state.cache.get("topic_words", [])

        results = []
        labels = list(self.config.labels)  # Convert FieldInfo to list
        for distribution in topic_distributions:
            dominant_topic_idx = distribution.argmax()
            confidence = float(distribution[dominant_topic_idx])

            all_topics = {}
            for j, prob in enumerate(distribution):
                all_topics[labels[j]] = {
                    "probability": float(prob),
                    "words": (
                        topic_words[j][: self.top_words_per_topic] if j < len(topic_words) else []
                    ),
                }

            metadata = {
                "all_topics": all_topics,
                "topic_words": (
                    topic_words[dominant_topic_idx] if dominant_topic_idx < len(topic_words) else []
                ),
                "topic_distribution": distribution.tolist(),
            }

            results.append(
                ClassificationResult(
                    label=labels[dominant_topic_idx],
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return results

    @classmethod
    def create_pretrained(
        cls,
        corpus: List[str],
        name: str = "pretrained_topic_classifier",
        description: str = "Pre-trained topic classifier",
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> "TopicClassifier":
        """
        Create a pre-trained topic classifier from a corpus.

        This method:
        1. Creates a new classifier instance
        2. Fits the LDA model on the provided corpus
        3. Returns the trained classifier

        Args:
            corpus: List of texts to train the model on
            name: Name for the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional keyword arguments for configuration

        Returns:
            A trained TopicClassifier instance

        Raises:
            ValueError: If corpus is empty or invalid
            RuntimeError: If model training fails
        """
        # Create default config if not provided
        if config is None:
            # Extract configuration parameters
            params = kwargs.pop("params", {})
            num_topics = params.get("num_topics", 5)
            min_confidence = params.get("min_confidence", 0.1)
            max_features = params.get("max_features", 1000)
            random_state = params.get("random_state", 42)
            top_words_per_topic = params.get("top_words_per_topic", 10)

            # Create config with standardized approach
            config = standardize_classifier_config(
                labels=[f"topic_{i}" for i in range(num_topics)],
                cost=cls.DEFAULT_COST,
                min_confidence=min_confidence,
                params={
                    "num_topics": num_topics,
                    "max_features": max_features,
                    "random_state": random_state,
                    "top_words_per_topic": top_words_per_topic,
                },
                **kwargs,
            )

        # Create instance with provided configuration
        classifier = cls(name=name, description=description, config=config)

        # Train the classifier and return it
        return classifier.fit(corpus)


def create_topic_classifier(
    name: str = "topic_classifier",
    description: str = "Classifies text into topics using LDA",
    num_topics: int = 5,
    min_confidence: float = 0.1,
    max_features: int = 1000,
    random_state: int = 42,
    top_words_per_topic: int = 10,
    cache_size: int = 100,
    cost: float = TopicClassifier.DEFAULT_COST,
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    **kwargs: Any,
) -> TopicClassifier:
    """
    Create a topic classifier.

    This factory function creates a TopicClassifier with the specified
    configuration options.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        num_topics: Number of topics to extract
        min_confidence: Minimum confidence threshold
        max_features: Maximum number of features for the vectorizer
        random_state: Random state for reproducibility
        top_words_per_topic: Number of words to extract per topic
        cache_size: Size of the cache for memoization
        cost: Cost of running the classifier
        config: Optional classifier configuration
        **kwargs: Additional configuration parameters

    Returns:
        A TopicClassifier instance

    Examples:
        ```python
        from sifaka.classifiers.topic import create_topic_classifier

        # Create a topic classifier with default settings
        classifier = create_topic_classifier()

        # Create a topic classifier with custom settings
        classifier = create_topic_classifier(
            name="custom_topic_classifier",
            description="Custom topic classifier with specific settings",
            num_topics=10,
            min_confidence=0.2,
            max_features=2000,
            top_words_per_topic=15,
            cache_size=200
        )

        # Fit the classifier on a corpus
        corpus = ["This is a document about technology.",
                 "This is a document about sports.",
                 "This is a document about politics."]
        classifier.fit(corpus)

        # Classify text
        result = classifier.classify("This is a document about artificial intelligence.")
        print(f"Topic: {result.label}, Confidence: {result.confidence:.2f}")
        ```
    """
    # Use standardize_classifier_config to handle different config formats
    classifier_config = standardize_classifier_config(
        config=config,
        labels=[f"topic_{i}" for i in range(num_topics)],
        min_confidence=min_confidence,
        cost=cost,
        cache_size=cache_size,
        params={
            "num_topics": num_topics,
            "max_features": max_features,
            "random_state": random_state,
            "top_words_per_topic": top_words_per_topic,
        },
        **kwargs,
    )

    return TopicClassifier(
        name=name,
        description=description,
        config=classifier_config,
    )
