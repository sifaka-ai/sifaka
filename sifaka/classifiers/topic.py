"""
Topic classifier using scikit-learn's LDA.

This module provides a classifier for extracting topics from text using
Latent Dirichlet Allocation (LDA) from scikit-learn.

## Architecture

TopicClassifier follows the composition over inheritance pattern:
1. **Classifier**: Provides the public API and handles caching
2. **Implementation**: Contains the core classification logic
3. **Factory Function**: Creates a classifier with the topic classification implementation

## Lifecycle

1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract topic parameters from config.params
   - Set up default values

2. **Warm-up**: Prepare resources for classification
   - Load scikit-learn dependencies
   - Initialize vectorizer and LDA model
   - Set up topic-word matrix

3. **Classification**: Process text inputs
   - Vectorize input text
   - Apply LDA model to get topic distributions
   - Return dominant topic with confidence scores

4. **Batch Classification**: Process multiple texts efficiently
   - Vectorize all texts at once
   - Apply LDA model in batch
   - Return topic distributions for each text

5. **Training**: Fit the model on a corpus
   - Transform texts to TF-IDF features
   - Fit LDA model on the corpus
   - Extract topic words and update labels
"""

import importlib
from typing import List, Optional, Any, Dict, ClassVar, Union

from pydantic import ConfigDict
from sifaka.classifiers.base import (
    Classifier,
    ClassifierImplementation,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.classifiers.config import standardize_classifier_config
from sifaka.utils.logging import get_logger
from sifaka.utils.state import ClassifierState

logger = get_logger(__name__)


class TopicClassifierImplementation:
    """
    Implementation of topic classification logic using scikit-learn's LDA.

    This implementation uses Latent Dirichlet Allocation (LDA) from scikit-learn
    to extract topics from text. It provides a comprehensive topic modeling system
    that can identify the dominant topics in text and return confidence scores
    for each topic.

    ## Architecture

    TopicClassifierImplementation follows the composition pattern:
    1. **Core Logic**: classify_impl() implements topic modeling
    2. **State Management**: Uses ClassifierState for internal state
    3. **Resource Management**: Loads and manages scikit-learn models

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Store configuration parameters
       - Initialize state

    2. **Warm-up**: Prepare resources for classification
       - Load scikit-learn dependencies
       - Initialize vectorizer and LDA model
       - Set up topic-word matrix

    3. **Classification**: Process text inputs
       - Vectorize input text
       - Apply LDA model to get topic distributions
       - Return dominant topic with confidence scores

    4. **Batch Classification**: Process multiple texts efficiently
       - Vectorize all texts at once
       - Apply LDA model in batch
       - Return topic distributions for each text

    5. **Training**: Fit the model on a corpus
       - Transform texts to TF-IDF features
       - Fit LDA model on the corpus
       - Extract topic words and update labels

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    # Class constants
    DEFAULT_COST: ClassVar[float] = 2.0

    def __init__(self, config: ClassifierConfig):
        """
        Initialize the topic classifier implementation.

        Args:
            config: Configuration for the classifier
        """
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False
        self._state.cache = {}

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

    def warm_up_impl(self) -> None:
        """
        Warm up the classifier by initializing the LDA model.

        This method prepares the classifier for use by:
        1. Loading required dependencies
        2. Initializing the vectorizer and LDA model
        3. Setting up the topic-word matrix

        Raises:
            RuntimeError: If model initialization fails
        """
        # Check if already initialized
        if not self._state.initialized:
            # Load dependencies
            sklearn = self._load_dependencies()
            self._state.dependencies_loaded = True

            # Get configuration from params
            max_features = self.config.params.get("max_features", 1000)
            random_state = self.config.params.get("random_state", 42)

            # Create vectorizer
            self._state.vectorizer = sklearn["feature_extraction_text"].TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
            )

            # Create LDA model
            self._state.model = sklearn["decomposition"].LatentDirichletAllocation(
                n_components=self.num_topics,
                random_state=random_state,
            )

            # Mark as initialized
            self._state.initialized = True

    def fit_impl(self, texts: List[str]) -> Dict[str, Any]:
        """
        Fit the topic model on a corpus of texts.

        Args:
            texts: List of texts to fit the model on

        Returns:
            Dictionary with updated configuration and state information
        """
        # Ensure initialized
        if not self._state.initialized:
            self.warm_up_impl()

        # Transform texts to TF-IDF features
        X = self._state.vectorizer.fit_transform(texts)

        # Fit LDA model
        self._state.model.fit(X)

        # Get feature names for later interpretation
        feature_names = self._state.vectorizer.get_feature_names_out()
        self._state.feature_names = feature_names

        # Extract the words that define each topic
        topic_words = []
        for _, topic in enumerate(self._state.model.components_):
            top_features_ind = topic.argsort()[: -self.top_words_per_topic - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_words.append(top_features)

        # Store topic words in state cache
        self._state.cache["topic_words"] = topic_words

        # Create updated labels with meaningful topic names
        labels = [f"topic_{i}_{'+'.join(words[:3])}" for i, words in enumerate(topic_words)]

        # Return updated configuration
        return {
            "labels": labels,
            "topic_words": topic_words,
        }

    def classify_impl(self, text: str) -> ClassificationResult[str]:
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
        if not self._state.model or not self._state.vectorizer:
            raise RuntimeError(
                "Model not initialized. You must call fit() with a corpus before classification."
            )

        # Transform the input text
        X = self._state.vectorizer.transform([text])

        # Get topic distribution
        topic_distribution = self._state.model.transform(X)[0]

        # Get dominant topic
        dominant_topic_idx = topic_distribution.argmax()
        confidence = float(topic_distribution[dominant_topic_idx])

        # Get topic words from state cache
        topic_words = self._state.cache.get("topic_words", [])

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

        return ClassificationResult[str](
            label=labels[dominant_topic_idx],
            confidence=confidence,
            metadata=metadata,
        )

    def batch_classify_impl(self, texts: List[str]) -> List[ClassificationResult[str]]:
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
        if not self._state.model or not self._state.vectorizer:
            raise RuntimeError(
                "Model not initialized. You must call fit() with a corpus before classification."
            )

        # Transform all texts at once
        X = self._state.vectorizer.transform(texts)

        # Get topic distributions for all texts
        topic_distributions = self._state.model.transform(X)

        # Get topic words from state cache
        topic_words = self._state.cache.get("topic_words", [])

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
                ClassificationResult[str](
                    label=labels[dominant_topic_idx],
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return results


def create_pretrained_topic_classifier(
    corpus: List[str],
    name: str = "pretrained_topic_classifier",
    description: str = "Pre-trained topic classifier",
    num_topics: int = 5,
    min_confidence: float = 0.1,
    max_features: int = 1000,
    random_state: int = 42,
    top_words_per_topic: int = 10,
    cache_size: int = 100,
    cost: float = TopicClassifierImplementation.DEFAULT_COST,
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Create a pre-trained topic classifier from a corpus.

    This factory function creates and trains a topic classifier with the provided
    corpus in one step. It handles the creation of the ClassifierConfig object,
    initializes the implementation, and trains the model on the provided corpus.

    Args:
        corpus: List of texts to train the model on
        name: Name for the classifier
        description: Description of the classifier
        num_topics: Number of topics to extract
        min_confidence: Minimum confidence threshold
        max_features: Maximum number of features for the vectorizer
        random_state: Random state for reproducibility
        top_words_per_topic: Number of words to extract per topic
        cache_size: Size of the cache for memoization
        cost: Cost of running the classifier
        config: Optional classifier configuration
        **kwargs: Additional keyword arguments for configuration

    Returns:
        A trained Classifier instance

    Examples:
        ```python
        from sifaka.classifiers.topic import create_pretrained_topic_classifier

        # Create and train a topic classifier in one step
        corpus = ["This is a document about technology.",
                 "This is a document about sports.",
                 "This is a document about politics."]

        classifier = create_pretrained_topic_classifier(
            corpus=corpus,
            name="my_topic_classifier",
            description="Custom topic classifier for my domain",
            num_topics=5,
            min_confidence=0.2
        )

        # Use the pre-trained classifier
        result = classifier.classify("This is a document about artificial intelligence.")
        print(f"Topic: {result.label}, Confidence: {result.confidence:.2f}")
        ```

    Raises:
        ValueError: If corpus is empty or invalid
        RuntimeError: If model training fails
    """
    # Create default config if not provided
    if config is None:
        # Extract configuration parameters
        params = kwargs.pop("params", {})
        params.update(
            {
                "num_topics": num_topics,
                "max_features": max_features,
                "random_state": random_state,
                "top_words_per_topic": top_words_per_topic,
            }
        )

        # Create config with standardized approach
        config = standardize_classifier_config(
            labels=[f"topic_{i}" for i in range(num_topics)],
            cost=cost,
            min_confidence=min_confidence,
            cache_size=cache_size,
            params=params,
            **kwargs,
        )

    # Create implementation
    implementation = TopicClassifierImplementation(config)

    # Initialize the implementation
    implementation.warm_up_impl()

    # Create classifier
    classifier = Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )

    # Train the classifier
    if corpus:
        # Fit the model
        result = implementation.fit_impl(corpus)

        # Update labels if available
        if "labels" in result:
            # Create new config with updated labels
            new_config = ClassifierConfig(
                labels=result["labels"],
                cost=config.cost,
                min_confidence=config.min_confidence,
                cache_size=config.cache_size,
                params=config.params,
            )

            # Create new classifier with updated config
            classifier = Classifier(
                name=name,
                description=description,
                config=new_config,
                implementation=implementation,
            )

    return classifier


def create_topic_classifier(
    name: str = "topic_classifier",
    description: str = "Classifies text into topics using LDA",
    num_topics: int = 5,
    min_confidence: float = 0.1,
    max_features: int = 1000,
    random_state: int = 42,
    top_words_per_topic: int = 10,
    cache_size: int = 100,
    cost: float = TopicClassifierImplementation.DEFAULT_COST,
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Create a topic classifier.

    This factory function creates a topic classifier with the specified
    configuration options. It handles the creation of the ClassifierConfig object
    and setting up the classifier with the appropriate parameters.

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
        A Classifier instance

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

        # To train the classifier, use create_pretrained_topic_classifier
        # or manually fit the implementation

        # Classify text
        result = classifier.classify("This is a document about artificial intelligence.")
        print(f"Topic: {result.label}, Confidence: {result.confidence:.2f}")
        ```
    """
    # Prepare params
    params = kwargs.pop("params", {})
    params.update(
        {
            "num_topics": num_topics,
            "max_features": max_features,
            "random_state": random_state,
            "top_words_per_topic": top_words_per_topic,
        }
    )

    # Create config
    classifier_config = standardize_classifier_config(
        config=config,
        labels=[f"topic_{i}" for i in range(num_topics)],
        min_confidence=min_confidence,
        cost=cost,
        cache_size=cache_size,
        params=params,
        **kwargs,
    )

    # Create implementation
    implementation = TopicClassifierImplementation(classifier_config)

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=classifier_config,
        implementation=implementation,
    )
