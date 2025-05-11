"""
Topic classifier using scikit-learn's LDA.

This module provides a classifier for extracting topics from text using
Latent Dirichlet Allocation (LDA) from scikit-learn. The classifier can
identify dominant topics in text and provide detailed topic distributions
with representative words for each topic.

## Overview
The TopicClassifier is a specialized classifier that extracts latent topics
from text documents using unsupervised learning. It leverages scikit-learn's
implementation of Latent Dirichlet Allocation (LDA) to identify patterns in
text and group them into coherent topics. The classifier can be trained on
a corpus of documents and then used to classify new text into the discovered
topics.

## Architecture
TopicClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods
2. **Caching Layer**: _classify_impl() handles caching
3. **Core Logic**: LDA model for topic extraction
4. **State Management**: Uses StateManager for internal state
5. **Training**: fit() method for model training
6. **Warm-up**: Lazy initialization of dependencies

## Lifecycle
1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract parameters from config.params
   - Set up default values for topics and features

2. **Warm-up**: Load scikit-learn dependencies
   - Import necessary modules on demand
   - Initialize vectorizer and LDA model
   - Handle initialization errors gracefully

3. **Training**: Fit the model on a corpus
   - Vectorize the corpus using TF-IDF
   - Train the LDA model on the vectorized corpus
   - Extract representative words for each topic
   - Update labels with meaningful topic names

4. **Classification**: Process input text
   - Vectorize the input text
   - Apply the trained LDA model
   - Extract topic distributions
   - Identify the dominant topic
   - Return detailed topic information

## Usage Examples
```python
from sifaka.classifiers.implementations.properties.topic import create_topic_classifier

# Create a topic classifier
classifier = create_topic_classifier(num_topics=5)

# Train the classifier on a corpus
corpus = [
    "This is a document about technology and computers.",
    "This is a document about sports and athletics.",
    "This is a document about politics and government.",
    "This is a document about health and medicine.",
    "This is a document about entertainment and movies."
]
classifier.fit(corpus)

# Classify a new document
result = classifier.classify("This document discusses artificial intelligence and machine learning.")
print(f"Topic: {result.label}, Confidence: {result.confidence:.2f}")

# Access topic details
for topic, details in result.metadata['all_topics'].items():
    print(f"{topic}: {details['probability']:.2f}")
    print(f"Words: {', '.join(details['words'])}")
```

## Error Handling
The classifier provides robust error handling:
- ImportError: When scikit-learn is not installed
- RuntimeError: When model initialization fails
- ValueError: When input text is empty or invalid
- Graceful handling of edge cases during classification

## Configuration
Key configuration options include:
- num_topics: Number of topics to extract (default: 5)
- min_confidence: Threshold for topic confidence (default: 0.1)
- max_features: Maximum number of features for vectorization (default: 1000)
- random_state: Random seed for reproducible results (default: 42)
- top_words_per_topic: Number of words to extract per topic (default: 10)
"""

# import os
# import pickle
import importlib
from typing import List, Optional, Any, Dict, ClassVar

from pydantic import ConfigDict
from sifaka.classifiers.classifier import Classifier
from sifaka.core.results import ClassificationResult
from sifaka.utils.config import ClassifierConfig, standardize_classifier_config
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state

logger = get_logger(__name__)


class TopicClassifier(Classifier):
    """
    A topic classifier using Latent Dirichlet Allocation from scikit-learn.

    This classifier extracts topics from text using LDA and returns the
    dominant topic with confidence scores for each topic. It can be trained
    on a corpus of documents to discover latent topics and then classify
    new text into these topics.

    ## Architecture
    TopicClassifier follows a component-based architecture:
    - Extends the base Classifier class for consistent interface
    - Uses scikit-learn's LDA for topic modeling
    - Implements TF-IDF vectorization for text preprocessing
    - Provides detailed topic distributions and word representations
    - Uses StateManager for efficient state tracking and caching
    - Supports both individual and batch classification
    - Implements lazy loading of dependencies for efficiency

    ## Lifecycle
    1. **Initialization**: Set up configuration and parameters
       - Initialize with name, description, and config
       - Extract parameters from config.params
       - Set up default values for topics and features
       - Initialize state cache for model storage

    2. **Warm-up**: Load scikit-learn dependencies
       - Import necessary modules on demand
       - Initialize vectorizer and LDA model
       - Handle initialization errors with clear messages
       - Mark initialization status in state

    3. **Training**: Fit the model on a corpus
       - Vectorize the corpus using TF-IDF
       - Train the LDA model on the vectorized corpus
       - Extract representative words for each topic
       - Update labels with meaningful topic names
       - Store topic words in state cache

    4. **Classification**: Process input text
       - Vectorize the input text
       - Apply the trained LDA model
       - Extract topic distributions
       - Identify the dominant topic
       - Return detailed topic information with confidence scores

    ## Examples
    ```python
    from sifaka.classifiers.implementations.properties.topic import TopicClassifier

    # Create a topic classifier
    classifier = TopicClassifier(
        name="my_topic_classifier",
        description="Custom topic classifier for document analysis"
    )

    # Train the classifier on a corpus
    corpus = [
        "This is a document about technology and computers.",
        "This is a document about sports and athletics.",
        "This is a document about politics and government."
    ]
    classifier.fit(corpus)

    # Classify a new document
    result = classifier.classify("This document discusses artificial intelligence.")
    print(f"Topic: {result.label}, Confidence: {result.confidence:.2f}")

    # Access topic details
    for topic, details in result.metadata['all_topics'].items():
        print(f"{topic}: {details['probability']:.2f}")
        print(f"Words: {', '.join(details['words'])}")
    ```

    ## Configuration Options
    - num_topics: Number of topics to extract (default: 5)
    - min_confidence: Threshold for topic confidence (default: 0.1)
    - max_features: Maximum number of features for vectorization (default: 1000)
    - random_state: Random seed for reproducible results (default: 42)
    - top_words_per_topic: Number of words to extract per topic (default: 10)

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    # Class constants
    DEFAULT_COST: ClassVar[float] = 2.0

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = [
        "technology",
        "business",
        "politics",
        "entertainment",
        "health",
        "science",
        "sports",
        "other",
    ]

    # State is inherited from BaseClassifier as _state_manager

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        name: str = "topic_classifier",
        description: str = "Classifies text by topic",
        config: Optional[ClassifierConfig[str]] = None,
        **kwargs: Any,
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
        state = self._state_manager.get("cache", {})
        state["initialized"] = False
        state["model"] = None
        self._state_manager.update("cache", state)

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
        state = self._state_manager.get("cache", {})

        # Check if already initialized
        if not state.get("initialized"):
            # Load dependencies
            sklearn = self._load_dependencies()
            state["dependencies_loaded"] = True

            # Get configuration from params
            max_features = self.config.params.get("max_features", 1000)
            random_state = self.config.params.get("random_state", 42)

            # Create vectorizer
            state["vectorizer"] = sklearn["feature_extraction_text"].TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
            )

            # Create LDA model
            state["model"] = sklearn["decomposition"].LatentDirichletAllocation(
                n_components=self.num_topics,
                random_state=random_state,
            )

            # Mark as initialized
            state["initialized"] = True

    def fit(self, texts: List[str]) -> "TopicClassifier":
        """
        Fit the topic model on a corpus of texts.

        Args:
            texts: List of texts to fit the model on

        Returns:
            self: The fitted classifier
        """
        # Get state
        state = self._state_manager.get("cache", {})

        # Ensure initialized
        self.warm_up()

        # Transform texts to TF-IDF features
        X = state["vectorizer"].fit_transform(texts)

        # Fit LDA model
        state["model"].fit(X)

        # Get feature names for later interpretation
        feature_names = state["vectorizer"].get_feature_names_out()
        state["feature_names"] = feature_names

        # Extract the words that define each topic
        topic_words = []
        for _, topic in enumerate(state["model"].components_):
            top_features_ind = topic.argsort()[: -self.top_words_per_topic - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_words.append(top_features)

        # Store topic words in state cache
        state["cache"]["topic_words"] = topic_words

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
        state = self._state_manager.get("cache", {})

        if not state.get("model") or not state.get("vectorizer"):
            raise RuntimeError(
                "Model not initialized. You must call fit() with a corpus before classification."
            )

        # Transform the input text
        X = state["vectorizer"].transform([text])

        # Get topic distribution
        topic_distribution = state["model"].transform(X)[0]

        # Get dominant topic
        dominant_topic_idx = topic_distribution.argmax()
        confidence = float(topic_distribution[dominant_topic_idx])

        # Get topic words from state cache
        topic_words = state["cache"].get("topic_words", [])

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
        state = self._state_manager.get("cache", {})

        self.validate_batch_input(texts)

        if not state.get("model") or not state.get("vectorizer"):
            raise RuntimeError(
                "Model not initialized. You must call fit() with a corpus before classification."
            )

        # Transform all texts at once
        X = state["vectorizer"].transform(texts)

        # Get topic distributions for all texts
        topic_distributions = state["model"].transform(X)

        # Get topic words from state cache
        topic_words = state["cache"].get("topic_words", [])

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
        config: Optional[Dict[str, Any]] = None,
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
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> TopicClassifier:
    """
    Factory function to create a topic classifier.

    This function provides a simpler interface for creating a topic classifier
    with the specified parameters, handling the creation of the ClassifierConfig
    object and setting up the classifier with the appropriate parameters.

    ## Architecture
    The factory function follows a standardized pattern:
    1. Extract and prepare parameters for configuration
    2. Create a configuration dictionary with standardized structure
    3. Pass the configuration to the classifier constructor
    4. Return the fully configured classifier instance

    ## Examples
    ```python
    from sifaka.classifiers.implementations.properties.topic import create_topic_classifier

    # Create with default settings
    classifier = create_topic_classifier()

    # Create with custom topic settings
    custom_classifier = create_topic_classifier(
        num_topics=10,           # Extract 10 topics instead of default 5
        max_features=2000,       # Use more features for better topic modeling
        top_words_per_topic=15,  # Extract more words per topic
        random_state=42          # Set random seed for reproducible results
    )

    # Create with custom name and description
    named_classifier = create_topic_classifier(
        name="custom_topic_classifier",
        description="Custom topic classifier for document analysis",
        min_confidence=0.2       # Higher threshold for topic confidence
    )
    ```

    Args:
        name: Name of the classifier for identification and logging
        description: Human-readable description of the classifier's purpose
        num_topics: Number of topics to extract from the text corpus
        min_confidence: Minimum confidence threshold for topic assignment
        max_features: Maximum number of features for the TF-IDF vectorizer
        random_state: Random seed for reproducible topic modeling results
        top_words_per_topic: Number of representative words to extract per topic
        cache_size: Size of the classification cache (0 to disable caching)
        cost: Computational cost metric for resource allocation decisions
        config: Optional classifier configuration (dict or ClassifierConfig)
        **kwargs: Additional configuration parameters to pass to the classifier

    Returns:
        Configured TopicClassifier instance ready for training and use

    Examples:
        ```python
        from sifaka.classifiers.implementations.properties.topic import create_topic_classifier

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
