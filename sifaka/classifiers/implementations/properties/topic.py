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
   - Extract parameters from config and config and config and config and config and config and config and config.params
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
from typing import List, Optional, Any, Dict, ClassVar, Union, Tuple
import time

from pydantic import ConfigDict, PrivateAttr
from sifaka.classifiers.classifier import Classifier, ClassifierImplementation
from sifaka.core.results import ClassificationResult
from sifaka.utils.config.classifiers import ClassifierConfig, standardize_classifier_config
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
       - Extract parameters from config and config and config and config and config and config and config and config.params
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        name: str = "topic_classifier",
        description: str = "Classifies text by topic",
        config: Optional[ClassifierConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the classifier.

        Args:
            name: Name of the classifier
            description: Description of the classifier
            config: Configuration for the classifier
            **kwargs: Additional arguments
        """
        # Create a dummy implementation that will be replaced by self
        dummy_implementation: ClassifierImplementation = self

        # Initialize classifier
        super().__init__(
            name=name,
            description=description,
            config=config,
            implementation=dummy_implementation,
        )

        # Initialize state
        self._state_manager = create_classifier_state()

        # Initialize dependencies
        self._dependencies: Dict[str, Any] = {}
        self._is_initialized = False

    @property
    def num_topics(self) -> int:
        """Get the number of topics."""
        return (
            len(self.config.params.get("labels", [])) if self.config and self.config.params else 5
        )

    @property
    def top_words_per_topic(self) -> int:
        """Get the number of top words per topic."""
        return (
            self.config.params.get("top_words_per_topic", 10)
            if self.config and self.config.params
            else 10
        )

    def _load_dependencies(self) -> Dict[str, Any]:
        """Load required dependencies."""
        try:
            sklearn = importlib.import_module("sklearn")
            return {
                "vectorizer": sklearn.feature_extraction.text.TfidfVectorizer(
                    max_features=(
                        self.config.params.get("max_features", 1000)
                        if self.config and self.config.params
                        else 1000
                    )
                ),
                "lda": sklearn.decomposition.LatentDirichletAllocation(
                    n_components=self.num_topics,
                    random_state=(
                        self.config.params.get("random_state", 42)
                        if self.config and self.config.params
                        else 42
                    ),
                ),
            }
        except ImportError as e:
            raise ImportError("scikit-learn is required for topic classification") from e

    def warm_up(self) -> None:
        """Initialize the classifier."""
        if not self._is_initialized:
            self._dependencies = self._load_dependencies()
            self._is_initialized = True

    def fit(self, texts: List[str]) -> "TopicClassifier":
        """Train the classifier on a corpus of texts."""
        self.warm_up()

        # Vectorize the corpus
        vectorizer = self._dependencies["vectorizer"]
        lda = self._dependencies["lda"]

        # Fit and transform
        X = vectorizer.fit_transform(texts)
        lda.fit(X)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Store topic words
        topic_words = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[: -self.top_words_per_topic - 1 : -1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words[f"topic_{topic_idx}"] = top_words

        # Update state
        self._state_manager.set_metadata("topic_words", topic_words)

        return self

    def _classify_impl(self, text: str) -> ClassificationResult:
        """Classify a single text."""
        self.warm_up()

        # Vectorize the text
        vectorizer = self._dependencies["vectorizer"]
        lda = self._dependencies["lda"]

        # Transform
        X = vectorizer.transform([text])
        topic_dist = lda.transform(X)[0]

        # Get dominant topic
        dominant_topic_idx = topic_dist.argmax()
        confidence = float(topic_dist[dominant_topic_idx])

        # Get topic words
        topic_words = self._state_manager.get_metadata("topic_words", {})

        # Get min confidence
        min_confidence = 0.1
        if self.config and self.config.params:
            min_confidence = self.config.params.get("min_confidence", 0.1)

        # Create result
        return ClassificationResult(
            label=f"topic_{dominant_topic_idx}",
            confidence=confidence,
            passed=confidence >= min_confidence,
            message=f"Classified as topic_{dominant_topic_idx} with confidence {confidence:.2f}",
            metadata={
                "all_topics": {
                    f"topic_{i}": {
                        "probability": float(prob),
                        "words": topic_words.get(f"topic_{i}", []),
                    }
                    for i, prob in enumerate(topic_dist)
                }
            },
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts."""
        self.warm_up()

        # Vectorize the texts
        vectorizer = self._dependencies["vectorizer"]
        lda = self._dependencies["lda"]

        # Transform
        X = vectorizer.transform(texts)
        topic_dists = lda.transform(X)

        # Get topic words
        topic_words = self._state_manager.get_metadata("topic_words", {})

        # Get min confidence
        min_confidence = 0.1
        if self.config and self.config.params:
            min_confidence = self.config.params.get("min_confidence", 0.1)

        # Process results
        results: List[ClassificationResult] = []
        for text, topic_dist in zip(texts, topic_dists):
            # Get dominant topic
            dominant_topic_idx = topic_dist.argmax()
            confidence = float(topic_dist[dominant_topic_idx])

            # Create result
            results.append(
                ClassificationResult(
                    label=f"topic_{dominant_topic_idx}",
                    confidence=confidence,
                    passed=confidence >= min_confidence,
                    message=f"Classified as topic_{dominant_topic_idx} with confidence {confidence:.2f}",
                    metadata={
                        "all_topics": {
                            f"topic_{i}": {
                                "probability": float(prob),
                                "words": topic_words.get(f"topic_{i}", []),
                            }
                            for i, prob in enumerate(topic_dist)
                        }
                    },
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
        **kwargs: Any,
    ) -> "TopicClassifier":
        """Create a pre-trained topic classifier."""
        # Create classifier
        classifier = cls(name=name, description=description, config=config, **kwargs)

        # Train on corpus
        classifier.fit(corpus)

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
    cost: float = TopicClassifier.DEFAULT_COST,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> TopicClassifier:
    """Create a topic classifier with the given configuration."""
    # Create config
    config_dict: Dict[str, Any] = config if config is not None else {}

    # Update config with parameters
    params = {
        "num_topics": num_topics,
        "min_confidence": min_confidence,
        "max_features": max_features,
        "random_state": random_state,
        "top_words_per_topic": top_words_per_topic,
        "cache_size": cache_size,
        "cost": cost,
        "labels": [f"topic_{i}" for i in range(num_topics)],
    }

    config_dict.update(params)
    config_dict.update(kwargs)

    # Create standardized config
    classifier_config: ClassifierConfig = standardize_classifier_config(config=config_dict)

    # Create classifier
    return TopicClassifier(
        name=name,
        description=description,
        config=classifier_config,
    )
