"""
Topic classifier using scikit-learn's LDA.
"""

import importlib
from dataclasses import dataclass
from typing import List, Optional

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TopicConfig:
    """
    Configuration for topic classification.

    Note: This class is provided for backward compatibility.
    The preferred way to configure topic classification is to use
    ClassifierConfig with params:

    ```python
    config = ClassifierConfig(
        labels=["topic_0", "topic_1", "topic_2", "topic_3", "topic_4"],
        cost=2.0,
        params={
            "num_topics": 5,
            "min_confidence": 0.1,
            "max_features": 1000,
            "random_state": 42,
            "top_words_per_topic": 10,
        }
    )
    ```
    """

    num_topics: int = 5  # Number of topics to extract
    min_confidence: float = 0.1  # Minimum confidence threshold
    max_features: int = 1000  # Max features for vectorization
    random_state: int = 42  # For reproducibility
    top_words_per_topic: int = 10  # Number of top words to include in topic representation

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_topics <= 0:
            raise ValueError("num_topics must be positive")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        if self.top_words_per_topic <= 0:
            raise ValueError("top_words_per_topic must be positive")


class TopicClassifier(BaseClassifier):
    """
    A topic classifier using Latent Dirichlet Allocation from scikit-learn.

    This classifier extracts topics from text using LDA and returns the
    dominant topic with confidence scores for each topic.

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    def __init__(
        self,
        name: str = "topic_classifier",
        description: str = "Classifies text into topics using LDA",
        topic_config: Optional[TopicConfig] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the topic classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            topic_config: Topic classification configuration
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Create config if not provided
        if config is None:
            # Extract params from kwargs if present
            params = kwargs.pop("params", {})

            # If topic_config is provided, add its values to params
            if topic_config is not None:
                params.update(
                    {
                        "num_topics": topic_config.num_topics,
                        "min_confidence": topic_config.min_confidence,
                        "max_features": topic_config.max_features,
                        "random_state": topic_config.random_state,
                        "top_words_per_topic": topic_config.top_words_per_topic,
                    }
                )

            # Get num_topics from params or use default
            num_topics = params["num_topics"] if "num_topics" in params else 5
            min_confidence = params["min_confidence"] if "min_confidence" in params else 0.1

            # Create config with remaining kwargs
            config = ClassifierConfig(
                labels=[f"topic_{i}" for i in range(num_topics)],
                cost=2.0,  # Default cost
                min_confidence=min_confidence,
                params=params,
                **kwargs,
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

        # Initialize other attributes
        self._vectorizer = None
        self._model = None
        self._feature_names = None
        self._topic_words = None
        self._initialized = False

    @property
    def num_topics(self) -> int:
        """Get the number of topics."""
        return self.config.params["num_topics"] if "num_topics" in self.config.params else 5

    @property
    def top_words_per_topic(self) -> int:
        """Get the number of top words per topic."""
        return (
            self.config.params["top_words_per_topic"]
            if "top_words_per_topic" in self.config.params
            else 10
        )

    def _load_dependencies(self) -> None:
        """Load scikit-learn dependencies."""
        try:
            # Import necessary scikit-learn modules
            self._sklearn_feature_extraction_text = importlib.import_module(
                "sklearn.feature_extraction.text"
            )
            self._sklearn_decomposition = importlib.import_module("sklearn.decomposition")
            return True
        except ImportError:
            raise ImportError(
                "scikit-learn is required for TopicClassifier. "
                "Install it with: pip install scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn modules: {e}")

    def warm_up(self) -> None:
        """
        Initialize the model if not already initialized.
        """
        if not self._initialized:
            self._load_dependencies()
            # Get configuration from params
            max_features = (
                self.config.params["max_features"] if "max_features" in self.config.params else 1000
            )
            random_state = (
                self.config.params["random_state"] if "random_state" in self.config.params else 42
            )

            self._vectorizer = self._sklearn_feature_extraction_text.TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
            )
            self._model = self._sklearn_decomposition.LatentDirichletAllocation(
                n_components=self.num_topics,
                random_state=random_state,
            )
            self._initialized = True

    def fit(self, texts: List[str]) -> "TopicClassifier":
        """
        Fit the topic model on a corpus of texts.

        Args:
            texts: List of texts to fit the model on

        Returns:
            self: The fitted classifier
        """
        self.warm_up()

        # Transform texts to TF-IDF features
        X = self._vectorizer.fit_transform(texts)

        # Fit LDA model
        self._model.fit(X)

        # Get feature names for later interpretation
        self._feature_names = self._vectorizer.get_feature_names_out()

        # Extract the words that define each topic
        self._topic_words = []
        for _, topic in enumerate(self._model.components_):
            top_features_ind = topic.argsort()[: -self.top_words_per_topic - 1 : -1]
            top_features = [self._feature_names[i] for i in top_features_ind]
            self._topic_words.append(top_features)

        # Update labels with meaningful topic names
        labels = [f"topic_{i}_{'+'.join(words[:3])}" for i, words in enumerate(self._topic_words)]
        self.config = ClassifierConfig(
            labels=labels,
            cost=self.config.cost,
            min_confidence=self.config.min_confidence,
            params=self.config.params,
        )

        return self

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement topic classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with detected topic and confidence
        """
        if not self._model or not self._vectorizer:
            raise RuntimeError(
                "Model not initialized. You must call fit() with a corpus before classification."
            )

        # Transform the input text
        X = self._vectorizer.transform([text])

        # Get topic distribution
        topic_distribution = self._model.transform(X)[0]

        # Get dominant topic
        dominant_topic_idx = topic_distribution.argmax()
        confidence = float(topic_distribution[dominant_topic_idx])

        # Create detailed metadata
        all_topics = {}
        labels = list(self.config.labels)  # Convert FieldInfo to list
        for i, prob in enumerate(topic_distribution):
            all_topics[labels[i]] = {
                "probability": float(prob),
                "words": self._topic_words[i][: self.top_words_per_topic],
            }

        metadata = {
            "all_topics": all_topics,
            "topic_words": self._topic_words[dominant_topic_idx],
            "topic_distribution": topic_distribution.tolist(),
        }

        return ClassificationResult(
            label=labels[dominant_topic_idx],
            confidence=confidence,
            metadata=metadata,
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        self.validate_batch_input(texts)

        if not self._model or not self._vectorizer:
            raise RuntimeError(
                "Model not initialized. You must call fit() with a corpus before classification."
            )

        # Transform all texts at once
        X = self._vectorizer.transform(texts)

        # Get topic distributions for all texts
        topic_distributions = self._model.transform(X)

        results = []
        labels = list(self.config.labels)  # Convert FieldInfo to list
        for distribution in topic_distributions:
            dominant_topic_idx = distribution.argmax()
            confidence = float(distribution[dominant_topic_idx])

            all_topics = {}
            for j, prob in enumerate(distribution):
                all_topics[labels[j]] = {
                    "probability": float(prob),
                    "words": self._topic_words[j][: self.top_words_per_topic],
                }

            metadata = {
                "all_topics": all_topics,
                "topic_words": self._topic_words[dominant_topic_idx],
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
        topic_config: Optional[TopicConfig] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> "TopicClassifier":
        """
        Create and train a topic classifier in one step.

        Args:
            corpus: Corpus of texts to train on
            name: Name of the classifier
            description: Description of the classifier
            topic_config: Topic classification configuration (for backward compatibility)
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Trained TopicClassifier
        """
        # If topic_config is provided but config is not, create config from topic_config
        if topic_config is not None and config is None:
            # Extract params from topic_config
            params = {
                "num_topics": topic_config.num_topics,
                "min_confidence": topic_config.min_confidence,
                "max_features": topic_config.max_features,
                "random_state": topic_config.random_state,
                "top_words_per_topic": topic_config.top_words_per_topic,
            }

            # Create config with params
            labels = [f"topic_{i}" for i in range(topic_config.num_topics)]
            config = ClassifierConfig(
                labels=labels,
                cost=2.0,
                min_confidence=topic_config.min_confidence,
                params=params,
            )

        # Create instance with provided configuration
        classifier = cls(
            name=name, description=description, topic_config=topic_config, config=config, **kwargs
        )

        # Train the classifier and return it
        return classifier.fit(corpus)
