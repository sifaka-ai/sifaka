"""
Genre classifier using scikit-learn's RandomForest.

This module provides a classifier for categorizing text into different genres
such as news, fiction, academic, technical, etc. using textual features.
"""

import importlib
import os
import pickle
from typing import Dict, List, Optional, Any, ClassVar

from pydantic import ConfigDict

from sifaka.classifiers.base import BaseClassifier
from sifaka.classifiers.models import ClassificationResult
from sifaka.classifiers.config import ClassifierConfig, standardize_classifier_config
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state

logger = get_logger(__name__)

# Default genres to classify
DEFAULT_GENRES: List[str] = [
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


class GenreClassifier(BaseClassifier[str, str]):
    """
    A genre classifier using RandomForest from scikit-learn.

    This classifier categorizes text into different genres such as news,
    fiction, academic, technical, etc. using textual features.

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = [
        "fiction",
        "non-fiction",
        "poetry",
        "technical",
        "academic",
        "news",
        "marketing",
        "review",
        "unknown",
    ]
    DEFAULT_COST: ClassVar[int] = 1  # Low cost for rule-based classification

    # State is inherited from BaseClassifier as _state_manager

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        validate_assignment=True,
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize the model after Pydantic initialization."""
        super().model_post_init(__context)

        # Create config if not provided
        if not self.config:
            # Create default config
            self.config = ClassifierConfig(
                labels=DEFAULT_GENRES,
                cost=self.DEFAULT_COST,
                min_confidence=0.6,
                params={
                    "min_confidence": 0.6,
                    "max_features": 2000,
                    "random_state": 42,
                    "model_path": None,
                    "use_ngrams": True,
                    "n_estimators": 100,
                    "default_genres": DEFAULT_GENRES,
                },
            )

        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False
        state.cache = {}

    def _load_dependencies(self) -> Dict[str, Any]:
        """
        Load required dependencies for genre classification.

        This method imports and initializes the necessary scikit-learn
        components for text classification.

        Returns:
            Dictionary containing loaded dependencies:
            - 'CountVectorizer': For text vectorization
            - 'RandomForestClassifier': For genre classification
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
                "ensemble": importlib.import_module("sklearn.ensemble"),
                "pipeline": importlib.import_module("sklearn.pipeline"),
            }
            return sklearn_modules
        except ImportError:
            raise ImportError(
                "scikit-learn is required for GenreClassifier. "
                "Install it with: pip install scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn modules: {e}")

    def warm_up(self) -> None:
        """
        Warm up the classifier by initializing the RandomForest model.

        This method prepares the classifier for use by:
        1. Loading required dependencies
        2. Initializing the vectorizer and classifier
        3. Loading a pre-trained model if available

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
            model_path = self.config.params.get("model_path")
            use_ngrams = self.config.params.get("use_ngrams", True)
            max_features = self.config.params.get("max_features", 2000)
            n_estimators = self.config.params.get("n_estimators", 100)
            random_state = self.config.params.get("random_state", 42)

            if model_path and os.path.exists(model_path):
                self._load_model(model_path)
            else:
                # Create TF-IDF vectorizer
                ngram_range = (1, 3) if use_ngrams else (1, 1)
                state.vectorizer = sklearn["feature_extraction_text"].TfidfVectorizer(
                    max_features=max_features,
                    stop_words="english",
                    ngram_range=ngram_range,
                )

                # Create RandomForest model
                state.model = sklearn["ensemble"].RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                )

                # Create pipeline
                state.pipeline = sklearn["pipeline"].Pipeline(
                    [
                        ("vectorizer", state.vectorizer),
                        ("classifier", state.model),
                    ]
                )

            # Mark as initialized
            state.initialized = True

    def _save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        This method saves:
        1. The trained RandomForest classifier
        2. The fitted vectorizer
        3. The label encoder

        Args:
            path: Path where to save the model files

        Raises:
            IOError: If model saving fails
            RuntimeError: If model is not trained
        """
        # Get state
        state = self._state_manager.get_state()

        try:
            # Get custom labels from state cache
            custom_labels = state.cache.get("custom_labels") or self.config.params.get(
                "default_genres", self.config.labels
            )

            model_data = {
                "pipeline": state.pipeline,
                "labels": custom_labels,
                "feature_importances": state.cache.get("feature_importances", {}),
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            state.error = f"Failed to save model: {e}"

    def _load_model(self, path: str) -> None:
        """
        Load a pre-trained model from disk.

        This method loads:
        1. The trained RandomForest classifier
        2. The fitted vectorizer
        3. The label encoder

        Args:
            path: Path from where to load the model files

        Raises:
            IOError: If model loading fails
            ValueError: If model files are invalid
        """
        # Get state
        state = self._state_manager.get_state()

        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

                # Extract model data
                state.pipeline = model_data["pipeline"]
                custom_labels = model_data.get("labels")
                state.cache["feature_importances"] = model_data.get("feature_importances", {})
                state.cache["custom_labels"] = custom_labels

                # Update config with loaded labels if available
                if custom_labels:
                    self._config = ClassifierConfig(
                        labels=custom_labels,
                        cost=self.DEFAULT_COST,
                        min_confidence=self.config.min_confidence,
                        params=self.config.params,
                    )

                # Extract vectorizer and model from pipeline
                state.vectorizer = state.pipeline.named_steps["vectorizer"]
                state.model = state.pipeline.named_steps["classifier"]

            logger.info(f"Model loaded from {path}")
            state.initialized = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            state.error = f"Failed to load model: {e}"
            raise RuntimeError(f"Failed to load model: {e}")

    def fit(self, texts: List[str], labels: List[str]) -> "GenreClassifier":
        """
        Fit the genre classifier on a corpus of texts.

        Args:
            texts: List of texts to fit the model on
            labels: List of genre labels

        Returns:
            self: The fitted classifier
        """
        # Get state
        state = self._state_manager.get_state()

        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

        self.warm_up()

        # Get unique labels
        unique_labels = sorted(set(labels))

        # Create label mapping
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_mapping[label] for label in labels]

        # Store custom labels in state cache
        state.cache["custom_labels"] = unique_labels

        # Update config with custom labels
        self._config = ClassifierConfig(
            labels=unique_labels,
            cost=self.DEFAULT_COST,
            min_confidence=self.config.min_confidence,
            params=self.config.params,
        )

        # Fit the pipeline
        state.pipeline.fit(texts, numeric_labels)

        # Extract feature importances
        state.cache["feature_importances"] = self._extract_feature_importances()

        # Save the model if path is provided
        model_path = self.config.params.get("model_path")
        if model_path:
            self._save_model(model_path)

        return self

    def _extract_feature_importances(self) -> Dict[str, Dict[str, float]]:
        """
        Extract feature importances from the trained model.

        This method analyzes the RandomForest model to determine:
        1. Most important features for each genre
        2. Feature importance scores
        3. Word importance rankings

        Returns:
            Dictionary mapping genres to feature importance scores

        Raises:
            RuntimeError: If model is not trained
        """
        # Get state
        state = self._state_manager.get_state()

        if not state.model or not hasattr(state.model, "feature_importances_"):
            return {}

        try:
            feature_names = state.vectorizer.get_feature_names_out()
            importances = state.model.feature_importances_

            # Get the top features
            top_features = {}
            for label in self._config.labels:
                # For RandomForest, we need to extract class-specific importances differently
                # Here we use the overall feature importance as a simplification
                top_indices = importances.argsort()[-20:][::-1]
                top_features[label] = {feature_names[j]: float(importances[j]) for j in top_indices}

            return top_features
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
            state.error = f"Could not extract feature importances: {e}"
            return {}

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Classify a single text into genres.

        This method:
        1. Vectorizes the input text
        2. Applies the RandomForest model
        3. Returns the predicted genre and confidence scores

        Args:
            text: The text to classify

        Returns:
            ClassificationResult containing:
            - predicted_genre: The most likely genre
            - confidence: Confidence score for the prediction
            - genre_distribution: Distribution across all genres
            - feature_importances: Importance of text features

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If classification fails
        """
        # Get state
        state = self._state_manager.get_state()

        if not state.pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Predict probability
        proba = state.pipeline.predict_proba([text])[0]

        # Get dominant class
        dominant_class_idx = proba.argmax()
        confidence = float(proba[dominant_class_idx])

        # Create metadata
        all_probs = {self._config.labels[i]: float(prob) for i, prob in enumerate(proba)}

        # Get top features for explanation
        top_features = {}
        feature_importances = state.cache.get("feature_importances", {})
        if feature_importances and self._config.labels[dominant_class_idx] in feature_importances:
            top_features = feature_importances[self._config.labels[dominant_class_idx]]

        # Get min_confidence from config
        min_confidence = self.config.params.get("min_confidence", 0.6)

        metadata = {
            "probabilities": all_probs,
            "threshold": min_confidence,
            "is_confident": confidence >= min_confidence,
            "top_features": top_features,
        }

        return ClassificationResult(
            label=self._config.labels[dominant_class_idx],
            confidence=confidence,
            metadata=metadata,
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts into genres.

        This method efficiently processes multiple texts by:
        1. Vectorizing all texts at once
        2. Applying the RandomForest model in batch
        3. Returning genre predictions for each text

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

        if not state.pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Predict probabilities for all texts
        probas = state.pipeline.predict_proba(texts)

        results = []
        feature_importances = state.cache.get("feature_importances", {})

        for proba in probas:
            dominant_class_idx = proba.argmax()
            confidence = float(proba[dominant_class_idx])

            all_probs = {self._config.labels[i]: float(prob) for i, prob in enumerate(proba)}

            # Get top features for explanation
            top_features = {}
            if (
                feature_importances
                and self._config.labels[dominant_class_idx] in feature_importances
            ):
                top_features = feature_importances[self._config.labels[dominant_class_idx]]

            # Get min_confidence from config
            min_confidence = self.config.params.get("min_confidence", 0.6)

            metadata = {
                "probabilities": all_probs,
                "threshold": min_confidence,
                "is_confident": confidence >= min_confidence,
                "top_features": top_features,
            }

            results.append(
                ClassificationResult(
                    label=self._config.labels[dominant_class_idx],
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return results

    @classmethod
    def create_pretrained(
        cls,
        texts: List[str],
        labels: List[str],
        name: str = "pretrained_genre_classifier",
        description: str = "Pre-trained genre classifier",
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "GenreClassifier":
        """
        Create a pre-trained genre classifier from labeled data.

        This method:
        1. Creates a new classifier instance
        2. Fits the RandomForest model on the provided data
        3. Returns the trained classifier

        Args:
            texts: List of training texts
            labels: List of corresponding genre labels
            name: Name for the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional keyword arguments for configuration

        Returns:
            A trained GenreClassifier instance

        Raises:
            ValueError: If training data is invalid or inconsistent
            RuntimeError: If model training fails
        """
        # If no config provided, create a default one
        if config is None:
            # Extract configuration parameters
            min_confidence = kwargs.pop("min_confidence", 0.6)
            max_features = kwargs.pop("max_features", 2000)
            random_state = kwargs.pop("random_state", 42)
            model_path = kwargs.pop("model_path", None)
            use_ngrams = kwargs.pop("use_ngrams", True)
            n_estimators = kwargs.pop("n_estimators", 100)

            # Create config with standardized approach
            config = standardize_classifier_config(
                labels=DEFAULT_GENRES,
                cost=cls.DEFAULT_COST,
                min_confidence=min_confidence,
                params={
                    "max_features": max_features,
                    "random_state": random_state,
                    "model_path": model_path,
                    "use_ngrams": use_ngrams,
                    "n_estimators": n_estimators,
                    "default_genres": DEFAULT_GENRES,
                },
                **kwargs,
            )

        # Create instance with provided configuration
        classifier = cls(name=name, description=description, config=config)

        # Train the classifier and return it
        return classifier.fit(texts, labels)


def create_genre_classifier(
    name: str = "genre_classifier",
    description: str = "Classifies text into different genres",
    labels: Optional[List[str]] = None,
    min_confidence: float = 0.6,
    max_features: int = 2000,
    random_state: int = 42,
    model_path: Optional[str] = None,
    use_ngrams: bool = True,
    n_estimators: int = 100,
    cache_size: int = 100,
    cost: float = GenreClassifier.DEFAULT_COST,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> GenreClassifier:
    """
    Create a genre classifier.

    This factory function creates a GenreClassifier with the specified
    configuration options.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        labels: List of genre labels to use (defaults to DEFAULT_GENRES)
        min_confidence: Minimum confidence threshold
        max_features: Maximum number of features for the vectorizer
        random_state: Random state for reproducibility
        model_path: Path to a pre-trained model file
        use_ngrams: Whether to use n-grams in the vectorizer
        n_estimators: Number of estimators for the RandomForest classifier
        cache_size: Size of the cache for memoization
        cost: Cost of running the classifier
        config: Optional classifier configuration
        **kwargs: Additional configuration parameters

    Returns:
        A GenreClassifier instance

    Examples:
        ```python
        from sifaka.classifiers.genre import create_genre_classifier

        # Create a genre classifier with default settings
        classifier = create_genre_classifier()

        # Create a genre classifier with custom settings
        classifier = create_genre_classifier(
            name="custom_genre_classifier",
            description="Custom genre classifier with specific genres",
            labels=["news", "fiction", "academic", "technical"],
            min_confidence=0.7,
            max_features=3000,
            use_ngrams=True,
            n_estimators=200,
            cache_size=200
        )

        # Classify text
        result = classifier.classify("This is a news article about recent events.")
        print(f"Genre: {result.label}, Confidence: {result.confidence:.2f}")
        ```
    """
    # Use standardize_classifier_config to handle different config formats
    classifier_config = standardize_classifier_config(
        config=config,
        labels=labels or DEFAULT_GENRES,
        min_confidence=min_confidence,
        cost=cost,
        cache_size=cache_size,
        params={
            "max_features": max_features,
            "random_state": random_state,
            "model_path": model_path,
            "use_ngrams": use_ngrams,
            "n_estimators": n_estimators,
            "default_genres": labels or DEFAULT_GENRES,
        },
        **kwargs,
    )

    return GenreClassifier(
        name=name,
        description=description,
        config=classifier_config,
    )
