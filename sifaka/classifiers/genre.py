"""
Genre classifier using scikit-learn's RandomForest.

This module provides a classifier for categorizing text into different genres
such as news, fiction, academic, technical, etc. using textual features.
"""

import importlib
import os
import pickle
from typing import Dict, List, Optional, Any, ClassVar, Union

from pydantic import ConfigDict, PrivateAttr

from sifaka.classifiers.base import (
    Classifier,
    ClassificationResult,
    ClassifierConfig,
    ClassifierImplementation,
)
from sifaka.classifiers.config import standardize_classifier_config
from sifaka.utils.logging import get_logger
from sifaka.utils.state import ClassifierState, create_classifier_state

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


class GenreClassifierImplementation:
    """
    Implementation of genre classification logic using scikit-learn's RandomForest.

    This implementation uses RandomForest from scikit-learn to categorize text into
    different genres such as news, fiction, academic, technical, etc. using textual
    features.

    ## Architecture

    GenreClassifierImplementation follows the composition pattern:
    1. **Core Logic**: classify_impl() implements genre classification
    2. **State Management**: Uses ClassifierState for internal state
    3. **Resource Management**: Loads and manages scikit-learn models

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Initialize with ClassifierConfig
       - Set up internal state

    2. **Warm-up**: Prepare resources with warm_up_impl()
       - Load scikit-learn dependencies
       - Initialize vectorizer and classifier
       - Load pre-trained model if available

    3. **Classification**: Process text with classify_impl()
       - Vectorize input text
       - Apply RandomForest model
       - Return genre prediction with confidence

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    # Class constants
    DEFAULT_COST: ClassVar[float] = 2.0

    def __init__(self, config: ClassifierConfig):
        """
        Initialize the genre classifier implementation.

        Args:
            config: Configuration for the classifier
        """
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False
        self._state.cache = {}

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

    def warm_up_impl(self) -> None:
        """
        Prepare resources for classification by initializing the RandomForest model.

        This method prepares the classifier for use by:
        1. Loading required dependencies
        2. Initializing the vectorizer and classifier
        3. Loading a pre-trained model if available

        Raises:
            RuntimeError: If model initialization fails
        """
        # Check if already initialized
        if not self._state.initialized:
            # Load dependencies
            sklearn = self._load_dependencies()
            self._state.dependencies_loaded = True

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
                self._state.vectorizer = sklearn["feature_extraction_text"].TfidfVectorizer(
                    max_features=max_features,
                    stop_words="english",
                    ngram_range=ngram_range,
                )

                # Create RandomForest model
                self._state.model = sklearn["ensemble"].RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                )

                # Create pipeline
                self._state.pipeline = sklearn["pipeline"].Pipeline(
                    [
                        ("vectorizer", self._state.vectorizer),
                        ("classifier", self._state.model),
                    ]
                )

            # Mark as initialized
            self._state.initialized = True

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
        try:
            # Get custom labels from state cache
            custom_labels = self._state.cache.get("custom_labels") or self.config.params.get(
                "default_genres", self.config.labels
            )

            model_data = {
                "pipeline": self._state.pipeline,
                "labels": custom_labels,
                "feature_importances": self._state.cache.get("feature_importances", {}),
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            self._state.error = f"Failed to save model: {e}"

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
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

                # Extract model data
                self._state.pipeline = model_data["pipeline"]
                custom_labels = model_data.get("labels")
                self._state.cache["feature_importances"] = model_data.get("feature_importances", {})
                self._state.cache["custom_labels"] = custom_labels

                # Update config with loaded labels if available
                if custom_labels:
                    self.config = ClassifierConfig(
                        labels=custom_labels,
                        cost=self.DEFAULT_COST,
                        min_confidence=self.config.min_confidence,
                        params=self.config.params,
                    )

                # Extract vectorizer and model from pipeline
                self._state.vectorizer = self._state.pipeline.named_steps["vectorizer"]
                self._state.model = self._state.pipeline.named_steps["classifier"]

            logger.info(f"Model loaded from {path}")
            self._state.initialized = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._state.error = f"Failed to load model: {e}"
            raise RuntimeError(f"Failed to load model: {e}")

    def fit_impl(self, texts: List[str], labels: List[str]) -> None:
        """
        Fit the genre classifier on a corpus of texts.

        Args:
            texts: List of texts to fit the model on
            labels: List of genre labels
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

        self.warm_up_impl()

        # Get unique labels
        unique_labels = sorted(set(labels))

        # Create label mapping
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_mapping[label] for label in labels]

        # Store custom labels in state cache
        self._state.cache["custom_labels"] = unique_labels

        # Update config with custom labels
        self.config = ClassifierConfig(
            labels=unique_labels,
            cost=self.DEFAULT_COST,
            min_confidence=self.config.min_confidence,
            params=self.config.params,
        )

        # Fit the pipeline
        self._state.pipeline.fit(texts, numeric_labels)

        # Extract feature importances
        self._state.cache["feature_importances"] = self._extract_feature_importances()

        # Save the model if path is provided
        model_path = self.config.params.get("model_path")
        if model_path:
            self._save_model(model_path)

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
        if not self._state.model or not hasattr(self._state.model, "feature_importances_"):
            return {}

        try:
            feature_names = self._state.vectorizer.get_feature_names_out()
            importances = self._state.model.feature_importances_

            # Get the top features
            top_features = {}
            for label in self.config.labels:
                # For RandomForest, we need to extract class-specific importances differently
                # Here we use the overall feature importance as a simplification
                top_indices = importances.argsort()[-20:][::-1]
                top_features[label] = {feature_names[j]: float(importances[j]) for j in top_indices}

            return top_features
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
            self._state.error = f"Could not extract feature importances: {e}"
            return {}

    def classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement genre classification logic.

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
        if not self._state.pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Predict probability
        proba = self._state.pipeline.predict_proba([text])[0]

        # Get dominant class
        dominant_class_idx = proba.argmax()
        confidence = float(proba[dominant_class_idx])

        # Create metadata
        all_probs = {self.config.labels[i]: float(prob) for i, prob in enumerate(proba)}

        # Get top features for explanation
        top_features = {}
        feature_importances = self._state.cache.get("feature_importances", {})
        if feature_importances and self.config.labels[dominant_class_idx] in feature_importances:
            top_features = feature_importances[self.config.labels[dominant_class_idx]]

        # Get min_confidence from config
        min_confidence = self.config.params.get("min_confidence", 0.6)

        metadata = {
            "probabilities": all_probs,
            "threshold": min_confidence,
            "is_confident": confidence >= min_confidence,
            "top_features": top_features,
        }

        return ClassificationResult(
            label=self.config.labels[dominant_class_idx],
            confidence=confidence,
            metadata=metadata,
        )

    def batch_classify_impl(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Implement batch classification logic for multiple texts.

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
        if not self._state.pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Predict probabilities for all texts
        probas = self._state.pipeline.predict_proba(texts)

        results = []
        feature_importances = self._state.cache.get("feature_importances", {})

        for proba in probas:
            dominant_class_idx = proba.argmax()
            confidence = float(proba[dominant_class_idx])

            all_probs = {self.config.labels[i]: float(prob) for i, prob in enumerate(proba)}

            # Get top features for explanation
            top_features = {}
            if (
                feature_importances
                and self.config.labels[dominant_class_idx] in feature_importances
            ):
                top_features = feature_importances[self.config.labels[dominant_class_idx]]

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
                    label=self.config.labels[dominant_class_idx],
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return results


def create_pretrained_genre_classifier(
    texts: List[str],
    labels: List[str],
    name: str = "pretrained_genre_classifier",
    description: str = "Pre-trained genre classifier",
    min_confidence: float = 0.6,
    max_features: int = 2000,
    random_state: int = 42,
    model_path: Optional[str] = None,
    use_ngrams: bool = True,
    n_estimators: int = 100,
    cache_size: int = 100,
    cost: float = GenreClassifierImplementation.DEFAULT_COST,
    config: Optional[ClassifierConfig] = None,
    **kwargs,
) -> Classifier[str, str]:
    """
    Create a pre-trained genre classifier from labeled data.

    This factory function creates and trains a genre classifier with the provided
    training data. It follows the composition over inheritance pattern,
    creating a Classifier with a GenreClassifierImplementation.

    Args:
        texts: List of training texts
        labels: List of corresponding genre labels
        name: Name for the classifier
        description: Description of the classifier
        min_confidence: Minimum confidence threshold
        max_features: Maximum number of features for the vectorizer
        random_state: Random state for reproducibility
        model_path: Path to save the trained model
        use_ngrams: Whether to use n-grams in the vectorizer
        n_estimators: Number of estimators for the RandomForest classifier
        cache_size: Size of the classification cache
        cost: Computational cost of this classifier
        config: Optional classifier configuration
        **kwargs: Additional configuration parameters

    Returns:
        A trained Classifier instance with GenreClassifierImplementation

    Raises:
        ValueError: If training data is invalid or inconsistent
        RuntimeError: If model training fails
    """
    # If no config provided, create a default one
    if config is None:
        # Create config with standardized approach
        config = standardize_classifier_config(
            labels=DEFAULT_GENRES,
            cost=cost,
            min_confidence=min_confidence,
            cache_size=cache_size,
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

    # Create implementation
    implementation = GenreClassifierImplementation(config)

    # Create classifier
    classifier = Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )

    # Train the classifier
    implementation.fit_impl(texts, labels)

    return classifier


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
    cost: float = GenreClassifierImplementation.DEFAULT_COST,
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Create a genre classifier.

    This factory function creates a genre classifier with the specified
    configuration options. It follows the composition over inheritance pattern,
    creating a Classifier with a GenreClassifierImplementation.

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
        A Classifier instance with GenreClassifierImplementation

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

    # Create implementation
    implementation = GenreClassifierImplementation(classifier_config)

    # Initialize the implementation if model path is provided
    if model_path and os.path.exists(model_path):
        implementation.warm_up_impl()

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=classifier_config,
        implementation=implementation,
    )
