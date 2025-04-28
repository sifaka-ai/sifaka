"""
Genre classifier using scikit-learn's RandomForest.
"""

import importlib
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pydantic import Field

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass(frozen=True)
class GenreConfig:
    """Configuration for genre classification."""

    min_confidence: float = 0.6  # Minimum confidence threshold
    max_features: int = 2000  # Max features for vectorization
    random_state: int = 42  # For reproducibility
    model_path: Optional[str] = None  # Path to save/load the model
    use_ngrams: bool = True  # Whether to use n-grams
    n_estimators: int = 100  # Number of trees in the forest

    # Default genre labels
    default_genres: List[str] = field(
        default_factory=lambda: [
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
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not self.default_genres:
            raise ValueError("default_genres cannot be empty")

class GenreClassifier(BaseClassifier):
    """
    A genre classifier using RandomForest from scikit-learn.

    This classifier categorizes text into different genres such as news,
    fiction, academic, technical, etc. using textual features.

    Requires scikit-learn to be installed:
    pip install scikit-learn
    """

    # Class constants
    DEFAULT_COST: float = 2.0

    # Additional Pydantic fields
    genre_config: GenreConfig = Field(
        default_factory=lambda: GenreConfig(),
        description="Genre classification configuration",
    )

    def __init__(
        self,
        name: str = "genre_classifier",
        description: str = "Classifies text into different genres",
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the genre classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            config: Classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Extract genre-specific config from kwargs or use defaults
        genre_kwargs = {}
        for key in ["max_features", "use_ngrams", "n_estimators", "model_path"]:
            if key in kwargs.get("additional_config", {}):
                genre_kwargs[key] = kwargs["additional_config"].pop(key)

        # Create genre config
        genre_config = GenreConfig(
            min_confidence=kwargs.get("min_confidence", 0.6),
            **genre_kwargs,
        )

        # If config is not provided, create one from genre_config values
        if config is None:
            config = ClassifierConfig(
                labels=genre_config.default_genres,
                cost=self.DEFAULT_COST,
                min_confidence=genre_config.min_confidence,
                **kwargs,
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

        # Set genre config
        self.genre_config = genre_config

        # Initialize other attributes
        self._vectorizer = None
        self._model = None
        self._pipeline = None
        self._feature_importances = None
        self._initialized = False
        self._custom_labels = None

    def _load_dependencies(self) -> None:
        """Load scikit-learn dependencies."""
        try:
            # Import necessary scikit-learn modules
            self._sklearn_feature_extraction_text = importlib.import_module(
                "sklearn.feature_extraction.text"
            )
            self._sklearn_ensemble = importlib.import_module("sklearn.ensemble")
            self._sklearn_pipeline = importlib.import_module("sklearn.pipeline")
            return True
        except ImportError:
            raise ImportError(
                "scikit-learn is required for GenreClassifier. "
                "Install it with: pip install scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn modules: {e}")

    def warm_up(self) -> None:
        """Initialize the model if needed."""
        if not self._initialized:
            self._load_dependencies()

            if self.genre_config.model_path and os.path.exists(self.genre_config.model_path):
                self._load_model(self.genre_config.model_path)
            else:
                # Create TF-IDF vectorizer
                ngram_range = (1, 3) if self.genre_config.use_ngrams else (1, 1)
                self._vectorizer = self._sklearn_feature_extraction_text.TfidfVectorizer(
                    max_features=self.genre_config.max_features,
                    stop_words="english",
                    ngram_range=ngram_range,
                )

                # Create RandomForest model
                self._model = self._sklearn_ensemble.RandomForestClassifier(
                    n_estimators=self.genre_config.n_estimators,
                    random_state=self.genre_config.random_state,
                )

                # Create pipeline
                self._pipeline = self._sklearn_pipeline.Pipeline(
                    [
                        ("vectorizer", self._vectorizer),
                        ("classifier", self._model),
                    ]
                )

            self._initialized = True

    def _save_model(self, path: str) -> None:
        """Save the model to a file."""
        try:
            model_data = {
                "pipeline": self._pipeline,
                "labels": self._custom_labels or self.genre_config.default_genres,
                "feature_importances": self._feature_importances,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model(self, path: str) -> None:
        """Load the model from a file."""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

                # Extract model data
                self._pipeline = model_data["pipeline"]
                self._custom_labels = model_data.get("labels")
                self._feature_importances = model_data.get("feature_importances")

                # Update config with loaded labels if available
                if self._custom_labels:
                    self._config = ClassifierConfig(
                        labels=self._custom_labels,
                        cost=self.DEFAULT_COST,
                    )

                # Extract vectorizer and model from pipeline
                self._vectorizer = self._pipeline.named_steps["vectorizer"]
                self._model = self._pipeline.named_steps["classifier"]

            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

        self.warm_up()

        # Get unique labels
        unique_labels = sorted(set(labels))

        # Create label mapping
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_mapping[label] for label in labels]

        # Store custom labels
        self._custom_labels = unique_labels

        # Update config with custom labels
        self._config = ClassifierConfig(
            labels=unique_labels,
            cost=self.DEFAULT_COST,
        )

        # Fit the pipeline
        self._pipeline.fit(texts, numeric_labels)

        # Extract feature importances
        self._feature_importances = self._extract_feature_importances()

        # Save the model if path is provided
        if self.genre_config.model_path:
            self._save_model(self.genre_config.model_path)

        return self

    def _extract_feature_importances(self) -> Dict[str, float]:
        """Extract and return the most important features for each genre."""
        if not self._model or not hasattr(self._model, "feature_importances_"):
            return {}

        try:
            feature_names = self._vectorizer.get_feature_names_out()
            importances = self._model.feature_importances_

            # Get the top features
            top_features = {}
            for i, label in enumerate(self._config.labels):
                # For RandomForest, we need to extract class-specific importances differently
                # Here we use the overall feature importance as a simplification
                top_indices = importances.argsort()[-20:][::-1]
                top_features[label] = {feature_names[j]: float(importances[j]) for j in top_indices}

            return top_features
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
            return {}

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement genre classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with genre label and confidence
        """
        if not self._pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Predict probability
        proba = self._pipeline.predict_proba([text])[0]

        # Get dominant class
        dominant_class_idx = proba.argmax()
        confidence = float(proba[dominant_class_idx])

        # Create metadata
        all_probs = {self._config.labels[i]: float(prob) for i, prob in enumerate(proba)}

        # Get top features for explanation
        top_features = {}
        if (
            self._feature_importances
            and self._config.labels[dominant_class_idx] in self._feature_importances
        ):
            top_features = self._feature_importances[self._config.labels[dominant_class_idx]]

        metadata = {
            "probabilities": all_probs,
            "threshold": self.genre_config.min_confidence,
            "is_confident": confidence >= self.genre_config.min_confidence,
            "top_features": top_features,
        }

        return ClassificationResult(
            label=self._config.labels[dominant_class_idx],
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

        if not self._pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Predict probabilities for all texts
        probas = self._pipeline.predict_proba(texts)

        results = []
        for i, proba in enumerate(probas):
            dominant_class_idx = proba.argmax()
            confidence = float(proba[dominant_class_idx])

            all_probs = {self._config.labels[i]: float(prob) for i, prob in enumerate(proba)}

            # Get top features for explanation
            top_features = {}
            if (
                self._feature_importances
                and self._config.labels[dominant_class_idx] in self._feature_importances
            ):
                top_features = self._feature_importances[self._config.labels[dominant_class_idx]]

            metadata = {
                "probabilities": all_probs,
                "threshold": self.genre_config.min_confidence,
                "is_confident": confidence >= self.genre_config.min_confidence,
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
        genre_config: Optional[GenreConfig] = None,
        **kwargs,
    ) -> "GenreClassifier":
        """
        Create and train a genre classifier in one step.

        Args:
            texts: List of texts to train on
            labels: List of genre labels
            name: Name of the classifier
            description: Description of the classifier
            genre_config: Genre classification configuration
            **kwargs: Additional configuration parameters

        Returns:
            Trained GenreClassifier
        """
        classifier = cls(name=name, description=description, genre_config=genre_config, **kwargs)
        return classifier.fit(texts, labels)
