"""
Genre classifier using scikit-learn's RandomForest.
"""

import importlib
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, model_validator, ConfigDict

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


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

    # Model fields
    vectorizer: Optional[Any] = Field(default=None)
    model: Optional[Any] = Field(default=None)
    pipeline: Optional[Any] = Field(default=None)
    feature_importances: Optional[Dict[str, Dict[str, float]]] = Field(default=None)
    initialized: bool = Field(default=False)
    custom_labels: Optional[List[str]] = Field(default=None)
    sklearn_feature_extraction_text: Optional[Any] = Field(default=None)
    sklearn_ensemble: Optional[Any] = Field(default=None)
    sklearn_pipeline: Optional[Any] = Field(default=None)

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
                labels=self.DEFAULT_GENRES,
                cost=self.DEFAULT_COST,
                min_confidence=0.6,
                params={
                    "min_confidence": 0.6,
                    "max_features": 2000,
                    "random_state": 42,
                    "model_path": None,
                    "use_ngrams": True,
                    "n_estimators": 100,
                    "default_genres": self.DEFAULT_GENRES,
                },
            )

    def _load_dependencies(self) -> None:
        """Load scikit-learn dependencies."""
        try:
            # Import necessary scikit-learn modules
            self.sklearn_feature_extraction_text = importlib.import_module(
                "sklearn.feature_extraction.text"
            )
            self.sklearn_ensemble = importlib.import_module("sklearn.ensemble")
            self.sklearn_pipeline = importlib.import_module("sklearn.pipeline")
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
        if not self.initialized:
            self._load_dependencies()

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
                self.vectorizer = self.sklearn_feature_extraction_text.TfidfVectorizer(
                    max_features=max_features,
                    stop_words="english",
                    ngram_range=ngram_range,
                )

                # Create RandomForest model
                self.model = self.sklearn_ensemble.RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                )

                # Create pipeline
                self.pipeline = self.sklearn_pipeline.Pipeline(
                    [
                        ("vectorizer", self.vectorizer),
                        ("classifier", self.model),
                    ]
                )

            self.initialized = True

    def _save_model(self, path: str) -> None:
        """Save the model to a file."""
        try:
            model_data = {
                "pipeline": self.pipeline,
                "labels": self.custom_labels
                or self.config.params.get("default_genres", self.config.labels),
                "feature_importances": self.feature_importances,
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
                self.pipeline = model_data["pipeline"]
                self.custom_labels = model_data.get("labels")
                self.feature_importances = model_data.get("feature_importances")

                # Update config with loaded labels if available
                if self.custom_labels:
                    self._config = ClassifierConfig(
                        labels=self.custom_labels,
                        cost=self.DEFAULT_COST,
                        min_confidence=self.config.min_confidence,
                        params=self.config.params,
                    )

                # Extract vectorizer and model from pipeline
                self.vectorizer = self.pipeline.named_steps["vectorizer"]
                self.model = self.pipeline.named_steps["classifier"]

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
        self.custom_labels = unique_labels

        # Update config with custom labels
        self._config = ClassifierConfig(
            labels=unique_labels,
            cost=self.DEFAULT_COST,
            min_confidence=self.config.min_confidence,
            params=self.config.params,
        )

        # Fit the pipeline
        self.pipeline.fit(texts, numeric_labels)

        # Extract feature importances
        self.feature_importances = self._extract_feature_importances()

        # Save the model if path is provided
        model_path = self.config.params.get("model_path")
        if model_path:
            self._save_model(model_path)

        return self

    def _extract_feature_importances(self) -> Dict[str, Dict[str, float]]:
        """Extract and return the most important features for each genre."""
        if not self.model or not hasattr(self.model, "feature_importances_"):
            return {}

        try:
            feature_names = self.vectorizer.get_feature_names_out()
            importances = self.model.feature_importances_

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
            return {}

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement genre classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with genre label and confidence
        """
        if not self.pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Predict probability
        proba = self.pipeline.predict_proba([text])[0]

        # Get dominant class
        dominant_class_idx = proba.argmax()
        confidence = float(proba[dominant_class_idx])

        # Create metadata
        all_probs = {self._config.labels[i]: float(prob) for i, prob in enumerate(proba)}

        # Get top features for explanation
        top_features = {}
        if (
            self.feature_importances
            and self._config.labels[dominant_class_idx] in self.feature_importances
        ):
            top_features = self.feature_importances[self._config.labels[dominant_class_idx]]

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
        Classify multiple texts efficiently.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        self.validate_batch_input(texts)

        if not self.pipeline:
            raise RuntimeError(
                "Model not initialized. You must either provide a model_path or call fit() before classification."
            )

        # Predict probabilities for all texts
        probas = self.pipeline.predict_proba(texts)

        results = []
        for proba in probas:
            dominant_class_idx = proba.argmax()
            confidence = float(proba[dominant_class_idx])

            all_probs = {self._config.labels[i]: float(prob) for i, prob in enumerate(proba)}

            # Get top features for explanation
            top_features = {}
            if (
                self.feature_importances
                and self._config.labels[dominant_class_idx] in self.feature_importances
            ):
                top_features = self.feature_importances[self._config.labels[dominant_class_idx]]

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
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> "GenreClassifier":
        """
        Create and train a genre classifier in one step.

        Args:
            texts: List of texts to train on
            labels: List of genre labels
            name: Name of the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Trained GenreClassifier
        """
        # If no config provided, create a default one
        if config is None:
            # Create config with default params
            config = ClassifierConfig(
                labels=cls.DEFAULT_GENRES,
                cost=cls.DEFAULT_COST,
                min_confidence=0.6,
                params={
                    "min_confidence": 0.6,
                    "max_features": 2000,
                    "random_state": 42,
                    "model_path": None,
                    "use_ngrams": True,
                    "n_estimators": 100,
                    "default_genres": cls.DEFAULT_GENRES,
                },
            )

        # Create instance with provided configuration
        classifier = cls(name=name, description=description, config=config, **kwargs)

        # Train the classifier and return it
        return classifier.fit(texts, labels)
