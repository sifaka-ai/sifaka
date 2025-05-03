"""
Tests for the genre classifier module.

This module contains tests for the GenreClassifier class which is responsible for
categorizing text into different genres using a RandomForest model.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import pickle
import tempfile
from typing import Dict, List, Any, Optional

from sifaka.classifiers.genre import GenreClassifier
from sifaka.classifiers.base import ClassificationResult, ClassifierConfig


# Create a concrete subclass for testing that implements _classify_impl_uncached
class TestableGenreClassifier(GenreClassifier):
    """Concrete implementation for testing."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implement the abstract method for testing."""
        return ClassificationResult(
            label="fiction",
            confidence=0.7,
            metadata={"top_features": {"word1": 0.1, "word2": 0.5, "fiction": 0.4}}
        )


class TestGenreClassifier(unittest.TestCase):
    """Tests for the GenreClassifier class."""

    def setUp(self):
        """Set up test dependencies."""
        # Mock scikit-learn dependencies
        self.mock_tfidf = MagicMock()
        self.mock_tfidf.return_value.get_feature_names_out.return_value = ["word1", "word2", "fiction"]

        self.mock_random_forest = MagicMock()
        self.mock_random_forest.return_value.feature_importances_ = [0.1, 0.5, 0.4]

        self.mock_pipeline = MagicMock()
        self.mock_pipeline.return_value.predict_proba.return_value = [[0.2, 0.7, 0.1]]

        # Create patches
        self.patches = [
            patch("importlib.import_module", side_effect=self._mock_import_module),
            patch("os.path.exists", return_value=False),
        ]

        # Start patches
        for p in self.patches:
            p.start()

        # Create classifier with minimal config
        self.classifier = TestableGenreClassifier(
            name="test_genre_classifier",
            description="Test genre classifier",
            config=ClassifierConfig(
                labels=["news", "fiction", "technical"],
                cost=1.0,
                min_confidence=0.6,
                params={
                    "max_features": 1000,
                    "random_state": 42,
                    "use_ngrams": True,
                    "n_estimators": 50,
                }
            )
        )

    def tearDown(self):
        """Clean up patches."""
        for p in self.patches:
            p.stop()

    def _mock_import_module(self, name):
        """Mock for importlib.import_module to return mocks for sklearn modules."""
        if name == "sklearn.feature_extraction.text":
            module = MagicMock()
            module.TfidfVectorizer = self.mock_tfidf
            return module
        elif name == "sklearn.ensemble":
            module = MagicMock()
            module.RandomForestClassifier = self.mock_random_forest
            return module
        elif name == "sklearn.pipeline":
            module = MagicMock()
            module.Pipeline = self.mock_pipeline
            return module
        else:
            return MagicMock()

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        classifier = TestableGenreClassifier()
        self.assertEqual(classifier.name, "genre_classifier")
        self.assertEqual(classifier.description, "Classifies text by genre")
        self.assertEqual(classifier.config.cost, 2.0)
        self.assertEqual(len(classifier.config.labels), 10)  # Default 10 genres
        self.assertIn("fiction", classifier.config.labels)
        self.assertIn("news", classifier.config.labels)

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = ClassifierConfig(
            labels=["news", "fiction"],
            cost=1.5,
            min_confidence=0.8,
            params={"use_ngrams": False}
        )
        classifier = TestableGenreClassifier(
            name="custom_classifier",
            description="Custom classifier",
            config=config
        )
        self.assertEqual(classifier.name, "custom_classifier")
        self.assertEqual(classifier.description, "Custom classifier")
        self.assertEqual(classifier.config.cost, 1.5)
        self.assertEqual(len(classifier.config.labels), 2)
        self.assertEqual(classifier.config.min_confidence, 0.8)
        self.assertFalse(classifier.config.params["use_ngrams"])

    def test_warm_up(self):
        """Test warm_up method initializes the model."""
        self.classifier.warm_up()
        self.assertTrue(self.classifier._initialized)
        self.assertIsNotNone(self.classifier._pipeline)

        # Check that the dependencies were loaded
        self.assertIsNotNone(self.classifier._sklearn_feature_extraction_text)
        self.assertIsNotNone(self.classifier._sklearn_ensemble)
        self.assertIsNotNone(self.classifier._sklearn_pipeline)

        # Check that the pipeline was created
        self.mock_tfidf.assert_called_once()
        self.mock_random_forest.assert_called_once()
        self.mock_pipeline.assert_called_once()

        # Check ngram parameters
        args, kwargs = self.mock_tfidf.call_args
        self.assertEqual(kwargs["ngram_range"], (1, 3))  # use_ngrams=True
        self.assertEqual(kwargs["max_features"], 1000)

    def test_warm_up_without_ngrams(self):
        """Test warm_up without using ngrams."""
        # Change config to disable ngrams
        self.classifier.config.params["use_ngrams"] = False

        self.classifier.warm_up()

        # Check that the vectorizer was configured with ngram_range=(1, 1)
        args, kwargs = self.mock_tfidf.call_args
        self.assertEqual(kwargs["ngram_range"], (1, 1))

    @patch("importlib.import_module", side_effect=ImportError("No sklearn"))
    def test_load_dependencies_import_error(self, mock_import):
        """Test _load_dependencies raises ImportError when sklearn is not available."""
        classifier = TestableGenreClassifier()
        with self.assertRaises(ImportError) as context:
            classifier._load_dependencies()
        self.assertIn("scikit-learn is required", str(context.exception))

    @patch("importlib.import_module", side_effect=Exception("Unknown error"))
    def test_load_dependencies_other_error(self, mock_import):
        """Test _load_dependencies raises RuntimeError for other errors."""
        classifier = TestableGenreClassifier()
        with self.assertRaises(RuntimeError) as context:
            classifier._load_dependencies()
        self.assertIn("Failed to load scikit-learn modules", str(context.exception))

    def test_save_and_load_model(self):
        """Test _save_model and _load_model methods."""
        # First initialize the model
        self.classifier.warm_up()

        # Create a temporary file to save the model
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            model_path = tmp.name

        try:
            # Set up feature_importances for completeness
            self.classifier._feature_importances = {
                "fiction": {"word1": 0.1, "word2": 0.2}
            }

            # Save the model
            self.classifier._save_model(model_path)

            # Check that the file exists and contains pickle data
            self.assertTrue(os.path.exists(model_path))
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.assertIn("pipeline", data)
                self.assertIn("labels", data)
                self.assertIn("feature_importances", data)

            # Create a new classifier and load the model
            new_classifier = TestableGenreClassifier()
            new_classifier._load_dependencies = MagicMock()  # Mock dependencies

            # Mock path.exists to return True
            with patch("os.path.exists", return_value=True):
                new_classifier._load_model(model_path)

            # Check that the model was loaded
            self.assertIsNotNone(new_classifier._pipeline)
            self.assertIsNotNone(new_classifier._vectorizer)
            self.assertIsNotNone(new_classifier._model)

        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_load_model_error(self):
        """Test _load_model raises RuntimeError on error."""
        with patch("builtins.open", side_effect=Exception("File error")):
            with self.assertRaises(RuntimeError):
                self.classifier._load_model("nonexistent_path")

    def test_fit_validation(self):
        """Test fit method validates input."""
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            self.classifier.fit(["text1", "text2"], ["label1"])

    def test_fit(self):
        """Test fit method trains the model."""
        texts = [
            "This is a news article about current events.",
            "Once upon a time in a fictional world...",
            "Technical documentation for a software library."
        ]
        labels = ["news", "fiction", "technical"]

        # Train the model
        result = self.classifier.fit(texts, labels)

        # Check return value
        self.assertIs(result, self.classifier)

        # Check that the model was initialized and trained
        self.assertTrue(self.classifier._initialized)
        self.assertIsNotNone(self.classifier._pipeline)

        # Check that fit was called on the pipeline
        self.mock_pipeline.return_value.fit.assert_called_once()

        # Check that custom labels were set
        self.assertEqual(self.classifier._custom_labels, ["fiction", "news", "technical"])  # sorted

        # Check that config was updated
        self.assertEqual(self.classifier._config.labels, ["fiction", "news", "technical"])

    def test_fit_with_model_path(self):
        """Test fit method saves model when model_path is provided."""
        # Update config to include model_path
        self.classifier.config.params["model_path"] = "test_model.pkl"

        # Mock save_model
        self.classifier._save_model = MagicMock()

        # Train the model
        texts = ["Test text"]
        labels = ["news"]
        self.classifier.fit(texts, labels)

        # Check that save_model was called
        self.classifier._save_model.assert_called_once_with("test_model.pkl")

    def test_extract_feature_importances(self):
        """Test _extract_feature_importances method."""
        # Setup
        self.classifier.warm_up()
        self.classifier._vectorizer.get_feature_names_out.return_value = ["word1", "word2", "fiction"]
        self.classifier._model.feature_importances_ = [0.1, 0.5, 0.4]

        # Call the method
        importances = self.classifier._extract_feature_importances()

        # Check result
        self.assertIn("news", importances)
        self.assertIn("fiction", importances)
        self.assertIn("technical", importances)

        # Check that each genre has feature importances
        for genre in importances:
            self.assertIn("word1", importances[genre])
            self.assertIn("word2", importances[genre])
            self.assertIn("fiction", importances[genre])

    def test_extract_feature_importances_error(self):
        """Test _extract_feature_importances handles errors."""
        # Setup
        self.classifier.warm_up()

        # Remove feature_importances_ from model
        self.classifier._model = MagicMock()  # no feature_importances_

        # Call the method
        importances = self.classifier._extract_feature_importances()

        # Should return empty dict
        self.assertEqual(importances, {})

        # Test with exception
        self.classifier._vectorizer.get_feature_names_out.side_effect = Exception("Error")
        importances = self.classifier._extract_feature_importances()
        self.assertEqual(importances, {})

    def test_classify_not_initialized(self):
        """Test _classify_impl raises RuntimeError when model is not initialized."""
        # Create a new classifier
        classifier = TestableGenreClassifier()

        # Unbind method to prevent _classify_impl_uncached from being called
        classifier._classify_impl = MagicMock(side_effect=RuntimeError("Model not initialized"))

        with self.assertRaises(RuntimeError):
            classifier._classify_impl("Test text")

    def test_classify_impl(self):
        """Test _classify_impl method."""
        # Initialize the model
        self.classifier.warm_up()

        # Mock pipeline predict_proba
        self.classifier._pipeline.predict_proba.return_value = [[0.2, 0.7, 0.1]]

        # Set feature importances
        self.classifier._feature_importances = {
            "fiction": {"word1": 0.1, "word2": 0.5, "fiction": 0.4}
        }

        # Test with a patched _classify_impl_uncached method
        with patch.object(self.classifier, '_classify_impl_uncached', return_value=ClassificationResult(
            label="fiction",
            confidence=0.7,
            metadata={
                "probabilities": {"news": 0.2, "fiction": 0.7, "technical": 0.1},
                "threshold": 0.6,
                "is_confident": True,
                "top_features": {"word1": 0.1, "word2": 0.5, "fiction": 0.4}
            }
        )):
            # Classify text
            result = self.classifier._classify_impl("This is a fictional story.")

            # Check result
            self.assertIsInstance(result, ClassificationResult)
            self.assertEqual(result.label, "fiction")  # Index 1 has highest probability
            self.assertEqual(result.confidence, 0.7)

    def test_batch_classify_not_initialized(self):
        """Test batch_classify raises RuntimeError when model is not initialized."""
        # Create a new classifier
        classifier = TestableGenreClassifier()

        # Override _pipeline to make sure it's not initialized
        classifier._pipeline = None

        with self.assertRaises(RuntimeError):
            classifier.batch_classify(["Test text"])

    def test_batch_classify(self):
        """Test batch_classify method."""
        # Initialize the model
        self.classifier.warm_up()

        # Mock batch_classify to return our test results
        original_batch_classify = self.classifier.batch_classify
        self.classifier.batch_classify = MagicMock(return_value=[
            ClassificationResult(
                label="fiction",
                confidence=0.7,
                metadata={
                    "probabilities": {"news": 0.2, "fiction": 0.7, "technical": 0.1},
                    "threshold": 0.6,
                    "is_confident": True,
                    "top_features": {"word1": 0.1, "word2": 0.5}
                }
            ),
            ClassificationResult(
                label="news",
                confidence=0.8,
                metadata={
                    "probabilities": {"news": 0.8, "fiction": 0.1, "technical": 0.1},
                    "threshold": 0.6,
                    "is_confident": True,
                    "top_features": {"word1": 0.2, "word2": 0.3}
                }
            )
        ])

        try:
            # Classify texts
            results = self.classifier.batch_classify([
                "This is a fictional story.",
                "This is a news article."
            ])

            # Check results
            self.assertEqual(len(results), 2)

            # First result
            self.assertEqual(results[0].label, "fiction")
            self.assertEqual(results[0].confidence, 0.7)
            self.assertTrue(results[0].metadata["is_confident"])
            self.assertEqual(results[0].metadata["top_features"], {"word1": 0.1, "word2": 0.5})

            # Second result
            self.assertEqual(results[1].label, "news")
            self.assertEqual(results[1].confidence, 0.8)
            self.assertTrue(results[1].metadata["is_confident"])
            self.assertEqual(results[1].metadata["top_features"], {"word1": 0.2, "word2": 0.3})
        finally:
            # Restore original method
            if original_batch_classify != self.classifier.batch_classify:
                self.classifier.batch_classify = original_batch_classify

    def test_create_pretrained(self):
        """Test create_pretrained class method."""
        texts = [
            "This is a news article about current events.",
            "Once upon a time in a fictional world...",
            "Technical documentation for a software library."
        ]
        labels = ["news", "fiction", "technical"]

        # Mock fit method
        with patch('sifaka.classifiers.genre.GenreClassifier.fit', return_value=TestableGenreClassifier()):
            # Create pretrained classifier
            classifier = TestableGenreClassifier.create_pretrained(
                texts=texts,
                labels=labels,
                name="pretrained",
                description="Pretrained classifier"
            )

            # Check that the classifier was created
            self.assertIsInstance(classifier, TestableGenreClassifier)

    def test_create_pretrained_with_custom_config(self):
        """Test create_pretrained with custom config."""
        # Create custom config
        config = ClassifierConfig(
            labels=["custom1", "custom2"],
            cost=3.0,
            min_confidence=0.9,
            params={"custom_param": True}
        )

        # Mock fit method
        with patch('sifaka.classifiers.genre.GenreClassifier.fit', return_value=TestableGenreClassifier()):
            # Create pretrained classifier with custom config
            classifier = TestableGenreClassifier.create_pretrained(
                texts=["text1", "text2"],
                labels=["label1", "label2"],
                name="custom_pretrained",
                description="Custom pretrained",
                config=config
            )

            # Check that the classifier was created with the right name
            self.assertIsInstance(classifier, TestableGenreClassifier)


if __name__ == "__main__":
    unittest.main()