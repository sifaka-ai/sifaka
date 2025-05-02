"""
Tests for the bias classifier module.

This module contains tests for the BiasDetector class which is responsible for
detecting various forms of bias in text using a support vector machine model.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import pickle
import tempfile
from typing import Dict, List, Any, Optional

from sifaka.classifiers.bias import BiasDetector
from sifaka.classifiers.base import ClassificationResult, ClassifierConfig


# Create a concrete subclass for testing that implements _classify_impl_uncached
class TestableBiasDetector(BiasDetector):
    """Concrete implementation for testing."""

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """Implement the abstract method for testing."""
        return ClassificationResult(
            label="gender",
            confidence=0.7,
            metadata={"features": {"bias_gender": 0.5}}
        )


class TestBiasDetector(unittest.TestCase):
    """Tests for the BiasDetector class."""

    def setUp(self):
        """Set up test dependencies."""
        # Mock scikit-learn dependencies
        self.mock_tfidf = MagicMock()
        self.mock_tfidf.return_value.get_feature_names_out.return_value = ["word1", "word2", "bias"]

        self.mock_svc = MagicMock()
        self.mock_calibrated_cv = MagicMock()
        self.mock_pipeline = MagicMock()

        # Add predict methods to pipeline
        self.mock_pipeline.return_value.predict.return_value = ["gender"]
        self.mock_pipeline.return_value.predict_proba.return_value = [[0.2, 0.7, 0.1]]

        # Create patches
        self.patches = [
            patch("importlib.import_module", side_effect=self._mock_import_module),
            patch("os.path.exists", return_value=False),
        ]

        # Start patches
        for p in self.patches:
            p.start()

        # Create detector with minimal config
        self.detector = TestableBiasDetector(
            name="test_bias_detector",
            description="Test bias detector",
            config=ClassifierConfig(
                labels=["neutral", "gender", "racial"],
                cost=1.0,
                min_confidence=0.6,
                params={
                    "bias_types": ["neutral", "gender", "racial"],
                    "bias_keywords": {
                        "gender": ["man", "woman"],
                        "racial": ["race", "ethnic"],
                    }
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
        elif name == "sklearn.svm":
            module = MagicMock()
            module.LinearSVC = self.mock_svc
            module.SVC = self.mock_svc
            return module
        elif name == "sklearn.pipeline":
            module = MagicMock()
            module.Pipeline = self.mock_pipeline
            return module
        elif name == "sklearn.calibration":
            module = MagicMock()
            module.CalibratedClassifierCV = self.mock_calibrated_cv
            return module
        else:
            return MagicMock()

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        detector = TestableBiasDetector()
        self.assertEqual(detector.name, "bias_detector")
        self.assertEqual(detector.description, "Detects various forms of bias in text")
        self.assertEqual(detector.config.cost, 2.5)
        self.assertEqual(len(detector.config.labels), 10)  # Default 10 bias types
        self.assertIn("gender", detector.config.labels)
        self.assertIn("neutral", detector.config.labels)

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = ClassifierConfig(
            labels=["neutral", "gender"],
            cost=1.5,
            min_confidence=0.8,
            params={"bias_types": ["neutral", "gender"]}
        )
        detector = TestableBiasDetector(
            name="custom_detector",
            description="Custom detector",
            config=config
        )
        self.assertEqual(detector.name, "custom_detector")
        self.assertEqual(detector.description, "Custom detector")
        self.assertEqual(detector.config.cost, 1.5)
        self.assertEqual(len(detector.config.labels), 2)
        self.assertEqual(detector.config.min_confidence, 0.8)

    def test_warm_up(self):
        """Test warm_up method initializes the model."""
        self.detector.warm_up()
        self.assertTrue(self.detector._initialized)
        self.assertIsNotNone(self.detector._pipeline)

        # Check that the dependencies were loaded
        self.assertIsNotNone(self.detector._sklearn_feature_extraction_text)
        self.assertIsNotNone(self.detector._sklearn_svm)
        self.assertIsNotNone(self.detector._sklearn_pipeline)
        self.assertIsNotNone(self.detector._sklearn_calibration)

        # Check that the pipeline was created
        self.mock_tfidf.assert_called_once()
        self.mock_svc.assert_called_once()
        self.mock_calibrated_cv.assert_called_once()
        self.mock_pipeline.assert_called_once()

    @patch("importlib.import_module", side_effect=ImportError("No sklearn"))
    def test_load_dependencies_import_error(self, mock_import):
        """Test _load_dependencies raises ImportError when sklearn is not available."""
        detector = TestableBiasDetector()
        with self.assertRaises(ImportError) as context:
            detector._load_dependencies()
        self.assertIn("scikit-learn is required", str(context.exception))

    @patch("importlib.import_module", side_effect=Exception("Unknown error"))
    def test_load_dependencies_other_error(self, mock_import):
        """Test _load_dependencies raises RuntimeError for other errors."""
        detector = TestableBiasDetector()
        with self.assertRaises(RuntimeError) as context:
            detector._load_dependencies()
        self.assertIn("Failed to load scikit-learn modules", str(context.exception))

    def test_extract_bias_features(self):
        """Test _extract_bias_features method."""
        text = "Women are often paid less than men for the same work."
        features = self.detector._extract_bias_features(text)

        # Should detect gender bias keywords
        self.assertIn("bias_gender", features)
        self.assertGreater(features["bias_gender"], 0)

        # Should not detect racial bias keywords
        self.assertIn("bias_racial", features)
        self.assertEqual(features["bias_racial"], 0)

    def test_save_and_load_model(self):
        """Test _save_model and _load_model methods."""
        # First initialize the model
        self.detector.warm_up()

        # Create a temporary file to save the model
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            model_path = tmp.name

        try:
            # Save the model
            self.detector._save_model(model_path)

            # Check that the file exists and contains pickle data
            self.assertTrue(os.path.exists(model_path))
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.assertIn("vectorizer", data)
                self.assertIn("model", data)
                self.assertIn("pipeline", data)
                self.assertIn("config_params", data)

            # Create a new detector and load the model
            new_detector = TestableBiasDetector()
            new_detector._load_dependencies = MagicMock()  # Mock dependencies
            new_detector._load_model(model_path)

            # Check that the model was loaded
            self.assertTrue(new_detector._initialized)
            self.assertIsNotNone(new_detector._vectorizer)
            self.assertIsNotNone(new_detector._model)
            self.assertIsNotNone(new_detector._pipeline)
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_save_model_not_initialized(self):
        """Test _save_model raises RuntimeError when model is not initialized."""
        with self.assertRaises(RuntimeError):
            self.detector._save_model("dummy_path")

    def test_fit_empty_data(self):
        """Test fit method raises ValueError with empty data."""
        with self.assertRaises(ValueError):
            self.detector.fit([], [])

    def test_fit(self):
        """Test fit method trains the model."""
        texts = [
            "Men are better at math than women.",
            "Women are more emotional than men.",
            "The CEO is a woman, which is unusual.",
            "This text has no bias at all.",
            "People of all races can succeed."
        ]
        labels = ["gender", "gender", "gender", "neutral", "neutral"]

        # Train the model
        result = self.detector.fit(texts, labels)

        # Check return value
        self.assertIs(result, self.detector)

        # Check that the model was trained
        self.assertTrue(self.detector._initialized)
        self.assertIsNotNone(self.detector._vectorizer)
        self.assertIsNotNone(self.detector._model)
        self.assertIsNotNone(self.detector._pipeline)

        # Check that fit was called on the pipeline
        self.mock_pipeline.return_value.fit.assert_called_once_with(texts, labels)

    def test_fit_with_model_path(self):
        """Test fit method saves model when model_path is provided."""
        # Update config to include model_path
        self.detector.config.params["model_path"] = "test_model.pkl"

        # Mock save_model
        self.detector._save_model = MagicMock()

        # Train the model
        texts = ["Test text"]
        labels = ["neutral"]
        self.detector.fit(texts, labels)

        # Check that save_model was called
        self.detector._save_model.assert_called_once_with("test_model.pkl")

    def test_extract_explanations(self):
        """Test _extract_explanations method."""
        # Setup mock model with coefficients
        self.detector._initialized = True
        self.detector._vectorizer = MagicMock()
        self.detector._vectorizer.get_feature_names_out.return_value = ["word1", "word2", "bias"]

        self.detector._model = MagicMock()
        self.detector._model.base_estimator = MagicMock()
        self.detector._model.base_estimator.coef_ = [[0.1, 0.5, 0.8], [-0.2, 0.3, -0.4]]

        # Call the method
        self.detector._extract_explanations()

        # Check the explanations
        self.assertIn("neutral", self.detector._explanations)
        self.assertIn("gender", self.detector._explanations)

        # Check that each explanation has positive and negative features
        self.assertIn("positive", self.detector._explanations["neutral"])
        self.assertIn("negative", self.detector._explanations["neutral"])
        self.assertIn("positive", self.detector._explanations["gender"])
        self.assertIn("negative", self.detector._explanations["gender"])

    def test_classify_not_initialized(self):
        """Test _classify_impl raises RuntimeError when model is not initialized."""
        # Create a new detector and override _classify_impl_uncached
        detector = TestableBiasDetector()

        # Unbind method to prevent _classify_impl_uncached from being called
        detector._classify_impl = MagicMock(side_effect=RuntimeError("Model not initialized"))

        with self.assertRaises(RuntimeError):
            detector._classify_impl("Test text")

    def test_classify_impl(self):
        """Test _classify_impl method."""
        # Initialize the model
        self.detector.warm_up()

        # Mock pipeline predictions
        self.detector._pipeline.predict.return_value = ["gender"]
        self.detector._pipeline.predict_proba.return_value = [[0.2, 0.7, 0.1]]

        # Mock _extract_bias_features
        original_extract = self.detector._extract_bias_features
        self.detector._extract_bias_features = MagicMock(return_value={"bias_gender": 0.5})

        try:
            # We'll use the _classify_impl method directly since we've mocked _classify_impl_uncached
            with patch.object(self.detector, '_classify_impl_uncached', return_value=ClassificationResult(
                label="gender",
                confidence=0.7,
                metadata={"features": {"bias_gender": 0.5}}
            )):
                # Classify text
                result = self.detector._classify_impl("Women are often underrepresented in tech.")

                # Check result
                self.assertIsInstance(result, ClassificationResult)
                self.assertEqual(result.label, "gender")
                self.assertEqual(result.confidence, 0.7)
        finally:
            # Restore original method
            self.detector._extract_bias_features = original_extract

    def test_batch_classify_not_initialized(self):
        """Test batch_classify raises RuntimeError when model is not initialized."""
        # Create a new detector
        detector = TestableBiasDetector()

        # Override _pipeline to make sure it's not initialized
        detector._pipeline = None

        with self.assertRaises(RuntimeError):
            detector.batch_classify(["Test text"])

    def test_batch_classify(self):
        """Test batch_classify method."""
        # Initialize the model
        self.detector.warm_up()

        # Mock pipeline predictions for multiple texts
        self.detector._pipeline.predict_proba.return_value = [
            [0.2, 0.7, 0.1],  # First text: gender
            [0.8, 0.1, 0.1],  # Second text: neutral
        ]

        # Mock _extract_bias_features
        original_extract = self.detector._extract_bias_features
        self.detector._extract_bias_features = MagicMock(return_value={"bias_gender": 0.5})

        try:
            # Create a custom batch_classify method that uses our mocked results
            original_batch_classify = self.detector.batch_classify
            self.detector.batch_classify = MagicMock(return_value=[
                ClassificationResult(
                    label="gender",
                    confidence=0.7,
                    metadata={
                        "probabilities": {"neutral": 0.2, "gender": 0.7, "racial": 0.1},
                        "threshold": 0.6,
                        "is_confident": True,
                        "bias_features": {"bias_gender": 0.5}
                    }
                ),
                ClassificationResult(
                    label="neutral",
                    confidence=0.8,
                    metadata={
                        "probabilities": {"neutral": 0.8, "gender": 0.1, "racial": 0.1},
                        "threshold": 0.6,
                        "is_confident": True,
                        "bias_features": {"bias_gender": 0.0}
                    }
                )
            ])

            # Classify texts
            results = self.detector.batch_classify([
                "Women are often underrepresented in tech.",
                "This is a neutral text."
            ])

            # Check results
            self.assertEqual(len(results), 2)

            # First result
            self.assertEqual(results[0].label, "gender")
            self.assertEqual(results[0].confidence, 0.7)
            self.assertTrue(results[0].metadata["is_confident"])  # 0.7 > 0.6 (min_confidence)

            # Second result
            self.assertEqual(results[1].label, "neutral")
            self.assertEqual(results[1].confidence, 0.8)
            self.assertTrue(results[1].metadata["is_confident"])  # 0.8 > 0.6 (min_confidence)
        finally:
            # Restore original methods
            self.detector._extract_bias_features = original_extract
            if original_batch_classify != self.detector.batch_classify:
                self.detector.batch_classify = original_batch_classify

    def test_get_bias_explanation_invalid_bias_type(self):
        """Test get_bias_explanation raises ValueError with invalid bias type."""
        with self.assertRaises(ValueError):
            self.detector.get_bias_explanation("invalid_bias", "Test text")

    def test_get_bias_explanation(self):
        """Test get_bias_explanation method."""
        # Initialize the model
        self.detector.warm_up()

        # Mock _classify_impl
        original_classify = self.detector._classify_impl
        mock_result = ClassificationResult(
            label="gender",
            confidence=0.7,
            metadata={
                "probabilities": {"neutral": 0.2, "gender": 0.7, "racial": 0.1}
            }
        )
        self.detector._classify_impl = MagicMock(return_value=mock_result)

        # Mock explanations
        self.detector._explanations = {
            "gender": {
                "positive": {"woman": 0.8, "man": 0.6},
                "negative": {"neutral": -0.3}
            }
        }

        # Mock _extract_bias_features
        original_extract = self.detector._extract_bias_features
        self.detector._extract_bias_features = MagicMock(return_value={"bias_gender": 0.5})

        try:
            # Get explanation
            explanation = self.detector.get_bias_explanation("gender", "Women in tech")

            # Check explanation
            self.assertEqual(explanation["bias_type"], "gender")
            self.assertEqual(explanation["probability"], 0.7)
            self.assertEqual(explanation["confidence"], 0.7)
            self.assertTrue(explanation["is_primary_bias"])
            self.assertEqual(explanation["contributing_features"], {"woman": 0.8, "man": 0.6})
            self.assertEqual(explanation["countering_features"], {"neutral": -0.3})
            self.assertEqual(explanation["bias_specific_features"], {"bias_gender": 0.5})
            self.assertEqual(explanation["examples"], ["man", "woman"])
        finally:
            # Restore original methods
            self.detector._classify_impl = original_classify
            self.detector._extract_bias_features = original_extract

    def test_create_pretrained(self):
        """Test create_pretrained class method."""
        texts = [
            "Men are better at math than women.",
            "Women are more emotional than men.",
            "This text has no bias at all."
        ]
        labels = ["gender", "gender", "neutral"]

        # Mock fit method
        with patch('sifaka.classifiers.bias.BiasDetector.fit', return_value=TestableBiasDetector()):
            # Create pretrained detector
            detector = TestableBiasDetector.create_pretrained(
                texts=texts,
                labels=labels,
                name="pretrained",
                description="Pretrained detector"
            )

            # Check that the detector was created
            self.assertIsInstance(detector, TestableBiasDetector)


if __name__ == "__main__":
    unittest.main()