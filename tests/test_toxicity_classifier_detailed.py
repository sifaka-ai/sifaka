"""
Detailed tests for the toxicity classifier.

This module contains more comprehensive tests for the toxicity classifier
to improve test coverage.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from sifaka.classifiers import ClassificationResult
from sifaka.classifiers.toxicity import ToxicityClassifier


class TestToxicityClassifierDetailed:
    """Detailed tests for the ToxicityClassifier."""

    def test_init_with_custom_parameters(self) -> None:
        """Test initializing with custom parameters."""
        classifier = ToxicityClassifier(
            threshold=0.7,
            model_name="unbiased",
            name="custom_toxicity",
            description="Custom toxicity classifier"
        )
        
        assert classifier.name == "custom_toxicity"
        assert classifier.description == "Custom toxicity classifier"
        # Access private attributes for testing
        assert classifier._threshold == 0.7
        assert classifier._model_name == "unbiased"
        assert classifier._initialized is False
        assert classifier._model is None

    def test_classify_empty_text(self) -> None:
        """Test classifying empty text."""
        classifier = ToxicityClassifier()
        result = classifier.classify("")
        
        assert result.label == "non_toxic"
        assert result.confidence == 1.0
        assert result.metadata["input_length"] == 0
        assert result.metadata["reason"] == "empty_text"
        assert result.metadata["scores"] == {}

    @patch("importlib.import_module")
    def test_load_detoxify_success(self, mock_import: MagicMock) -> None:
        """Test successful loading of the Detoxify library."""
        # Create a mock Detoxify module
        mock_detoxify = MagicMock()
        mock_detoxify_class = MagicMock()
        mock_detoxify.Detoxify.return_value = mock_detoxify_class
        mock_import.return_value = mock_detoxify
        
        classifier = ToxicityClassifier()
        model = classifier._load_detoxify()
        
        # Check that the import was called correctly
        mock_import.assert_called_once_with("detoxify")
        # Check that Detoxify was initialized with the correct model type
        mock_detoxify.Detoxify.assert_called_once_with(model_type="original")
        # Check that the model was returned
        assert model == mock_detoxify_class

    @patch("importlib.import_module")
    def test_load_detoxify_import_error(self, mock_import: MagicMock) -> None:
        """Test handling of ImportError when loading Detoxify."""
        # Make the import raise an ImportError
        mock_import.side_effect = ImportError("No module named 'detoxify'")
        
        classifier = ToxicityClassifier()
        
        # Check that the correct error is raised
        with pytest.raises(ImportError) as excinfo:
            classifier._load_detoxify()
        
        assert "detoxify package is required" in str(excinfo.value)
        assert "pip install detoxify" in str(excinfo.value)

    @patch("importlib.import_module")
    def test_load_detoxify_other_error(self, mock_import: MagicMock) -> None:
        """Test handling of other errors when loading Detoxify."""
        # Make the Detoxify constructor raise an error
        mock_detoxify = MagicMock()
        mock_detoxify.Detoxify.side_effect = RuntimeError("Failed to initialize model")
        mock_import.return_value = mock_detoxify
        
        classifier = ToxicityClassifier()
        
        # Check that the correct error is raised
        with pytest.raises(RuntimeError) as excinfo:
            classifier._load_detoxify()
        
        assert "Failed to load Detoxify model" in str(excinfo.value)

    @patch.object(ToxicityClassifier, "_load_detoxify")
    def test_initialize(self, mock_load_detoxify: MagicMock) -> None:
        """Test initialization of the toxicity model."""
        # Create a mock model
        mock_model = MagicMock()
        mock_load_detoxify.return_value = mock_model
        
        classifier = ToxicityClassifier()
        assert classifier._initialized is False
        assert classifier._model is None
        
        # Initialize the model
        classifier._initialize()
        
        # Check that the model was loaded
        mock_load_detoxify.assert_called_once()
        assert classifier._initialized is True
        assert classifier._model == mock_model
        
        # Reset the mock and call initialize again
        mock_load_detoxify.reset_mock()
        classifier._initialize()
        
        # Check that the model was not loaded again
        mock_load_detoxify.assert_not_called()

    @patch.object(ToxicityClassifier, "_initialize")
    def test_classify_with_mock_model(self, mock_initialize: MagicMock) -> None:
        """Test classification with a mock model."""
        # Create a classifier with a mock model
        classifier = ToxicityClassifier()
        mock_model = MagicMock()
        classifier._model = mock_model
        classifier._initialized = True
        
        # Set up the mock model to return toxicity scores
        mock_model.predict.return_value = {
            "toxic": 0.8,
            "severe_toxic": 0.3,
            "obscene": 0.6,
            "threat": 0.1,
            "insult": 0.7,
            "identity_attack": 0.2
        }
        
        # Classify some text
        result = classifier.classify("This is a test text.")
        
        # Check that initialize was called
        mock_initialize.assert_called_once()
        
        # Check that the model was called with the correct text
        mock_model.predict.assert_called_once_with("This is a test text.")
        
        # Check the result
        assert result.label == "toxic"  # toxic has the highest score above threshold
        assert result.confidence == 0.8
        assert "Text classified as toxic" in result.metadata["message"]
        assert result.metadata["input_length"] == len("This is a test text.")
        assert result.metadata["scores"]["toxic"] == 0.8

    @patch.object(ToxicityClassifier, "_initialize")
    def test_classify_non_toxic(self, mock_initialize: MagicMock) -> None:
        """Test classification of non-toxic text."""
        # Create a classifier with a mock model
        classifier = ToxicityClassifier()
        mock_model = MagicMock()
        classifier._model = mock_model
        classifier._initialized = True
        
        # Set up the mock model to return low toxicity scores
        mock_model.predict.return_value = {
            "toxic": 0.1,
            "severe_toxic": 0.05,
            "obscene": 0.2,
            "threat": 0.01,
            "insult": 0.15,
            "identity_attack": 0.03
        }
        
        # Classify some text
        result = classifier.classify("This is a friendly message.")
        
        # Check the result
        assert result.label == "non_toxic"
        assert result.confidence == 0.8  # 1.0 - max(scores) = 1.0 - 0.2 = 0.8
        assert "Text classified as non-toxic" in result.metadata["message"]

    @patch.object(ToxicityClassifier, "_initialize")
    def test_classify_error_handling(self, mock_initialize: MagicMock) -> None:
        """Test error handling during classification."""
        # Create a classifier with a mock model that raises an error
        classifier = ToxicityClassifier()
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Model prediction failed")
        classifier._model = mock_model
        classifier._initialized = True
        
        # Classify some text
        result = classifier.classify("This is a test text.")
        
        # Check that the result indicates an error
        assert result.label == "non_toxic"
        assert result.confidence == 0.5  # Default confidence for errors
        assert "Model prediction failed" in result.metadata["error"]
        assert result.metadata["reason"] == "classification_error"
