"""
Detailed tests for the spam classifier.

This module contains more comprehensive tests for the spam classifier
to improve test coverage.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import pickle
from typing import Dict, Any, Tuple

from sifaka.classifiers import ClassificationResult
from sifaka.classifiers.spam import SpamClassifier


# Create mock modules for testing
class MockSklearn:
    """Mock for the sklearn module."""

    class naive_bayes:
        """Mock for sklearn.naive_bayes."""

        class MultinomialNB:
            """Mock for MultinomialNB."""

            def __init__(self, alpha=1.0):
                self.alpha = alpha
                self.classes_ = ["spam", "ham"]
                self._fitted = False

            def fit(self, X, y):
                """Mock fit method."""
                self._fitted = True
                return self

            def predict_proba(self, X):
                """Mock predict_proba method."""
                # Return different probabilities based on the input
                if hasattr(X, "toarray") and callable(X.toarray):
                    # This is a mock sparse matrix
                    if "spam" in str(X.text).lower():
                        return [[0.8, 0.2]]  # High spam probability
                    elif "ham" in str(X.text).lower():
                        return [[0.1, 0.9]]  # High ham probability
                    else:
                        return [[0.4, 0.6]]  # Moderate ham probability
                else:
                    # For batch classification
                    results = []
                    for i in range(len(X)):
                        if "spam" in str(X[i]).lower():
                            results.append([0.8, 0.2])
                        elif "ham" in str(X[i]).lower():
                            results.append([0.1, 0.9])
                        else:
                            results.append([0.4, 0.6])
                    return results


class MockFeatureExtraction:
    """Mock for sklearn.feature_extraction."""

    class text:
        """Mock for sklearn.feature_extraction.text."""

        class TfidfVectorizer:
            """Mock for TfidfVectorizer."""

            def __init__(self, max_features=None, min_df=None, max_df=None, stop_words=None):
                self.max_features = max_features
                self.min_df = min_df
                self.max_df = max_df
                self.stop_words = stop_words
                self.feature_names = [
                    "buy",
                    "offer",
                    "free",
                    "money",
                    "work",
                    "meeting",
                    "report",
                    "email",
                    "project",
                    "office",
                ]
                self.text = None

            def fit_transform(self, texts):
                """Mock fit_transform method."""
                return MockSparseMatrix(texts)

            def transform(self, texts):
                """Mock transform method."""
                if isinstance(texts, list):
                    return MockSparseMatrix(texts)
                return MockSparseMatrix(texts)

            def get_feature_names_out(self):
                """Mock get_feature_names_out method."""
                return self.feature_names


class MockSparseMatrix:
    """Mock for sparse matrix."""

    def __init__(self, text):
        self.text = text

    def toarray(self):
        """Mock toarray method."""
        # Create a mock array with some non-zero values
        if isinstance(self.text, list):
            return [[0.1, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in self.text]
        else:
            return [[0.1, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    def __getitem__(self, index):
        """Support indexing for batch operations."""
        if isinstance(self.text, list):
            return MockSparseMatrix(self.text[index])
        raise IndexError("Cannot index a single text")


class MockNumpy:
    """Mock for numpy."""

    @staticmethod
    def argsort(arr):
        """Mock argsort method."""
        # Return indices of non-zero elements
        return [i for i, val in enumerate(arr) if val > 0]


# Create a mock for the modules
@pytest.fixture
def mock_modules():
    """Fixture to mock the required modules."""
    sklearn = MagicMock()
    sklearn.naive_bayes = MockSklearn.naive_bayes

    feature_extraction = MockFeatureExtraction()
    sklearn.feature_extraction = feature_extraction

    numpy = MockNumpy()

    # Create a patch for importlib.import_module
    with patch("importlib.import_module") as mock_import:
        # Configure the mock to return our mock modules
        def side_effect(name):
            if name == "sklearn":
                return sklearn
            elif name == "numpy":
                return numpy
            elif name == "sklearn.naive_bayes":
                return sklearn.naive_bayes
            elif name == "sklearn.feature_extraction.text":
                return feature_extraction.text
            else:
                # For other modules, use the real import
                import importlib

                return importlib.import_module(name)

        mock_import.side_effect = side_effect
        yield (sklearn, numpy)


class TestSpamClassifierDetailed:
    """Detailed tests for the SpamClassifier."""

    def test_init_with_custom_parameters(self) -> None:
        """Test initializing with custom parameters."""
        classifier = SpamClassifier(
            model_path="custom_model.pkl",
            threshold=0.7,
            name="custom_spam",
            description="Custom spam classifier",
        )

        assert classifier.name == "custom_spam"
        assert classifier.description == "Custom spam classifier"
        # Access private attributes for testing
        assert classifier._model_path == "custom_model.pkl"
        assert classifier._threshold == 0.7
        assert classifier._model is None
        assert classifier._vectorizer is None
        assert classifier._initialized is False

    def test_load_scikit_learn_error(self) -> None:
        """Test error handling when scikit-learn is not available."""
        with patch("importlib.import_module", side_effect=ImportError("No module named 'sklearn'")):
            classifier = SpamClassifier()

            with pytest.raises(ImportError) as excinfo:
                classifier._load_scikit_learn()

            assert "scikit-learn and numpy packages are required" in str(excinfo.value)
            assert "pip install scikit-learn numpy" in str(excinfo.value)

    def test_create_default_model(self, mock_modules) -> None:
        """Test creating a default model."""
        # Patch the _load_scikit_learn method to return our mock modules
        sklearn, numpy = mock_modules
        with patch.object(SpamClassifier, "_load_scikit_learn", return_value=(sklearn, numpy)):
            classifier = SpamClassifier()

            model, vectorizer = classifier._create_default_model()

            assert model is not None
            assert vectorizer is not None
            assert model._fitted is True  # Should be fitted with default data
            assert vectorizer.max_features == 5000
            assert vectorizer.stop_words == "english"

    def test_load_model_with_nonexistent_path(self, mock_modules) -> None:
        """Test loading a model with a nonexistent path."""
        # Patch the _load_scikit_learn method to return our mock modules
        sklearn, numpy = mock_modules

        # Directly patch os.path.exists instead of using patch as a context manager
        original_exists = os.path.exists
        try:
            os.path.exists = lambda path: False

            with patch.object(SpamClassifier, "_load_scikit_learn", return_value=(sklearn, numpy)):
                classifier = SpamClassifier(model_path="nonexistent.pkl")

                # Should fall back to creating a default model
                model, vectorizer = classifier._load_model()

                assert model is not None
                assert vectorizer is not None
                assert model._fitted is True
        finally:
            # Restore the original function
            os.path.exists = original_exists

    def test_load_model_with_valid_path(self) -> None:
        """Test loading a model from a valid path."""
        # Create a mock model file
        model_data = {"model": MagicMock(), "vectorizer": MagicMock()}

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", create=True),
            patch("pickle.load", return_value=model_data),
        ):

            classifier = SpamClassifier(model_path="valid_model.pkl")
            model, vectorizer = classifier._load_model()

            assert model == model_data["model"]
            assert vectorizer == model_data["vectorizer"]

    def test_initialize(self, mock_modules) -> None:
        """Test initialization of the classifier."""
        # Patch the _load_scikit_learn method to return our mock modules
        sklearn, numpy = mock_modules
        with patch.object(SpamClassifier, "_load_scikit_learn", return_value=(sklearn, numpy)):
            classifier = SpamClassifier()

            assert classifier._initialized is False
            assert classifier._model is None
            assert classifier._vectorizer is None

            # Initialize the classifier
            classifier._initialize()

            assert classifier._initialized is True
            assert classifier._model is not None
            assert classifier._vectorizer is not None

            # Calling initialize again should not change anything
            model = classifier._model
            vectorizer = classifier._vectorizer

            classifier._initialize()

            assert classifier._model == model
            assert classifier._vectorizer == vectorizer

    def test_classify_empty_text(self) -> None:
        """Test classifying empty text."""
        classifier = SpamClassifier()

        result = classifier.classify("")

        assert result.label == "ham"
        assert result.confidence == 0.7
        assert result.metadata["input_length"] == 0
        assert result.metadata["reason"] == "empty_text"
        assert result.metadata["probabilities"] == {"spam": 0.3, "ham": 0.7}

    def test_classify_spam_text(self, mock_modules) -> None:
        """Test classifying spam text."""
        # Patch the _load_scikit_learn method to return our mock modules
        sklearn, numpy = mock_modules
        with patch.object(SpamClassifier, "_load_scikit_learn", return_value=(sklearn, numpy)):
            classifier = SpamClassifier()

            # Initialize the classifier with our mocks
            classifier._initialize()

            # Set up the model to return spam probabilities
            if hasattr(classifier._model, "predict_proba"):
                classifier._model.predict_proba = MagicMock(return_value=[[0.8, 0.2]])

            result = classifier.classify("Buy now! Free offer! Make money fast! This is spam text.")

            assert result.label == "spam"
            assert result.confidence > 0.7
            assert result.metadata["input_length"] == len(
                "Buy now! Free offer! Make money fast! This is spam text."
            )
            assert "probabilities" in result.metadata
            assert "top_features" in result.metadata

    def test_classify_ham_text(self, mock_modules) -> None:
        """Test classifying ham (non-spam) text."""
        # Patch the _load_scikit_learn method to return our mock modules
        sklearn, numpy = mock_modules
        with patch.object(SpamClassifier, "_load_scikit_learn", return_value=(sklearn, numpy)):
            classifier = SpamClassifier()

            # Initialize the classifier with our mocks
            classifier._initialize()

            # Set up the model to return ham probabilities
            if hasattr(classifier._model, "predict_proba"):
                classifier._model.predict_proba = MagicMock(return_value=[[0.1, 0.9]])

            result = classifier.classify(
                "Please review the report for tomorrow's meeting. This is ham text."
            )

            assert result.label == "ham"
            assert result.confidence > 0.7
            assert result.metadata["input_length"] == len(
                "Please review the report for tomorrow's meeting. This is ham text."
            )
            assert "probabilities" in result.metadata
            assert "top_features" in result.metadata

    def test_classify_with_custom_threshold(self, mock_modules) -> None:
        """Test classifying with a custom threshold."""
        # Patch the _load_scikit_learn method to return our mock modules
        sklearn, numpy = mock_modules
        with patch.object(SpamClassifier, "_load_scikit_learn", return_value=(sklearn, numpy)):
            # Set a high threshold that will classify borderline cases as ham
            classifier = SpamClassifier(threshold=0.9)

            # Initialize the classifier with our mocks
            classifier._initialize()

            # Set up the model to return moderate spam probabilities
            if hasattr(classifier._model, "predict_proba"):
                classifier._model.predict_proba = MagicMock(return_value=[[0.8, 0.2]])

            result = classifier.classify("This text has some spam-like words: offer, money.")

            # With the mock returning 0.8 for spam probability and threshold at 0.9,
            # this should be classified as ham
            assert result.label == "ham"
            assert "probabilities" in result.metadata
            assert result.metadata["probabilities"]["spam"] < classifier._threshold

    def test_batch_classify(self, mock_modules) -> None:
        """Test batch classification of multiple texts."""
        # Patch the _load_scikit_learn method to return our mock modules
        sklearn, numpy = mock_modules
        with patch.object(SpamClassifier, "_load_scikit_learn", return_value=(sklearn, numpy)):
            classifier = SpamClassifier()

            # Initialize the classifier with our mocks
            classifier._initialize()

            # Set up the model to return different probabilities for different texts
            if hasattr(classifier._model, "predict_proba"):
                # For batch classification, we need to return multiple results
                classifier._model.predict_proba = MagicMock(
                    return_value=[
                        [0.8, 0.2],  # spam
                        [0.1, 0.9],  # ham
                        [0.3, 0.7],  # neutral
                    ]
                )

            texts = [
                "Buy now! Free offer! This is spam.",
                "Please review the report. This is ham.",
                "",  # Empty text is handled separately
                "This is a neutral text.",
            ]

            results = classifier.batch_classify(texts)

            assert len(results) == 4
            # Empty text is handled separately and should be ham
            assert results[2].label == "ham"
            assert results[2].metadata["reason"] == "empty_text"

    def test_classify_error_handling(self) -> None:
        """Test error handling during classification."""
        classifier = SpamClassifier()

        # Patch _initialize to raise an exception
        with patch.object(
            SpamClassifier, "_initialize", side_effect=RuntimeError("Initialization failed")
        ):
            result = classifier.classify("Test text")

            assert result.label == "ham"  # Default fallback
            assert result.confidence == 0.5  # Default confidence for errors
            assert "error" in result.metadata
            assert "Initialization failed" in result.metadata["error"]
            assert result.metadata["reason"] == "classification_error"
