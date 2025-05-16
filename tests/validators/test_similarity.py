"""
Tests for the SimilarityValidator.
"""

import pytest
from unittest.mock import patch
from sifaka.validators import SimilarityValidator
from sifaka.types import ValidationResult


class TestSimilarityValidator:
    """Test cases for SimilarityValidator."""

    def test_initialization(self):
        """Test SimilarityValidator initializes with correct parameters."""
        validator = SimilarityValidator(
            reference="This is a reference text",
            min_similarity=0.7,
            max_similarity=0.9,
            metric="jaccard",
            case_sensitive=True,
            tokenize_method="word",
            remove_punctuation=False,
            normalize_whitespace=True,
        )

        assert validator.reference == ["This is a reference text"]
        assert validator.min_similarity == 0.7
        assert validator.max_similarity == 0.9
        assert validator.metric == "jaccard"
        assert validator.case_sensitive is True
        assert validator.tokenize_method == "word"
        assert validator.remove_punctuation is False
        assert validator.normalize_whitespace is True

    def test_invalid_configuration(self):
        """Test validation of validator configuration."""
        # Test min_similarity > max_similarity
        with pytest.raises(ValueError) as excinfo:
            SimilarityValidator(reference="Test", min_similarity=0.8, max_similarity=0.7)
        assert "min_similarity cannot be greater than max_similarity" in str(excinfo.value)

        # Test no thresholds provided
        with pytest.raises(ValueError) as excinfo:
            SimilarityValidator(reference="Test")
        assert "At least one of min_similarity or max_similarity must be specified" in str(
            excinfo.value
        )

        # Test invalid metric
        with pytest.raises(ValueError) as excinfo:
            SimilarityValidator(reference="Test", min_similarity=0.5, metric="invalid_metric")
        assert "Unknown metric" in str(excinfo.value)

    def test_empty_text(self):
        """Test handling of empty text."""
        validator = SimilarityValidator(reference="Reference", min_similarity=0.5)
        result = validator.validate("")

        assert result.passed is False
        assert "Empty text" in result.message
        assert result.score == 0.0
        assert "Text is empty" in result.issues[0]

    def test_empty_reference(self):
        """Test handling of empty reference."""
        validator = SimilarityValidator(reference="", min_similarity=0.5)
        result = validator.validate("Some text")

        assert result.passed is False
        assert "No valid reference text" in result.message
        assert result.score == 0.0
        assert "Reference text is empty" in result.issues[0]

    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        validator = SimilarityValidator(
            reference="The quick brown fox jumps over the lazy dog",
            min_similarity=0.4,
            metric="jaccard",
        )

        # Test high similarity
        result1 = validator.validate("The quick brown fox jumps over a lazy dog")
        assert result1.passed is True
        assert result1.score > 0.4
        assert "using jaccard metric" in result1.message

        # Test low similarity
        result2 = validator.validate("Completely different text with no similarity")
        assert result2.passed is False
        assert result2.score < 0.4
        assert "below minimum threshold" in result2.issues[0]

    def test_exact_match(self):
        """Test exact match similarity."""
        validator = SimilarityValidator(
            reference="Exact text to match",
            min_similarity=1.0,  # Require exact match
            metric="exact",
        )

        # Test exact match
        result1 = validator.validate("Exact text to match")
        assert result1.passed is True
        assert result1.score == 1.0

        # Test case sensitivity
        validator_case_sensitive = SimilarityValidator(
            reference="Exact text to match", min_similarity=1.0, metric="exact", case_sensitive=True
        )
        result2 = validator_case_sensitive.validate("exact text to match")
        assert result2.passed is False
        assert result2.score == 0.0

    def test_preprocess_text(self):
        """Test text preprocessing."""
        validator = SimilarityValidator(
            reference="Reference text with punctuation!",
            min_similarity=0.7,
            remove_punctuation=True,
            normalize_whitespace=True,
            case_sensitive=False,
        )

        # Check preprocessing
        processed = validator.preprocess_text("REFerence    TEXT with, punctuation?!")
        assert processed == "reference text with punctuation"

        # Verify it affects similarity calculation
        result = validator.validate("REFERENCE text WITH punctuation!!")
        assert result.passed is True  # Should match because preprocessing normalizes the text

    def test_tokenize_methods(self):
        """Test different tokenization methods."""
        reference = "This is a test sentence. Here's another one."

        # Word tokenization
        validator1 = SimilarityValidator(
            reference=reference, min_similarity=0.5, tokenize_method="word"
        )
        tokens1 = validator1.tokenize(reference)
        assert len(tokens1) > 0
        assert "This" in tokens1

        # Character tokenization
        validator2 = SimilarityValidator(
            reference=reference, min_similarity=0.5, tokenize_method="character"
        )
        tokens2 = validator2.tokenize(reference)
        assert len(tokens2) == len(reference)
        assert "T" in tokens2

        # Sentence tokenization
        validator3 = SimilarityValidator(
            reference=reference, min_similarity=0.5, tokenize_method="sentence"
        )
        tokens3 = validator3.tokenize(reference)
        assert len(tokens3) == 2
        assert "This is a test sentence" in tokens3

    def test_multiple_references(self):
        """Test similarity against multiple reference texts."""
        validator = SimilarityValidator(
            reference=["First reference", "Second reference", "Third reference"],
            min_similarity=0.5,
            metric="jaccard",
        )

        # Should match the most similar reference
        result = validator.validate("This is similar to the second reference text")
        assert result.passed is True
        assert result.metadata["best_match"]["reference_index"] == 1  # Second reference
        assert len(result.metadata["similarity_scores"]) == 3

    @patch("sifaka.validators.similarity.Levenshtein")
    def test_levenshtein_similarity(self, mock_levenshtein):
        """Test Levenshtein-based similarity calculation."""
        # Mock the Levenshtein distance function
        mock_levenshtein.distance.return_value = 5  # Fixed distance for testing

        validator = SimilarityValidator(
            reference="Reference text", min_similarity=0.5, metric="levenshtein"
        )

        result = validator.validate("Different text")

        mock_levenshtein.distance.assert_called_once()
        assert "using levenshtein metric" in result.message

    @patch("sifaka.validators.similarity.np")
    @patch("sifaka.validators.similarity.CountVectorizer")
    def test_cosine_similarity(self, mock_vectorizer_cls, mock_np):
        """Test cosine similarity calculation."""
        # Create mocks for the vectorizer and numpy
        mock_vectorizer = mock_vectorizer_cls.return_value
        mock_vectors = mock_vectorizer.fit_transform.return_value

        # Set up the mocks for cosine similarity calculation
        mock_vectors.__getitem__.return_value = mock_vectors
        mock_vectors.__mul__.return_value = mock_vectors
        mock_vectors.T = mock_vectors
        mock_vectors.toarray.return_value = [[0.8]]

        # Mock numpy calculations
        mock_np.sqrt.return_value = 1.0

        validator = SimilarityValidator(
            reference="Reference text", min_similarity=0.7, metric="cosine"
        )

        result = validator.validate("Similar text")

        # Verify the mocks were called
        mock_vectorizer_cls.assert_called_once()
        mock_vectorizer.fit_transform.assert_called_once()
        assert "using cosine metric" in result.message

    def test_custom_metric_function(self):
        """Test using a custom metric function."""

        # Define a simple custom similarity function that returns a fixed value
        def custom_similarity(text1, text2):
            return 0.75 if "similar" in text2.lower() else 0.25

        validator = SimilarityValidator(
            reference="Reference text", min_similarity=0.5, custom_metric_func=custom_similarity
        )

        # Test with text that will get 0.75 similarity
        result1 = validator.validate("This is similar text")
        assert result1.passed is True
        assert result1.score == 0.75

        # Test with text that will get 0.25 similarity
        result2 = validator.validate("This is different text")
        assert result2.passed is False
        assert result2.score == 0.0  # Score is 0 when failing min_similarity
        assert "below minimum threshold" in result2.issues[0]

    def test_max_similarity_threshold(self):
        """Test max_similarity threshold enforcement."""
        validator = SimilarityValidator(
            reference="Reference text for testing",
            max_similarity=0.7,  # Text must be different enough
            metric="jaccard",
        )

        # Test text that's too similar
        result1 = validator.validate("Reference text for tests")
        assert result1.passed is False
        assert "exceeds maximum threshold" in result1.issues[0]

        # Test text that's different enough
        result2 = validator.validate("Something completely different")
        assert result2.passed is True
