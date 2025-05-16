"""
Similarity validator for comparing text against reference content.

This module provides a validator that checks how similar text is to reference content
using various similarity metrics.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import re
from ..types import ValidationResult


class SimilarityValidator:
    """
    Validator that checks text similarity against reference content.

    This validator compares input text against reference content using
    configurable similarity metrics. It can be used to ensure text is
    similar enough to (or different enough from) reference content.
    """

    # Available similarity metrics
    METRICS = {
        "jaccard": "Token-based Jaccard similarity",
        "cosine": "Vector-based cosine similarity",
        "levenshtein": "Edit distance-based similarity",
        "exact": "Exact match checking",
    }

    def __init__(
        self,
        reference: Union[str, List[str]],
        min_similarity: Optional[float] = None,
        max_similarity: Optional[float] = None,
        metric: str = "jaccard",
        case_sensitive: bool = False,
        custom_metric_func: Optional[Callable[[str, str], float]] = None,
        tokenize_method: str = "word",
        remove_punctuation: bool = True,
        normalize_whitespace: bool = True,
    ):
        """
        Initialize the similarity validator.

        Args:
            reference: Reference text or list of reference texts to compare against
            min_similarity: Minimum similarity score required (0.0 to 1.0)
            max_similarity: Maximum similarity score allowed (0.0 to 1.0)
            metric: Similarity metric to use ('jaccard', 'cosine', 'levenshtein', 'exact')
            case_sensitive: Whether comparisons should be case sensitive
            custom_metric_func: Optional custom function for similarity calculation
            tokenize_method: Method for tokenizing text ('word', 'character', 'sentence')
            remove_punctuation: Whether to remove punctuation before comparison
            normalize_whitespace: Whether to normalize whitespace before comparison
        """
        self.reference = [reference] if isinstance(reference, str) else reference
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.metric = metric
        self.case_sensitive = case_sensitive
        self.custom_metric_func = custom_metric_func
        self.tokenize_method = tokenize_method
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace

        # Validate configuration
        if min_similarity is not None and max_similarity is not None:
            if min_similarity > max_similarity:
                raise ValueError("min_similarity cannot be greater than max_similarity")

        if min_similarity is None and max_similarity is None:
            raise ValueError("At least one of min_similarity or max_similarity must be specified")

        if metric not in self.METRICS and custom_metric_func is None:
            raise ValueError(
                f"Unknown metric: {metric}. Available metrics: {list(self.METRICS.keys())}"
            )

    def preprocess_text(self, text: str) -> str:
        """Preprocess text according to configuration."""
        if not self.case_sensitive:
            text = text.lower()

        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text according to configuration."""
        if self.tokenize_method == "character":
            return list(text)
        elif self.tokenize_method == "sentence":
            return re.split(r"[.!?]+", text)
        else:  # Default to word tokenization
            return text.split()

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        # Preprocess
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)

        # Tokenize
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))

        # Calculate Jaccard similarity (intersection over union)
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        try:
            import numpy as np
            from sklearn.feature_extraction.text import CountVectorizer
        except ImportError:
            raise ImportError(
                "scikit-learn and numpy are required for cosine similarity. "
                "Install them with: pip install scikit-learn numpy"
            )

        # Preprocess
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)

        # Vectorize
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])

        # Calculate cosine similarity
        similarity = (vectors[0] * vectors[1].T).toarray()[0, 0]
        denominator = np.sqrt((vectors[0].toarray() ** 2).sum()) * np.sqrt(
            (vectors[1].toarray() ** 2).sum()
        )

        return similarity / denominator if denominator > 0 else 0.0

    def levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Calculate Levenshtein-based similarity between two texts."""
        try:
            import Levenshtein
        except ImportError:
            raise ImportError(
                "python-Levenshtein is required for Levenshtein similarity. "
                "Install it with: pip install python-Levenshtein"
            )

        # Preprocess
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)

        # Calculate edit distance
        distance = Levenshtein.distance(text1, text2)
        max_length = max(len(text1), len(text2))

        # Convert to similarity (1.0 means identical, 0.0 means completely different)
        return 1.0 - (distance / max_length) if max_length > 0 else 1.0

    def exact_match(self, text1: str, text2: str) -> float:
        """Check for exact match between two texts."""
        # Preprocess
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)

        # Return 1.0 for exact match, 0.0 otherwise
        return 1.0 if text1 == text2 else 0.0

    def get_similarity_function(self) -> Callable[[str, str], float]:
        """Get the appropriate similarity function based on configuration."""
        if self.custom_metric_func is not None:
            return self.custom_metric_func

        if self.metric == "jaccard":
            return self.jaccard_similarity
        elif self.metric == "cosine":
            return self.cosine_similarity
        elif self.metric == "levenshtein":
            return self.levenshtein_similarity
        elif self.metric == "exact":
            return self.exact_match
        else:
            # Default to Jaccard if metric is unknown
            return self.jaccard_similarity

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text similarity against reference content.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult with similarity metrics
        """
        if not text.strip():
            return ValidationResult(
                passed=False,
                message="Empty text fails similarity validation",
                score=0.0,
                issues=["Text is empty"],
                suggestions=["Provide non-empty content"],
                metadata={"similarity_scores": []},
            )

        if not self.reference or all(not ref.strip() for ref in self.reference):
            return ValidationResult(
                passed=False,
                message="No valid reference text for comparison",
                score=0.0,
                issues=["Reference text is empty"],
                suggestions=["Provide valid reference text"],
                metadata={"similarity_scores": []},
            )

        try:
            # Get similarity function
            similarity_func = self.get_similarity_function()

            # Calculate similarity for each reference text
            similarity_scores = []
            for i, ref in enumerate(self.reference):
                score = similarity_func(text, ref)
                similarity_scores.append(
                    {
                        "reference_index": i,
                        "similarity": score,
                    }
                )

            # Find the best matching reference (highest similarity)
            best_match = max(similarity_scores, key=lambda x: x["similarity"])
            best_score = best_match["similarity"]
            best_index = best_match["reference_index"]

            # Determine if validation passed
            min_passed = True if self.min_similarity is None else best_score >= self.min_similarity
            max_passed = True if self.max_similarity is None else best_score <= self.max_similarity
            passed = min_passed and max_passed

            # Create issues and suggestions
            issues = []
            suggestions = []

            if not min_passed:
                issues.append(
                    f"Similarity score ({best_score:.2f}) is below minimum threshold ({self.min_similarity:.2f})"
                )
                suggestions.append("Make text more similar to the reference content")

            if not max_passed:
                issues.append(
                    f"Similarity score ({best_score:.2f}) exceeds maximum threshold ({self.max_similarity:.2f})"
                )
                suggestions.append("Make text more different from the reference content")

            # Create message
            if passed:
                message = f"Similarity validation passed with score {best_score:.2f} using {self.metric} metric"
            else:
                message = f"Similarity validation failed with score {best_score:.2f} using {self.metric} metric"

            # Use similarity score directly for the result score
            score = best_score if passed else (0.1 if min_passed else 0.0)

            return ValidationResult(
                passed=passed,
                message=message,
                score=score,
                issues=issues,
                suggestions=suggestions,
                metadata={
                    "similarity_scores": similarity_scores,
                    "best_match": {
                        "reference_index": best_index,
                        "similarity": best_score,
                        "reference_text": (
                            self.reference[best_index][:100] + "..."
                            if len(self.reference[best_index]) > 100
                            else self.reference[best_index]
                        ),
                    },
                    "metric": self.metric,
                },
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Similarity validation error: {str(e)}",
                score=0.0,
                issues=[f"Error during similarity validation: {str(e)}"],
                suggestions=["Check input text format or try a different similarity metric"],
                metadata={"error": str(e)},
            )
