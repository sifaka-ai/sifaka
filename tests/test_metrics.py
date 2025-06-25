"""Tests for metrics module."""

import pytest
from sifaka.core.metrics import analyze_suggestion_implementation


class TestAnalyzeSuggestionImplementation:
    """Test the analyze_suggestion_implementation function."""

    def test_empty_suggestions(self):
        """Test with no suggestions."""
        result = analyze_suggestion_implementation([], "old text", "new text")
        assert result["suggestions_given"] == []
        assert result["suggestions_implemented"] == []
        assert result["suggestions_not_implemented"] == []
        assert result["implementation_rate"] == 0
        assert result["implementation_count"] == 0

    def test_single_add_suggestion_implemented(self):
        """Test when an 'add' suggestion is implemented."""
        suggestions = ["Add examples of machine learning applications."]
        old_text = "Machine learning is a field of AI."
        new_text = "Machine learning is a field of AI. Examples include image recognition and natural language processing."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        assert result["suggestions_implemented"] == suggestions
        assert result["suggestions_not_implemented"] == []
        assert result["implementation_rate"] == 1.0
        assert result["implementation_count"] == 1

    def test_single_add_suggestion_not_implemented(self):
        """Test when an 'add' suggestion is not implemented."""
        suggestions = ["Add examples of quantum computing."]
        old_text = "Machine learning is a field of AI."
        new_text = "Machine learning is a field of AI and has many applications."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        assert result["suggestions_implemented"] == []
        assert result["suggestions_not_implemented"] == suggestions
        assert result["implementation_rate"] == 0.0
        assert result["implementation_count"] == 0

    def test_multiple_pattern_types(self):
        """Test various suggestion patterns."""
        suggestions = [
            "Include statistics about renewable energy.",
            "Provide more details on solar panels.",
            "Expand on wind energy benefits.",
            "Mention environmental impact.",
            "Discuss cost effectiveness.",
            "Add examples of successful projects."
        ]
        old_text = "Renewable energy is important."
        new_text = """
        Renewable energy is important. Statistics show that renewable energy
        now accounts for 30% of global electricity. Solar panels have become
        more efficient. Wind energy benefits include reduced emissions.
        The environmental impact is significant. Successful projects include
        the largest solar farm in California.
        """
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        assert len(result["suggestions_implemented"]) >= 4
        assert result["implementation_rate"] >= 0.66

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        suggestions = ["Add information about PYTHON programming."]
        old_text = "Programming is fun."
        new_text = "Programming is fun. Python is a popular language."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1

    def test_partial_implementation(self):
        """Test when only some suggestions are implemented."""
        suggestions = [
            "Add examples of databases.",
            "Include information about SQL.",
            "Provide details on NoSQL systems."
        ]
        old_text = "Databases store data."
        new_text = "Databases store data. Examples include MySQL and PostgreSQL. SQL is the standard query language."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        # At least one suggestion should be implemented (examples of databases)
        assert len(result["suggestions_implemented"]) >= 1
        assert len(result["suggestions_not_implemented"]) <= 2
        assert result["implementation_rate"] >= 1/3

    def test_word_filtering(self):
        """Test that common words are filtered out."""
        suggestions = ["Add the and for with information."]
        old_text = "Test text."
        new_text = "Test text with more words."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        # Should not count implementation based on common words
        assert result["implementation_count"] == 0

    def test_word_count_increase(self):
        """Test implementation detection based on word count increase."""
        suggestions = ["Expand on machine learning algorithms."]
        old_text = "Machine learning is powerful."
        new_text = "Machine learning is powerful. Machine learning algorithms include decision trees and neural networks."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1

    def test_unicode_and_special_characters(self):
        """Test with unicode and special characters."""
        suggestions = ["Add information about café culture."]
        old_text = "Paris is beautiful."
        new_text = "Paris is beautiful. The café culture is vibrant with many bistros."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        assert result["suggestions_implemented"] == suggestions

    def test_punctuation_in_suggestions(self):
        """Test suggestions with various punctuation."""
        suggestions = [
            "Add, if possible, more examples.",
            "Include (but don't overdo) statistics.",
            "Provide details - especially about costs."
        ]
        old_text = "Initial text."
        new_text = "Initial text with examples and statistics about costs."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        assert len(result["suggestions_implemented"]) >= 2

    def test_very_long_text(self):
        """Test with long texts to check performance."""
        long_text = " ".join(["word"] * 1000)
        suggestions = ["Add information about performance testing."]
        new_text = long_text + " Performance testing is crucial for system reliability."
        
        result = analyze_suggestion_implementation(suggestions, long_text, new_text)
        assert result["suggestions_implemented"] == suggestions

    def test_no_text_change(self):
        """Test when text doesn't change."""
        suggestions = ["Add more details.", "Include examples."]
        text = "Same text."
        
        result = analyze_suggestion_implementation(suggestions, text, text)
        assert result["suggestions_implemented"] == []
        assert result["suggestions_not_implemented"] == suggestions
        assert result["implementation_rate"] == 0.0

    def test_all_pattern_types(self):
        """Test all regex patterns in the function."""
        test_cases = [
            ("Add machine learning concepts.", "concepts", "machine learning"),
            ("Include data science principles.", "principles", "science"),
            ("Provide algorithm explanations.", "explanations", "algorithm"),
            ("Expand on neural networks.", "neural networks", "neural"),
            ("Mention deep learning.", "deep learning", "learning"),
            ("Discuss reinforcement learning.", "reinforcement learning", "reinforcement"),
            ("Examples of classification tasks.", "classification tasks", "classification"),
        ]
        
        for suggestion, expected_phrase, key_word in test_cases:
            old_text = "AI is fascinating."
            new_text = f"AI is fascinating. {key_word.capitalize()} is an important concept."
            
            result = analyze_suggestion_implementation([suggestion], old_text, new_text)
            assert result["implementation_count"] == 1, f"Failed for pattern: {suggestion}"

    def test_multiple_matches_same_pattern(self):
        """Test when a pattern matches multiple times in suggestion."""
        suggestions = ["Add examples of A, add examples of B, add examples of C."]
        old_text = "Initial content."
        new_text = "Initial content with examples of A, B, and C included."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        assert result["suggestions_implemented"] == suggestions

    def test_edge_case_empty_texts(self):
        """Test with empty texts."""
        suggestions = ["Add content."]
        
        # Empty old text
        result = analyze_suggestion_implementation(suggestions, "", "New content added.")
        assert result["implementation_count"] == 1
        
        # Empty new text
        result = analyze_suggestion_implementation(suggestions, "Old content.", "")
        assert result["implementation_count"] == 0
        
        # Both empty
        result = analyze_suggestion_implementation(suggestions, "", "")
        assert result["implementation_count"] == 0

    def test_suggestion_with_numbers(self):
        """Test suggestions containing numbers."""
        suggestions = ["Add 5 examples of best practices."]
        old_text = "Best practices are important."
        new_text = "Best practices are important. Examples include code reviews, testing, documentation, monitoring, and deployment automation."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        assert result["suggestions_implemented"] == suggestions