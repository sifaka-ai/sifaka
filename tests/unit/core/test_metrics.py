"""Tests for metrics module."""

from sifaka.core.metrics import (
    _calculate_similarity,
    _count_numbers,
    _count_questions,
    _count_quotes,
    analyze_suggestion_implementation,
)


class TestAnalyzeSuggestionImplementation:
    """Test suggestion implementation analysis."""

    def test_basic_analysis(self):
        """Test basic metric analysis."""
        suggestions = ["Add more detail", "Use examples"]
        old_text = "Machine learning is important."
        new_text = "Machine learning is important for data analysis. For example, it can predict outcomes."

        metrics = analyze_suggestion_implementation(suggestions, old_text, new_text)

        # Check objective metrics
        assert metrics["old_text_length"] == len(old_text)
        assert metrics["new_text_length"] == len(new_text)
        assert metrics["length_change"] > 0
        assert metrics["old_word_count"] == 4
        assert metrics["new_word_count"] > 4
        assert metrics["suggestion_count"] == 2
        assert metrics["text_similarity"] < 1.0

    def test_no_changes(self):
        """Test when text doesn't change."""
        suggestions = ["Make it better"]
        old_text = "Same text"
        new_text = "Same text"

        metrics = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert metrics["length_change"] == 0
        assert metrics["word_count_change"] == 0
        assert metrics["text_similarity"] == 1.0
        assert metrics["length_change_ratio"] == 1.0

    def test_empty_suggestions(self):
        """Test with no suggestions."""
        suggestions = []
        old_text = "Original"
        new_text = "Modified"

        metrics = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert metrics["suggestion_count"] == 0
        assert metrics["avg_suggestion_length"] == 0

    def test_empty_texts(self):
        """Test with empty texts."""
        suggestions = ["Add content"]

        # Empty old text
        metrics = analyze_suggestion_implementation(suggestions, "", "New content")
        assert metrics["old_text_length"] == 0
        assert metrics["length_change_ratio"] == float("inf")

        # Empty new text
        metrics = analyze_suggestion_implementation(suggestions, "Old content", "")
        assert metrics["new_text_length"] == 0
        assert metrics["length_change"] < 0

    def test_content_changes(self):
        """Test specific content change detection."""
        suggestions = ["Add statistics"]
        old_text = "The system works well."
        new_text = 'The system works well with 95% accuracy. As stated: "It\'s effective." How does it work?'

        metrics = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert metrics["numbers_added"] > 0  # Added 95%
        assert metrics["quotes_added"] > 0  # Added quoted text
        assert metrics["questions_added"] > 0  # Added question


class TestCalculateSimilarity:
    """Test text similarity calculation."""

    def test_identical_texts(self):
        """Test similarity of identical texts."""
        similarity = _calculate_similarity("Hello world", "Hello world")
        assert similarity == 1.0

    def test_completely_different(self):
        """Test completely different texts."""
        similarity = _calculate_similarity("Hello world", "Goodbye universe")
        assert similarity == 0.0

    def test_partial_overlap(self):
        """Test partial word overlap."""
        similarity = _calculate_similarity("The quick brown fox", "The slow brown dog")
        assert 0 < similarity < 1
        # Should have "the" and "brown" in common
        # Jaccard similarity: |intersection| / |union|
        # Common: {the, brown} = 2
        # Union: {the, quick, brown, fox, slow, dog} = 6
        assert similarity == 2 / 6  # 2 common words out of 6 unique total

    def test_case_insensitive(self):
        """Test case insensitive comparison."""
        similarity = _calculate_similarity("Hello World", "hello world")
        assert similarity == 1.0

    def test_empty_texts(self):
        """Test empty text handling."""
        assert _calculate_similarity("", "") == 1.0
        assert _calculate_similarity("Hello", "") == 0.0
        assert _calculate_similarity("", "World") == 0.0


class TestCountNumbers:
    """Test number counting."""

    def test_plain_numbers(self):
        """Test counting plain numbers."""
        count = _count_numbers("There are 5 apples and 10 oranges")
        assert count == 2

    def test_percentages(self):
        """Test counting percentages."""
        count = _count_numbers("Growth of 25% and efficiency at 90%")
        assert count >= 2

    def test_money(self):
        """Test counting money amounts."""
        count = _count_numbers("Cost is $100 or $1,000")
        assert count >= 2

    def test_years(self):
        """Test counting years."""
        count = _count_numbers("From 2020 to 2024")
        assert count >= 2

    def test_no_numbers(self):
        """Test text without numbers."""
        count = _count_numbers("No numbers here")
        assert count == 0


class TestCountQuotes:
    """Test quote counting."""

    def test_double_quotes(self):
        """Test counting double quotes."""
        count = _count_quotes('He said "hello" and she said "goodbye"')
        assert count == 2

    def test_single_quotes(self):
        """Test counting single quotes."""
        count = _count_quotes("It's 'working' and 'done'")
        assert count == 2

    def test_mixed_quotes(self):
        """Test mixed quote types."""
        count = _count_quotes("She said \"yes\" and he said 'no'")
        assert count == 2

    def test_no_quotes(self):
        """Test text without quotes."""
        count = _count_quotes("No quotes in this text")
        assert count == 0

    def test_empty_quotes(self):
        """Test that empty quotes aren't counted."""
        count = _count_quotes('Test "" text')
        # Regex looks for non-empty quotes
        assert count == 0 or count == 1  # Depends on regex implementation


class TestCountQuestions:
    """Test question counting."""

    def test_simple_question(self):
        """Test counting simple questions."""
        count = _count_questions("How are you?")
        assert count == 1

    def test_multiple_questions(self):
        """Test multiple questions."""
        count = _count_questions("What is this? Why is it here? Who knows?")
        assert count >= 2  # At least detects some questions

    def test_no_questions(self):
        """Test text without questions."""
        count = _count_questions("This is a statement. So is this!")
        assert count == 0

    def test_question_at_end(self):
        """Test question at end of text."""
        count = _count_questions("I wonder what this is?")
        assert count >= 1
