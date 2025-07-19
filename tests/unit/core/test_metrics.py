"""Tests for the metrics module."""

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

    def test_objective_metrics_basic(self):
        """Test basic objective metrics calculation."""
        suggestions = ["Add examples", "Improve clarity"]
        old_text = "Short text."
        new_text = "This is a much longer text with more details and examples."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        # Backward compatibility fields (always same values now)
        assert result["suggestions_given"] == suggestions
        assert result["suggestions_implemented"] == []
        assert result["suggestions_not_implemented"] == suggestions
        assert result["implementation_rate"] == 0.0
        assert result["implementation_count"] == 0

        # Objective metrics
        assert result["old_text_length"] == len(old_text)
        assert result["new_text_length"] == len(new_text)
        assert result["length_change"] > 0
        assert result["word_count_change"] > 0
        assert result["text_similarity"] < 0.5  # Very different texts

    def test_no_changes_made(self):
        """Test when text remains unchanged."""
        suggestions = ["Add more details", "Include examples"]
        text = "This is the same text."

        result = analyze_suggestion_implementation(suggestions, text, text)

        assert result["suggestions_implemented"] == []
        assert result["suggestions_not_implemented"] == suggestions
        assert result["length_change"] == 0
        assert result["word_count_change"] == 0
        assert result["text_similarity"] == 1.0  # Identical texts

    def test_sentence_and_paragraph_counting(self):
        """Test sentence and paragraph counting."""
        old_text = "First sentence. Second one."
        new_text = """First paragraph with one sentence.

Second paragraph. It has two sentences!

Third paragraph? Yes, with a question."""

        result = analyze_suggestion_implementation([], old_text, new_text)

        assert result["old_sentence_count"] == 2
        assert result["new_sentence_count"] == 5
        assert result["old_paragraph_count"] == 1
        assert result["new_paragraph_count"] == 3

    def test_content_change_detection(self):
        """Test detection of specific content changes."""
        old_text = "Basic information about the topic."
        new_text = """Basic information about the topic. In 2023, studies showed
        75% improvement. As Einstein said, "Imagination is more important
        than knowledge." But what about the future?"""

        result = analyze_suggestion_implementation([], old_text, new_text)

        assert result["numbers_added"] > 0  # "2023", "75%"
        assert result["quotes_added"] > 0  # Einstein quote
        assert result["questions_added"] > 0  # "But what about..."

    def test_text_similarity_calculation(self):
        """Test text similarity metric."""
        # Identical texts
        text1 = "The quick brown fox"
        result1 = analyze_suggestion_implementation([], text1, text1)
        assert result1["text_similarity"] == 1.0

        # Completely different texts
        text2 = "Completely different content here"
        result2 = analyze_suggestion_implementation([], text1, text2)
        assert result2["text_similarity"] == 0.0

        # Partially similar texts
        text3 = "The quick brown cat"
        result3 = analyze_suggestion_implementation([], text1, text3)
        assert 0.5 < result3["text_similarity"] < 1.0

    def test_empty_texts(self):
        """Test with empty old or new text."""
        suggestions = ["Add content"]

        # Empty old text
        result1 = analyze_suggestion_implementation(
            suggestions, "", "New content added"
        )
        assert result1["old_text_length"] == 0
        assert result1["new_text_length"] > 0
        assert result1["length_change_ratio"] == float("inf")

        # Empty new text
        result2 = analyze_suggestion_implementation(suggestions, "Old content", "")
        assert result2["old_text_length"] > 0
        assert result2["new_text_length"] == 0
        assert result2["length_change"] < 0

    def test_suggestion_metrics(self):
        """Test metrics about the suggestions themselves."""
        short_suggestions = ["Fix", "Add", "Change"]
        long_suggestions = [
            "Add comprehensive examples with detailed explanations",
            "Restructure the entire document for better flow",
            "Include statistical data from recent studies",
        ]

        result1 = analyze_suggestion_implementation(short_suggestions, "text", "text")
        result2 = analyze_suggestion_implementation(long_suggestions, "text", "text")

        assert result1["suggestion_count"] == 3
        assert result2["suggestion_count"] == 3
        assert result1["avg_suggestion_length"] < result2["avg_suggestion_length"]

    def test_return_value_structure(self):
        """Test that return value has all expected keys."""
        result = analyze_suggestion_implementation(["Test suggestion"], "old", "new")

        # Backward compatibility fields
        assert "suggestions_given" in result
        assert "suggestions_implemented" in result
        assert "suggestions_not_implemented" in result
        assert "implementation_rate" in result
        assert "implementation_count" in result

        # Objective metric fields
        assert "old_text_length" in result
        assert "new_text_length" in result
        assert "length_change" in result
        assert "length_change_ratio" in result
        assert "old_word_count" in result
        assert "new_word_count" in result
        assert "word_count_change" in result
        assert "word_count_ratio" in result
        assert "old_sentence_count" in result
        assert "new_sentence_count" in result
        assert "old_paragraph_count" in result
        assert "new_paragraph_count" in result
        assert "suggestion_count" in result
        assert "avg_suggestion_length" in result
        assert "text_similarity" in result
        assert "numbers_added" in result
        assert "quotes_added" in result
        assert "questions_added" in result

        # Check types
        assert isinstance(result["suggestions_given"], list)
        assert isinstance(result["suggestions_implemented"], list)
        assert isinstance(result["suggestions_not_implemented"], list)
        assert isinstance(result["implementation_rate"], (int, float))
        assert isinstance(result["implementation_count"], int)

    def test_word_count_with_punctuation(self):
        """Test word counting handles punctuation correctly."""
        old_text = "Hello, world! How are you?"
        new_text = "Hello, world! How are you? I'm fine, thanks."

        result = analyze_suggestion_implementation([], old_text, new_text)

        assert result["old_word_count"] == 5
        assert result["new_word_count"] == 8  # "I'm" counts as one word
        assert result["word_count_change"] == 3

    def test_number_detection_patterns(self):
        """Test various number detection patterns."""
        old_text = "Some text"
        new_text = "In 2024, we saw 45% growth, spent $1,234, and reached 1000 users."

        result = analyze_suggestion_implementation([], old_text, new_text)

        # Should detect: 2024 (year), 45% (percentage), $1,234 (money), 1000 (plain)
        assert result["numbers_added"] >= 4

    def test_question_detection(self):
        """Test question detection in various formats."""
        old_text = "Statement one. Statement two."
        new_text = "Statement one. But why? Statement two. What about this? Really?"

        result = analyze_suggestion_implementation([], old_text, new_text)

        assert result["questions_added"] == 3

    def test_quote_detection(self):
        """Test quote detection with different quote styles."""
        old_text = "Plain text"
        new_text = """He said "hello" and she replied 'hi there'."""

        result = analyze_suggestion_implementation([], old_text, new_text)

        assert result["quotes_added"] == 2
