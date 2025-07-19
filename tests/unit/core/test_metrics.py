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

    def test_single_suggestion_implemented(self):
        """Test with a single suggestion that was implemented."""
        suggestions = ["Add examples of machine learning applications"]
        old_text = "Machine learning is a powerful technology."
        new_text = "Machine learning is a powerful technology. Examples include image recognition, natural language processing, and recommendation systems."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert result["suggestions_given"] == suggestions
        assert len(result["suggestions_implemented"]) == 1
        assert result["suggestions_implemented"][0] == suggestions[0]
        assert result["suggestions_not_implemented"] == []
        assert result["implementation_rate"] == 1.0
        assert result["implementation_count"] == 1

    def test_single_suggestion_not_implemented(self):
        """Test with a single suggestion that was not implemented."""
        suggestions = ["Add examples of quantum computing"]
        old_text = "Machine learning is important."
        new_text = "Machine learning is very important."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert result["suggestions_given"] == suggestions
        assert result["suggestions_implemented"] == []
        assert len(result["suggestions_not_implemented"]) == 1
        assert result["suggestions_not_implemented"][0] == suggestions[0]
        assert result["implementation_rate"] == 0.0
        assert result["implementation_count"] == 0

    def test_multiple_suggestions_mixed(self):
        """Test with multiple suggestions, some implemented."""
        suggestions = [
            "Include statistics about growth",
            "Provide examples of real-world applications",
            "Mention the future prospects",
            "Add a conclusion paragraph",
        ]
        old_text = "AI is transforming industries."
        new_text = """AI is transforming industries. Recent statistics show 50% growth
        in AI adoption. Real-world applications include autonomous vehicles and
        medical diagnosis. The technology continues to evolve rapidly."""

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert result["suggestions_given"] == suggestions
        assert (
            len(result["suggestions_implemented"]) == 2
        )  # statistics and applications
        assert len(result["suggestions_not_implemented"]) == 2  # future and conclusion
        assert result["implementation_rate"] == 0.5
        assert result["implementation_count"] == 2

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        suggestions = ["Add information about Python programming"]
        old_text = "Programming languages are useful."
        new_text = "Programming languages are useful. PYTHON is particularly popular."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert len(result["suggestions_implemented"]) == 1
        assert result["implementation_rate"] == 1.0

    def test_different_action_words(self):
        """Test various action word patterns."""
        suggestions = [
            "Add more details",
            "Include specific examples",
            "Provide clearer explanations",
            "Expand on the methodology",
            "Mention key limitations",
            "Discuss potential applications",
        ]
        old_text = "This is a basic overview."
        new_text = """This is a basic overview with more details. Specific examples
        include A and B. The explanations are now clearer. The methodology has been
        expanded. Key limitations include X and Y. Potential applications are vast."""

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        # All suggestions should be detected as implemented
        assert len(result["suggestions_implemented"]) == 6
        assert result["implementation_rate"] == 1.0

    def test_partial_phrase_implementation(self):
        """Test when only part of a suggested phrase is implemented."""
        suggestions = ["Add examples of neural networks and deep learning"]
        old_text = "AI is advancing."
        new_text = "AI is advancing. Neural networks are one key technology."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        # Should be implemented because "neural" appears
        assert len(result["suggestions_implemented"]) == 1

    def test_word_count_increase_detection(self):
        """Test detection based on word count increase."""
        suggestions = ["Expand on the benefits"]
        old_text = "Technology has benefits."
        new_text = "Technology has many benefits. Benefits include efficiency and cost savings."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        # Should detect implementation due to increased "benefits" count
        assert len(result["suggestions_implemented"]) == 1

    def test_ignore_common_words(self):
        """Test that common words are ignored in matching."""
        suggestions = ["Add the and for with"]
        old_text = "Basic text."
        new_text = "Basic text with some additions."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        # Should not be implemented because only common words
        assert len(result["suggestions_implemented"]) == 0

    def test_multiple_pattern_matches(self):
        """Test when a suggestion matches multiple patterns."""
        suggestions = ["Add and include examples of best practices"]
        old_text = "Follow guidelines."
        new_text = "Follow guidelines. Best practices include code review and testing."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert len(result["suggestions_implemented"]) == 1

    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        suggestions = ["Mention the cost ($100)"]
        old_text = "Product overview."
        new_text = "Product overview. The cost is $100."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert len(result["suggestions_implemented"]) == 1

    def test_no_changes_made(self):
        """Test when text remains unchanged."""
        suggestions = ["Add more details", "Include examples"]
        text = "This is the same text."

        result = analyze_suggestion_implementation(suggestions, text, text)

        assert result["suggestions_implemented"] == []
        assert len(result["suggestions_not_implemented"]) == 2
        assert result["implementation_rate"] == 0.0

    def test_all_suggestions_implemented(self):
        """Test when all suggestions are implemented."""
        suggestions = [
            "Provide an introduction",
            "Add technical details",
            "Include a summary",
        ]
        old_text = "Main content."
        new_text = """Introduction: This document covers important topics.
        Main content with technical details about implementation.
        Summary: Key points were discussed."""

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert len(result["suggestions_implemented"]) == 3
        assert result["suggestions_not_implemented"] == []
        assert result["implementation_rate"] == 1.0

    def test_suggestion_with_punctuation(self):
        """Test suggestions containing various punctuation."""
        suggestions = [
            "Add more details, especially about performance.",
            "Include examples; they help understanding!",
        ]
        old_text = "Basic description."
        new_text = (
            "Basic description with details about performance. Examples: A, B, C."
        )

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert (
            len(result["suggestions_implemented"]) == 1
        )  # Only first suggestion matches

    def test_long_suggestions(self):
        """Test with longer, more complex suggestions."""
        suggestions = [
            "Expand on the theoretical background by discussing foundational concepts and their historical development"
        ]
        old_text = "Theory section."
        new_text = """Theory section. The theoretical background includes foundational
        concepts from the 1950s. Historical development shows progression from simple
        models to complex systems."""

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert len(result["suggestions_implemented"]) == 1

    def test_similar_words_different_forms(self):
        """Test detection of similar words in different forms."""
        suggestions = ["Discuss optimization techniques"]
        old_text = "Algorithm overview."
        new_text = "Algorithm overview. We optimize using gradient descent."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        # Should not detect "optimize" as implementation of "optimization"
        # because we do exact word matching
        assert len(result["suggestions_implemented"]) == 0

    def test_empty_texts(self):
        """Test with empty old or new text."""
        suggestions = ["Add content"]

        # Empty old text
        result1 = analyze_suggestion_implementation(
            suggestions, "", "New content added"
        )
        assert len(result1["suggestions_implemented"]) == 1

        # Empty new text
        result2 = analyze_suggestion_implementation(suggestions, "Old content", "")
        assert len(result2["suggestions_implemented"]) == 0

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        suggestions = ["Add    information   about   spacing"]
        old_text = "Text."
        new_text = "Text with information about spacing and formatting."

        result = analyze_suggestion_implementation(suggestions, old_text, new_text)

        assert len(result["suggestions_implemented"]) == 1

    def test_return_value_structure(self):
        """Test that return value has all expected keys."""
        result = analyze_suggestion_implementation(["Test suggestion"], "old", "new")

        assert isinstance(result, dict)
        assert "suggestions_given" in result
        assert "suggestions_implemented" in result
        assert "suggestions_not_implemented" in result
        assert "implementation_rate" in result
        assert "implementation_count" in result

        # Check types
        assert isinstance(result["suggestions_given"], list)
        assert isinstance(result["suggestions_implemented"], list)
        assert isinstance(result["suggestions_not_implemented"], list)
        assert isinstance(result["implementation_rate"], (int, float))
        assert isinstance(result["implementation_count"], int)
