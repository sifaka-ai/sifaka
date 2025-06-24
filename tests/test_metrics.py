"""Tests for the metrics module."""

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
    
    def test_simple_add_suggestion_implemented(self):
        """Test when an 'add' suggestion is implemented."""
        suggestions = ["Add examples of renewable energy"]
        old_text = "Renewable energy is important."
        new_text = "Renewable energy is important. Examples include solar and wind power."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        assert result["suggestions_implemented"] == suggestions
        assert result["suggestions_not_implemented"] == []
        assert result["implementation_rate"] == 1.0
        assert result["implementation_count"] == 1
    
    def test_simple_add_suggestion_not_implemented(self):
        """Test when an 'add' suggestion is not implemented."""
        suggestions = ["Add specific statistics about costs"]
        old_text = "Renewable energy is important."
        new_text = "Renewable energy is very important."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        assert result["suggestions_implemented"] == []
        assert result["suggestions_not_implemented"] == suggestions
        assert result["implementation_rate"] == 0.0
        assert result["implementation_count"] == 0
    
    def test_multiple_suggestions_partial_implementation(self):
        """Test with multiple suggestions, some implemented."""
        suggestions = [
            "Include statistics about solar power",
            "Provide examples of wind energy",
            "Mention cost savings"
        ]
        old_text = "Renewable energy is the future."
        new_text = "Renewable energy is the future. Solar power has grown 20% annually. Wind turbines are common examples."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        # Should detect statistics and examples were added
        assert len(result["suggestions_implemented"]) == 2
        assert len(result["suggestions_not_implemented"]) == 1
        assert "Mention cost savings" in result["suggestions_not_implemented"]
        assert result["implementation_rate"] == 2/3
        assert result["implementation_count"] == 2
    
    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        suggestions = ["Add information about SOLAR PANELS"]
        old_text = "Energy is important."
        new_text = "Energy is important. Solar panels convert sunlight to electricity."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1
    
    def test_expand_suggestion_pattern(self):
        """Test the 'expand on' pattern."""
        suggestions = ["Expand on the benefits of recycling"]
        old_text = "Recycling is good."
        new_text = "Recycling is good. Benefits include reducing waste, saving energy, and protecting the environment."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1
    
    def test_discuss_suggestion_pattern(self):
        """Test the 'discuss' pattern."""
        suggestions = ["Discuss the impact on climate change"]
        old_text = "Cars produce emissions."
        new_text = "Cars produce emissions. The impact on climate change is significant, contributing to global warming."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1
    
    def test_mention_suggestion_pattern(self):
        """Test the 'mention' pattern."""
        suggestions = ["Mention government policies"]
        old_text = "Clean energy is growing."
        new_text = "Clean energy is growing. Government policies like tax incentives support this growth."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1
    
    def test_provide_suggestion_pattern(self):
        """Test the 'provide' pattern."""
        suggestions = ["Provide concrete data on emissions"]
        old_text = "Transportation causes pollution."
        new_text = "Transportation causes pollution. Concrete data shows emissions of 1.9 billion tons annually."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1
    
    def test_examples_suggestion_pattern(self):
        """Test the 'examples of' pattern."""
        suggestions = ["Include examples of green technology"]
        old_text = "Technology can help the environment."
        new_text = "Technology can help the environment. Green technology examples include electric vehicles and smart grids."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1
    
    def test_complex_suggestion_with_multiple_keywords(self):
        """Test suggestion with multiple important keywords."""
        suggestions = ["Add statistics about renewable energy adoption in Europe"]
        old_text = "Clean energy is growing globally."
        new_text = "Clean energy is growing globally. In Europe, renewable energy adoption reached 40% in 2023, with statistics showing rapid growth."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        # Should detect multiple keywords: statistics, renewable, energy, adoption, Europe
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1
    
    def test_filters_common_words(self):
        """Test that common words are filtered out."""
        suggestions = ["Add the information for the users with the data"]
        old_text = "System provides reports."
        new_text = "System provides reports. Information and data are available to users."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        # Should still detect implementation based on important words
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1
    
    def test_no_pattern_match_but_keywords_present(self):
        """Test when suggestion doesn't match patterns but keywords are added."""
        suggestions = ["Include examples of sustainability practices"]
        old_text = "Companies are changing."
        new_text = "Companies are changing. Sustainability practices are now central to business."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        # Should detect implementation based on keyword presence
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1
    
    def test_punctuation_in_suggestions(self):
        """Test suggestions with various punctuation."""
        suggestions = [
            "Add data (including percentages).",
            "Mention costs, benefits, and risks!",
            "Provide examples: solar, wind, hydro"
        ]
        old_text = "Energy transition is happening."
        new_text = "Energy transition is happening. Data shows 25% growth. Costs are decreasing while benefits increase. Examples include solar panels and wind farms."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        # Should handle punctuation correctly
        assert len(result["suggestions_implemented"]) >= 2
        assert result["implementation_rate"] >= 0.66
    
    def test_word_count_increase_detection(self):
        """Test detection based on word count increase."""
        suggestions = ["Add information about machine learning applications"]
        old_text = "Machine learning is useful."
        new_text = "Machine learning is useful. Machine learning applications include image recognition, natural language processing, and predictive analytics. Machine learning continues to evolve."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        # Should detect implementation due to increased mentions of key terms
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1
    
    def test_unicode_and_special_characters(self):
        """Test with unicode and special characters."""
        suggestions = ["Add information about café's environmental impact"]
        old_text = "The café is popular."
        new_text = "The café is popular. Its environmental impact includes using 100% renewable energy."
        
        result = analyze_suggestion_implementation(suggestions, old_text, new_text)
        
        assert result["suggestions_implemented"] == suggestions
        assert result["implementation_count"] == 1