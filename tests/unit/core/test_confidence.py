"""Tests for confidence calculator."""

from sifaka.critics.core.confidence import ConfidenceCalculator


class TestConfidenceCalculator:
    """Test the ConfidenceCalculator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        calc = ConfidenceCalculator()
        assert calc.base_confidence == 0.7

    def test_initialization_custom(self):
        """Test initialization with custom base confidence."""
        calc = ConfidenceCalculator(base_confidence=0.85)
        assert calc.base_confidence == 0.85

    def test_calculate_base_case(self):
        """Test basic calculation with neutral inputs."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        confidence = calc.calculate(
            feedback="This is some feedback.",
            suggestions=["Do this", "Do that"],
            response_length=200,
        )
        # Should be close to base confidence with minor adjustments
        assert 0.649 <= confidence <= 0.75

    def test_long_response_bonus(self):
        """Test that long responses increase confidence."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        confidence = calc.calculate(
            feedback="This is some feedback.",
            suggestions=["Do this"],
            response_length=600,  # > 500
        )
        # Should get +0.1 bonus
        assert confidence > 0.7

    def test_short_response_penalty(self):
        """Test that short responses decrease confidence."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        confidence = calc.calculate(
            feedback="Brief.",
            suggestions=["Do this"],
            response_length=50,  # < 100
        )
        # Should get -0.1 penalty
        assert confidence < 0.7

    def test_many_suggestions_bonus(self):
        """Test that many suggestions increase confidence."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        confidence = calc.calculate(
            feedback="Feedback",
            suggestions=["One", "Two", "Three", "Four"],  # > 3
            response_length=200,
        )
        # Should get +0.05 bonus
        assert confidence > 0.7

    def test_no_suggestions_penalty(self):
        """Test that no suggestions decrease confidence."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        confidence = calc.calculate(
            feedback="Feedback",
            suggestions=[],
            response_length=200,  # No suggestions
        )
        # Should get -0.1 penalty
        assert confidence < 0.7

    def test_specificity_scoring_high(self):
        """Test high specificity increases confidence."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        feedback = """
        Specifically, you should add more examples. For example, including
        data science applications such as machine learning. Precisely speaking,
        the text clearly needs more concrete examples.
        """
        confidence = calc.calculate(
            feedback=feedback, suggestions=["Add examples"], response_length=200
        )
        # Should get bonus from specificity
        assert confidence > 0.75

    def test_specificity_scoring_multiple_indicators(self):
        """Test all specificity indicators."""
        calc = ConfidenceCalculator()

        # Test each indicator
        indicators = [
            "specifically",
            "particularly",
            "exactly",
            "precisely",
            "clearly",
            "definitely",
            "certainly",
            "for example",
            "such as",
            "including",
            "namely",
            "in particular",
        ]

        for indicator in indicators:
            score = calc._score_specificity(f"The text {indicator} mentions this.")
            assert score > 0

    def test_uncertainty_scoring_high(self):
        """Test high uncertainty decreases confidence."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        feedback = """
        This might be good, but perhaps it could be better. Maybe you should
        possibly consider adding more content. It seems somewhat unclear.
        """
        confidence = calc.calculate(
            feedback=feedback, suggestions=["Maybe add content"], response_length=200
        )
        # Should get penalty from uncertainty
        assert confidence < 0.65

    def test_uncertainty_scoring_multiple_indicators(self):
        """Test all uncertainty indicators."""
        calc = ConfidenceCalculator()

        # Test each indicator
        indicators = [
            "might",
            "maybe",
            "perhaps",
            "possibly",
            "could be",
            "seems",
            "appears",
            "somewhat",
            "relatively",
            "fairly",
            "probably",
            "potentially",
            "unclear if",
            "not sure",
        ]

        for indicator in indicators:
            score = calc._score_uncertainty(f"The text {indicator} is good.")
            assert score > 0

    def test_score_specificity_capping(self):
        """Test that specificity score is capped at 1.0."""
        calc = ConfidenceCalculator()
        # Text with many specificity indicators
        text = " ".join(["specifically", "particularly", "exactly", "precisely"] * 5)
        score = calc._score_specificity(text)
        assert score == 1.0

    def test_score_uncertainty_capping(self):
        """Test that uncertainty score is capped at 1.0."""
        calc = ConfidenceCalculator()
        # Text with many uncertainty indicators
        text = " ".join(["might", "maybe", "perhaps", "possibly"] * 5)
        score = calc._score_uncertainty(text)
        assert score == 1.0

    def test_confidence_bounds_maximum(self):
        """Test that confidence is capped at 1.0."""
        calc = ConfidenceCalculator(base_confidence=0.95)
        confidence = calc.calculate(
            feedback="Specifically and precisely perfect in particular.",
            suggestions=["One", "Two", "Three", "Four", "Five"],
            response_length=1000,
        )
        assert confidence <= 1.0

    def test_confidence_bounds_minimum(self):
        """Test that confidence is floored at 0.0."""
        calc = ConfidenceCalculator(base_confidence=0.1)
        confidence = calc.calculate(
            feedback="Maybe possibly might be unclear if good.",
            suggestions=[],
            response_length=50,
        )
        assert confidence >= 0.0

    def test_case_insensitive_matching(self):
        """Test that indicator matching is case insensitive."""
        calc = ConfidenceCalculator()

        # Test uppercase
        score1 = calc._score_specificity("SPECIFICALLY mentioned")
        assert score1 > 0

        # Test mixed case
        score2 = calc._score_uncertainty("MaYbE this is good")
        assert score2 > 0

    def test_with_metadata(self):
        """Test calculation with metadata (currently unused but shouldn't fail)."""
        calc = ConfidenceCalculator()
        confidence = calc.calculate(
            feedback="Good feedback",
            suggestions=["Suggestion"],
            response_length=200,
            metadata={"extra": "data", "score": 0.8},
        )
        assert 0.0 <= confidence <= 1.0

    def test_edge_case_empty_feedback(self):
        """Test with empty feedback."""
        calc = ConfidenceCalculator()
        confidence = calc.calculate(
            feedback="", suggestions=["Suggestion"], response_length=0
        )
        # Should handle gracefully
        assert 0.0 <= confidence <= 1.0

    def test_realistic_scenario_high_confidence(self):
        """Test realistic high-confidence scenario."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        feedback = """
        Specifically, the introduction needs more context. For example, you should
        include background information about the topic. Additionally, the conclusion
        clearly lacks a strong call-to-action. In particular, consider adding
        specific next steps for the reader.
        """
        suggestions = [
            "Add more context to introduction",
            "Include background information",
            "Strengthen the conclusion",
            "Add specific call-to-action",
            "Include next steps for reader",
        ]
        confidence = calc.calculate(
            feedback=feedback, suggestions=suggestions, response_length=len(feedback)
        )
        # Should have high confidence
        assert confidence > 0.8

    def test_realistic_scenario_low_confidence(self):
        """Test realistic low-confidence scenario."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        feedback = "Maybe okay, possibly could be better somehow."
        suggestions = []
        confidence = calc.calculate(
            feedback=feedback, suggestions=suggestions, response_length=len(feedback)
        )
        # Should have low confidence
        assert confidence < 0.6
