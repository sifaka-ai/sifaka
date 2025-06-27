"""Tests for the confidence calculator in sifaka.critics.core.confidence."""

from sifaka.critics.core.confidence import ConfidenceCalculator


class TestConfidenceCalculator:
    """Test the ConfidenceCalculator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        calc = ConfidenceCalculator()
        assert calc.base_confidence == 0.7

    def test_initialization_custom_base(self):
        """Test initialization with custom base confidence."""
        calc = ConfidenceCalculator(base_confidence=0.5)
        assert calc.base_confidence == 0.5

    def test_basic_calculation(self):
        """Test basic confidence calculation with neutral inputs."""
        calc = ConfidenceCalculator()
        confidence = calc.calculate(
            feedback="This is basic feedback",
            suggestions=["Add more details"],
            response_length=200,
            metadata=None,
        )
        # Base confidence 0.7, no major adjustments
        assert 0.6 <= confidence <= 0.8

    def test_long_response_bonus(self):
        """Test that long responses get confidence bonus."""
        calc = ConfidenceCalculator(base_confidence=0.5)
        confidence = calc.calculate(
            feedback="Very detailed analysis here",
            suggestions=["Minor improvement"],
            response_length=600,  # > 500
            metadata=None,
        )
        # Should add 0.1 for thorough response
        assert confidence > 0.5

    def test_short_response_penalty(self):
        """Test that short responses get confidence penalty."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        confidence = calc.calculate(
            feedback="Brief feedback",
            suggestions=["Add more"],
            response_length=50,  # < 100
            metadata=None,
        )
        # Should subtract 0.1 for being too brief
        assert confidence < 0.7

    def test_specificity_indicators_boost(self):
        """Test that specific feedback increases confidence."""
        calc = ConfidenceCalculator(base_confidence=0.5)

        # Feedback with many specificity indicators
        feedback = """
        Specifically, the introduction needs work. For example, you could
        add more context. In particular, the second paragraph clearly lacks
        detail. This is precisely what needs improvement, namely the examples.
        """

        confidence = calc.calculate(
            feedback=feedback,
            suggestions=["Add examples"],
            response_length=200,
            metadata=None,
        )

        # Should get significant boost from specificity
        assert confidence > 0.5

    def test_uncertainty_indicators_penalty(self):
        """Test that uncertain feedback decreases confidence."""
        calc = ConfidenceCalculator(base_confidence=0.7)

        # Feedback with many uncertainty indicators
        feedback = """
        This might need improvement, but I'm not sure. It seems like it could be
        better. Perhaps adding more detail would help, or possibly restructuring.
        It's somewhat unclear if this approach is right. Maybe try something else.
        """

        confidence = calc.calculate(
            feedback=feedback,
            suggestions=["Maybe improve"],
            response_length=200,
            metadata=None,
        )

        # Should get penalty from uncertainty
        assert confidence < 0.7

    def test_many_suggestions_boost(self):
        """Test that many suggestions increase confidence."""
        calc = ConfidenceCalculator(base_confidence=0.6)
        confidence = calc.calculate(
            feedback="Multiple improvements needed",
            suggestions=[
                "Add introduction",
                "Improve clarity",
                "Add examples",
                "Fix grammar",
                "Restructure conclusion",
            ],  # > 3 suggestions
            response_length=200,
            metadata=None,
        )
        # Should add 0.05 for many specific suggestions
        assert confidence > 0.6

    def test_no_suggestions_penalty(self):
        """Test that no suggestions decrease confidence."""
        calc = ConfidenceCalculator(base_confidence=0.7)
        confidence = calc.calculate(
            feedback="Some issues exist",
            suggestions=[],  # No actionable suggestions
            response_length=200,
            metadata=None,
        )
        # Should subtract 0.1 for no actionable suggestions
        assert confidence < 0.7

    def test_confidence_range_bounds(self):
        """Test that confidence is always between 0.0 and 1.0."""
        calc = ConfidenceCalculator(base_confidence=0.9)

        # Test upper bound
        confidence_high = calc.calculate(
            feedback="Specifically, this is exactly right. Clearly perfect.",
            suggestions=["A", "B", "C", "D", "E"],
            response_length=1000,
            metadata=None,
        )
        assert confidence_high <= 1.0

        # Test lower bound
        calc_low = ConfidenceCalculator(base_confidence=0.1)
        confidence_low = calc_low.calculate(
            feedback="Maybe bad", suggestions=[], response_length=50, metadata=None
        )
        assert confidence_low >= 0.0

    def test_score_specificity_calculation(self):
        """Test the specificity scoring method."""
        calc = ConfidenceCalculator()

        # No specificity indicators
        assert calc._score_specificity("Generic feedback") == 0.0

        # One indicator
        assert calc._score_specificity("Specifically, this needs work") == 1 / 3

        # Multiple indicators
        text = "Specifically, for example, this clearly needs work, particularly here"
        score = calc._score_specificity(text)
        assert score > 0.5

        # Many indicators (should cap at 1.0)
        text_many = """
        Specifically, particularly, exactly, precisely, clearly, definitely,
        certainly, for example, such as, including, namely, in particular
        """
        assert calc._score_specificity(text_many) == 1.0

    def test_score_uncertainty_calculation(self):
        """Test the uncertainty scoring method."""
        calc = ConfidenceCalculator()

        # No uncertainty indicators
        assert calc._score_uncertainty("Definite feedback") == 0.0

        # One indicator
        assert calc._score_uncertainty("This might need work") == 1 / 3

        # Multiple indicators
        text = "This might be wrong, maybe needs work, possibly unclear"
        score = calc._score_uncertainty(text)
        assert score > 0.5

        # Many indicators (should cap at 1.0)
        text_many = """
        Might, maybe, perhaps, possibly, could be, seems, appears, somewhat,
        relatively, fairly, probably, potentially, unclear if, not sure
        """
        assert calc._score_uncertainty(text_many) == 1.0

    def test_case_insensitive_indicators(self):
        """Test that indicator matching is case insensitive."""
        calc = ConfidenceCalculator()

        # Uppercase specificity
        assert calc._score_specificity("SPECIFICALLY, this is CLEARLY wrong") > 0

        # Uppercase uncertainty
        assert calc._score_uncertainty("This MIGHT be wrong, MAYBE") > 0

    def test_metadata_parameter_ignored(self):
        """Test that metadata parameter doesn't affect calculation."""
        calc = ConfidenceCalculator()

        # Same inputs except metadata
        confidence1 = calc.calculate(
            feedback="Test feedback",
            suggestions=["Test"],
            response_length=200,
            metadata=None,
        )

        confidence2 = calc.calculate(
            feedback="Test feedback",
            suggestions=["Test"],
            response_length=200,
            metadata={"key": "value"},
        )

        assert confidence1 == confidence2

    def test_combined_adjustments(self):
        """Test multiple adjustments working together."""
        calc = ConfidenceCalculator(base_confidence=0.5)

        # Best case: long, specific, many suggestions
        confidence_high = calc.calculate(
            feedback="Specifically, this clearly needs these exact improvements",
            suggestions=["A", "B", "C", "D", "E"],
            response_length=600,
            metadata=None,
        )

        # Worst case: short, uncertain, no suggestions
        confidence_low = calc.calculate(
            feedback="Maybe this might need something",
            suggestions=[],
            response_length=50,
            metadata=None,
        )

        assert confidence_high > confidence_low
        assert confidence_high > 0.7  # Multiple positive adjustments
        assert confidence_low < 0.3  # Multiple negative adjustments

    def test_edge_case_response_lengths(self):
        """Test edge cases for response length thresholds."""
        calc = ConfidenceCalculator(base_confidence=0.5)

        # Exactly at thresholds
        confidence_100 = calc.calculate(
            feedback="Test", suggestions=["Test"], response_length=100, metadata=None
        )
        confidence_500 = calc.calculate(
            feedback="Test", suggestions=["Test"], response_length=500, metadata=None
        )

        # Just above/below thresholds
        confidence_99 = calc.calculate(
            feedback="Test", suggestions=["Test"], response_length=99, metadata=None
        )
        confidence_501 = calc.calculate(
            feedback="Test", suggestions=["Test"], response_length=501, metadata=None
        )

        assert confidence_99 < confidence_100  # Penalty applied
        assert confidence_501 > confidence_500  # Bonus applied

    def test_suggestion_count_thresholds(self):
        """Test edge cases for suggestion count thresholds."""
        calc = ConfidenceCalculator(base_confidence=0.5)

        # Different suggestion counts
        confidence_0 = calc.calculate(
            feedback="Test", suggestions=[], response_length=200, metadata=None
        )
        confidence_1 = calc.calculate(
            feedback="Test", suggestions=["A"], response_length=200, metadata=None
        )
        confidence_3 = calc.calculate(
            feedback="Test",
            suggestions=["A", "B", "C"],
            response_length=200,
            metadata=None,
        )
        confidence_4 = calc.calculate(
            feedback="Test",
            suggestions=["A", "B", "C", "D"],
            response_length=200,
            metadata=None,
        )

        assert confidence_0 < confidence_1  # Penalty for no suggestions
        assert confidence_4 > confidence_3  # Bonus for >3 suggestions

    def test_partial_word_matching(self):
        """Test that partial word matches don't count as indicators."""
        calc = ConfidenceCalculator()

        # "clear" is in "clearly" but shouldn't match "unclear"
        score = calc._score_specificity("The requirements are unclear")
        assert score == 0.0  # "clearly" indicator shouldn't match "unclear"

        # But should match when standalone
        score = calc._score_specificity("The requirements are clearly stated")
        assert score > 0.0

    def test_real_world_examples(self):
        """Test with realistic feedback examples."""
        calc = ConfidenceCalculator()

        # High confidence example
        high_conf_feedback = """
        Specifically, the introduction lacks a clear thesis statement. For example,
        the first paragraph jumps directly into technical details without context.
        In particular, readers need to understand the problem being solved. The
        methodology section clearly needs more detail about the experimental setup.
        """

        confidence = calc.calculate(
            feedback=high_conf_feedback,
            suggestions=[
                "Add thesis statement to introduction",
                "Provide problem context before technical details",
                "Expand methodology section with experimental setup details",
                "Include specific examples in results section",
            ],
            response_length=len(high_conf_feedback),
            metadata=None,
        )

        assert confidence > 0.75  # Should be high confidence

        # Low confidence example
        low_conf_feedback = """
        This might be okay, but maybe could use some work. It seems like there
        could be issues, possibly with the structure. Not sure if the approach
        is right. Perhaps try something different?
        """

        confidence = calc.calculate(
            feedback=low_conf_feedback,
            suggestions=[],
            response_length=len(low_conf_feedback),
            metadata=None,
        )

        assert confidence < 0.5  # Should be low confidence
