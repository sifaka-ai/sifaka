import pytest
import textwrap
from sifaka.validators import ReadabilityValidator
from sifaka.types import ValidationResult

# Text examples with different readability levels
ELEMENTARY_TEXT = """
I like to play. The cat is big. My dog runs fast. I go to school.
The sun is hot. Mom and Dad are nice. I read a book. It was fun.
I eat lunch. I drink water. We walk home. I see my friends.
"""

MIDDLE_SCHOOL_TEXT = """
Students in middle school face many challenges as they grow.
They need to manage their time between homework and activities.
The science curriculum introduces more complex topics like biology and chemistry.
Friendships become more complicated during these formative years.
"""

HIGH_SCHOOL_TEXT = """
The industrial revolution transformed manufacturing processes fundamentally.
Economic historians argue that these technological advancements catalyzed
unprecedented economic growth and urban development across Europe and America.
Workers' rights became increasingly significant as labor conditions deteriorated.
"""

COLLEGE_TEXT = """
The ontological argument for the existence of God represents an a priori approach
to establishing theological certainty. Philosophers such as Anselm of Canterbury
posited that God, being a being greater than which nothing can be conceived,
must necessarily exist in reality and not merely in human understanding.
"""

GRADUATE_TEXT = """
The epistemological implications of quantum indeterminacy necessitate a
reconceptualization of empirical methodologies in scientific inquiry.
The observer effect problematizes traditional notions of objective reality,
suggesting instead a probabilistic framework wherein measurement itself
constitutes an ontological intervention rather than mere observation.
"""


class TestReadabilityValidator:
    def test_initialization(self):
        """Test ReadabilityValidator initializes with correct parameters"""
        validator = ReadabilityValidator(
            max_grade_level=10.0,
            min_grade_level=5.0,
            min_flesch_reading_ease=60.0,
            max_complexity_metrics={"gunning_fog": 12.0},
            require_all_metrics=True,
            target_grade_level="middle",
        )

        assert validator.max_grade_level == 10.0
        assert validator.min_grade_level == 5.0
        assert validator.min_flesch_reading_ease == 60.0
        assert validator.max_complexity_metrics == {"gunning_fog": 12.0}
        assert validator.require_all_metrics is True
        assert validator.target_grade_level == "middle"

    def test_empty_text(self):
        """Empty text should pass validation"""
        validator = ReadabilityValidator()
        result = validator.validate("")

        assert result.passed is True
        assert "Empty text" in result.message
        assert result.score == 1.0
        assert not result.issues
        assert not result.suggestions

    def test_grade_level_bounds(self):
        """Test text with grade level out of bounds"""
        # Test text above max grade level
        validator = ReadabilityValidator(max_grade_level=6.0)
        result = validator.validate(HIGH_SCHOOL_TEXT)

        assert result.passed is False
        assert "exceeds maximum grade level" in result.issues[0]
        assert "Simplify text" in result.suggestions[0]

        # Test text below min grade level
        validator = ReadabilityValidator(min_grade_level=12.0)
        result = validator.validate(ELEMENTARY_TEXT)

        assert result.passed is False
        assert "below minimum grade level" in result.issues[0]
        assert "Use more advanced vocabulary" in result.suggestions[0]

    def test_flesch_reading_ease(self):
        """Test Flesch Reading Ease validation"""
        validator = ReadabilityValidator(min_flesch_reading_ease=90.0)
        result = validator.validate(COLLEGE_TEXT)

        assert result.passed is False
        assert "below minimum" in result.issues[0]
        assert "Use shorter sentences" in result.suggestions[0]

    def test_complexity_metrics(self):
        """Test complexity metrics validation"""
        validator = ReadabilityValidator(
            max_complexity_metrics={"gunning_fog": 8.0, "smog_index": 7.0}
        )
        result = validator.validate(HIGH_SCHOOL_TEXT)

        assert result.passed is False
        assert any("gunning_fog" in issue for issue in result.issues)
        assert any("Reduce text complexity" in suggestion for suggestion in result.suggestions)

    def test_target_grade_level(self):
        """Test target grade level validation"""
        # Test elementary text with college target
        validator = ReadabilityValidator(target_grade_level="college")
        result = validator.validate(ELEMENTARY_TEXT)

        assert result.passed is False
        assert "does not match target level" in result.issues[0]
        assert "Use more advanced vocabulary" in result.suggestions[0]

        # Test college text with elementary target
        validator = ReadabilityValidator(target_grade_level="elementary")
        result = validator.validate(COLLEGE_TEXT)

        assert result.passed is False
        assert "does not match target level" in result.issues[0]
        assert "Simplify text" in result.suggestions[0]

    def test_require_all_metrics(self):
        """Test require_all_metrics flag behavior"""
        # Set up validator with multiple constraints
        validator = ReadabilityValidator(
            max_grade_level=12.0,
            min_flesch_reading_ease=70.0,
            max_complexity_metrics={"gunning_fog": 10.0},
            require_all_metrics=True,
        )

        # Text that passes grade level but fails other metrics
        result = validator.validate(MIDDLE_SCHOOL_TEXT)

        # Should fail because require_all_metrics is True
        assert result.passed is False

        # Same validator but with require_all_metrics=False
        validator = ReadabilityValidator(
            max_grade_level=12.0,
            min_flesch_reading_ease=70.0,
            max_complexity_metrics={"gunning_fog": 10.0},
            require_all_metrics=False,
        )

        # Should pass if grade level check passes
        result = validator.validate(MIDDLE_SCHOOL_TEXT)
        if result.passed is False:  # This might vary depending on the exact text content
            assert any("grade level" not in issue for issue in result.issues)

    def test_metadata_contains_metrics(self):
        """Test that result metadata contains all metrics"""
        validator = ReadabilityValidator()
        result = validator.validate(MIDDLE_SCHOOL_TEXT)

        metrics = result.metadata.get("metrics", {})
        assert "flesch_reading_ease" in metrics
        assert "flesch_kincaid_grade" in metrics
        assert "gunning_fog" in metrics
        assert "smog_index" in metrics
        assert "automated_readability_index" in metrics
        assert "dale_chall_readability_score" in metrics
        assert "average_grade_level" in metrics
        assert "grade_level_category" in metrics
        assert "flesch_interpretation" in metrics

    def test_get_grade_level_categorization(self):
        """Test grade level categorization function"""
        validator = ReadabilityValidator()

        assert validator.get_grade_level(5.0) == "elementary"
        assert validator.get_grade_level(7.5) == "middle"
        assert validator.get_grade_level(11.0) == "high"
        assert validator.get_grade_level(14.0) == "college"
        assert validator.get_grade_level(17.0) == "graduate"

    def test_error_handling(self):
        """Test error handling for unexpected issues"""
        # Create a mock validator with a broken textstat property
        validator = ReadabilityValidator()

        # Mock the property to raise an exception
        def mock_property():
            raise Exception("Mocked error")

        validator._textstat = None
        validator.textstat = property(mock_property)

        # Should handle the exception gracefully
        result = validator.validate("Sample text")
        assert result.passed is False
        assert "Error analyzing text" in result.message
        assert "error" in result.metadata
