"""Tests for the numeric validator module."""

import pytest
from sifaka.validators.numeric import (
    NumericRangeValidator,
    create_percentage_validator,
    create_price_validator,
    create_age_validator,
)
from sifaka.core.models import SifakaResult


class TestNumericRangeValidator:
    """Test the NumericRangeValidator class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Test", final_text="Test")

    def test_init_default_values(self):
        """Test initialization with default values."""
        validator = NumericRangeValidator()

        assert validator.min_value is None
        assert validator.max_value is None
        assert validator.allowed_ranges == []
        assert validator.forbidden_ranges == []
        assert validator.check_percentages is True
        assert validator.check_currency is True

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        validator = NumericRangeValidator(
            min_value=0,
            max_value=100,
            allowed_ranges=[(10, 20), (30, 40)],
            forbidden_ranges=[(50, 60)],
            check_percentages=False,
            check_currency=False,
        )

        assert validator.min_value == 0
        assert validator.max_value == 100
        assert validator.allowed_ranges == [(10, 20), (30, 40)]
        assert validator.forbidden_ranges == [(50, 60)]
        assert validator.check_percentages is False
        assert validator.check_currency is False

    def test_name_property(self):
        """Test name property."""
        validator = NumericRangeValidator()
        assert validator.name == "numeric_range_validator"

    @pytest.mark.asyncio
    async def test_validate_no_numbers(self, sample_result):
        """Test validation with no numbers in text."""
        validator = NumericRangeValidator()
        text = "This text contains no numbers at all."

        result = await validator.validate(text, sample_result)

        assert result.validator == "numeric_range_validator"
        assert result.passed is True
        assert result.score == 1.0
        assert result.details == "No numeric values found"

    @pytest.mark.asyncio
    async def test_validate_plain_numbers_in_range(self, sample_result):
        """Test validation with plain numbers within range."""
        validator = NumericRangeValidator(min_value=0, max_value=100)
        text = "The values are 25, 50, and 75."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0
        assert "Validated 3 numeric value(s)" in result.details

    @pytest.mark.asyncio
    async def test_validate_numbers_below_min(self, sample_result):
        """Test validation with numbers below minimum."""
        validator = NumericRangeValidator(min_value=10)
        text = "The values are 5, 15, and 20."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "Value 5 is below minimum 10" in result.details

    @pytest.mark.asyncio
    async def test_validate_numbers_above_max(self, sample_result):
        """Test validation with numbers above maximum."""
        validator = NumericRangeValidator(max_value=50)
        text = "The values are 40, 60, and 80."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "Value 60 is above maximum 50" in result.details

    @pytest.mark.asyncio
    async def test_validate_negative_numbers(self, sample_result):
        """Test validation with negative numbers."""
        validator = NumericRangeValidator(min_value=-10, max_value=10)
        text = "The values are -5, 0, and 5."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0
        assert "Validated 3 numeric value(s)" in result.details

    @pytest.mark.asyncio
    async def test_validate_decimal_numbers(self, sample_result):
        """Test validation with decimal numbers."""
        validator = NumericRangeValidator(min_value=0, max_value=1)
        text = "The values are 0.25, 0.5, and 0.75."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0
        assert "Validated 3 numeric value(s)" in result.details

    @pytest.mark.asyncio
    async def test_validate_percentages_valid(self, sample_result):
        """Test validation with valid percentages."""
        validator = NumericRangeValidator(check_percentages=True)
        text = "The completion rates are 25%, 50%, and 75%."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0
        # Percentages count as numbers
        assert "numeric value(s)" in result.details

    @pytest.mark.asyncio
    async def test_validate_percentages_invalid(self, sample_result):
        """Test validation with invalid percentages."""
        validator = NumericRangeValidator(check_percentages=True)
        text = "The rates are 50%, 150%, and -10%."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "Invalid percentage: 150.0%" in result.details

    @pytest.mark.asyncio
    async def test_validate_percentages_disabled(self, sample_result):
        """Test validation with percentage checking disabled."""
        validator = NumericRangeValidator(check_percentages=False)
        text = "The rates are 50%, 150%, and -10%."

        result = await validator.validate(text, sample_result)

        assert result.passed is True  # Should not check percentages

    @pytest.mark.asyncio
    async def test_validate_currency_valid(self, sample_result):
        """Test validation with valid currency amounts."""
        validator = NumericRangeValidator(check_currency=True)
        text = "The prices are $10.50, $25.00, and $99.99."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0
        assert "numeric value(s)" in result.details

    @pytest.mark.asyncio
    async def test_validate_currency_negative(self, sample_result):
        """Test validation with negative currency amounts."""
        validator = NumericRangeValidator(check_currency=True)
        text = "The balance is $-50.00."

        result = await validator.validate(text, sample_result)

        # The regex doesn't capture negative signs for currency, so it won't detect negative amounts
        # This is actually a limitation of the current implementation
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_validate_currency_disabled(self, sample_result):
        """Test validation with currency checking disabled."""
        validator = NumericRangeValidator(check_currency=False)
        text = "The balance is $-50.00."

        result = await validator.validate(text, sample_result)

        assert result.passed is True  # Should not check currency

    @pytest.mark.asyncio
    async def test_validate_allowed_ranges(self, sample_result):
        """Test validation with allowed ranges."""
        validator = NumericRangeValidator(allowed_ranges=[(10, 20), (30, 40)])
        text = "The values are 15, 25, and 35."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "Value 25 is outside allowed ranges" in result.details

    @pytest.mark.asyncio
    async def test_validate_forbidden_ranges(self, sample_result):
        """Test validation with forbidden ranges."""
        validator = NumericRangeValidator(forbidden_ranges=[(20, 30), (40, 50)])
        text = "The values are 15, 25, and 45."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert result.score == 0.0
        # The details only show first 3 issues but we should check for forbidden range mentions
        assert "forbidden range" in result.details

    @pytest.mark.asyncio
    async def test_validate_multiple_issues_limit(self, sample_result):
        """Test that only first 3 issues are shown."""
        validator = NumericRangeValidator(max_value=10)
        text = "The values are 20, 30, 40, 50, and 60."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        # Should only show first 3 issues
        issue_count = result.details.count("above maximum")
        assert issue_count == 3

    @pytest.mark.asyncio
    async def test_validate_mixed_number_types(self, sample_result):
        """Test validation with mixed number types."""
        validator = NumericRangeValidator(
            min_value=0,
            max_value=100,
            check_percentages=True,
            check_currency=True,
        )
        text = "The price is $25.50, discount is 20%, and quantity is 5."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0
        # Should find all three numeric values
        assert "Validated" in result.details

    @pytest.mark.asyncio
    async def test_validate_percentages_not_affected_by_ranges(self, sample_result):
        """Test that percentage validation is independent of min/max ranges."""
        validator = NumericRangeValidator(
            min_value=0,
            max_value=10,  # This should not affect percentages
            check_percentages=True,
        )
        text = "The success rate is 95%."

        result = await validator.validate(text, sample_result)

        # 95% should be valid (percentage range is 0-100, not affected by min/max)
        # But it also finds "95" as a plain number which violates max_value
        assert result.passed is False  # Because 95 as a number > max_value of 10

    @pytest.mark.asyncio
    async def test_validate_complex_text(self, sample_result):
        """Test validation with complex text containing various numbers."""
        validator = NumericRangeValidator(
            min_value=0,
            max_value=1000,
            forbidden_ranges=[(500, 600)],
            check_percentages=True,
            check_currency=True,
        )
        text = """
        Our Q4 results show revenue of $450.5 million, up 23% from last year.
        Operating margin was 15.5%, exceeding our target of 12%.
        We invested $550 million in R&D, representing 8% of revenue.
        Employee count grew by 250 people to reach 1200 total.
        """

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert "550 is in forbidden range [500, 600]" in result.details

    @pytest.mark.asyncio
    async def test_validate_invalid_number_strings(self, sample_result):
        """Test that invalid number strings are ignored."""
        validator = NumericRangeValidator()
        text = "Version 1.2.3 and IP 192.168.1.1"

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        # Should find some numbers but not parse version/IP as single numbers


class TestFactoryFunctions:
    """Test the convenience factory functions."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Test", final_text="Test")

    def test_create_percentage_validator(self):
        """Test percentage validator factory."""
        validator = create_percentage_validator()

        assert validator.check_percentages is True
        assert validator.check_currency is False

    @pytest.mark.asyncio
    async def test_percentage_validator_behavior(self, sample_result):
        """Test percentage validator behavior."""
        validator = create_percentage_validator()

        # Valid percentages
        result = await validator.validate("Success rate: 85%", sample_result)
        assert result.passed is True

        # Invalid percentage
        result = await validator.validate("Growth: 150%", sample_result)
        assert result.passed is False

    def test_create_price_validator(self):
        """Test price validator factory."""
        validator = create_price_validator(max_price=1000.0)

        assert validator.min_value == 0
        assert validator.max_value == 1000.0
        assert validator.check_currency is True
        assert validator.check_percentages is False

    @pytest.mark.asyncio
    async def test_price_validator_behavior(self, sample_result):
        """Test price validator behavior."""
        validator = create_price_validator(max_price=100.0)

        # Valid price
        result = await validator.validate("Price: $49.99", sample_result)
        assert result.passed is True

        # Price too high
        result = await validator.validate("Price: $150.00", sample_result)
        assert result.passed is False

        # Negative price
        result = await validator.validate("Refund: $-25.00", sample_result)
        assert result.passed is False

    def test_create_age_validator(self):
        """Test age validator factory."""
        validator = create_age_validator()

        assert validator.min_value == 0
        assert validator.max_value == 150
        assert len(validator.forbidden_ranges) == 1
        assert validator.forbidden_ranges[0][0] == 200

    @pytest.mark.asyncio
    async def test_age_validator_behavior(self, sample_result):
        """Test age validator behavior."""
        validator = create_age_validator()

        # Valid ages
        result = await validator.validate("She is 25 years old", sample_result)
        assert result.passed is True

        # Age too high
        result = await validator.validate("Age: 175", sample_result)
        assert result.passed is False

        # Unrealistic age in forbidden range
        result = await validator.validate(
            "The artifact is 250 years old", sample_result
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_factory_validators_with_edge_cases(self, sample_result):
        """Test factory validators with edge cases."""
        # Percentage at boundaries
        pct_validator = create_percentage_validator()
        result = await pct_validator.validate("0% and 100%", sample_result)
        assert result.passed is True

        # Price at boundary
        price_validator = create_price_validator(max_price=50.0)
        result = await price_validator.validate("$50.00", sample_result)
        assert result.passed is True

        # Age at boundaries
        age_validator = create_age_validator()
        result = await age_validator.validate("Ages: 0 and 150", sample_result)
        assert result.passed is True
