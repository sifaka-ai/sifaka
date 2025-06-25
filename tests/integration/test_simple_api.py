"""Integration tests for the simple API."""

import pytest
from sifaka import (
    improve_sync,
    improve_email,
    improve_academic,
    improve_code_docs,
    quick_improve,
    improve_with_length,
    improve_with_tone,
    UseCase,
    LengthValidator,
)


class TestSimpleAPI:
    """Test the simplified API functions."""

    def test_improve_sync_basic(self):
        """Test basic synchronous improvement."""
        text = "AI is important."
        result = improve_sync(text, use_case=UseCase.GENERAL)

        assert isinstance(result, str)
        assert len(result) > len(text)  # Should expand brief text
        assert result != text  # Should be different

    def test_improve_sync_use_cases(self):
        """Test different use cases produce different results."""
        text = "The data shows interesting patterns."

        academic = improve_sync(text, use_case=UseCase.ACADEMIC)
        business = improve_sync(text, use_case=UseCase.BUSINESS)
        creative = improve_sync(text, use_case=UseCase.CREATIVE)

        # Different use cases should produce different results
        assert academic != business
        assert business != creative
        assert academic != creative

    @pytest.mark.asyncio
    async def test_improve_email(self):
        """Test email improvement function."""
        text = "hey can u send me the files asap? thx"
        result = await improve_email(text, max_length=200)

        assert isinstance(result, str)
        assert len(result) <= 200  # Respects length limit
        assert "hey" not in result.lower()  # More professional
        assert len(result) > len(text)  # Expands abbreviated text

    @pytest.mark.asyncio
    async def test_improve_academic(self):
        """Test academic improvement function."""
        text = "Machine learning uses data to make predictions."

        # With fact checking
        with_facts = await improve_academic(text, check_facts=True)

        # Without fact checking (should be faster)
        without_facts = await improve_academic(text, check_facts=False)

        assert isinstance(with_facts, str)
        assert isinstance(without_facts, str)
        assert len(with_facts) > len(text)  # More detailed

    @pytest.mark.asyncio
    async def test_improve_code_docs(self):
        """Test code documentation improvement."""
        text = "This function does stuff"
        result = await improve_code_docs(text)

        assert isinstance(result, str)
        assert len(result) > len(text)
        assert "stuff" not in result  # More specific

    @pytest.mark.asyncio
    async def test_quick_improve(self):
        """Test quick single-pass improvement."""
        text = "The product is good and works well."
        result = await quick_improve(text)

        assert isinstance(result, str)
        assert result != text

    @pytest.mark.asyncio
    async def test_improve_with_length_min(self):
        """Test improvement with minimum length."""
        text = "Good product."
        result = await improve_with_length(text, min_length=100)

        assert isinstance(result, str)
        assert len(result) >= 100
        assert text in result or "product" in result.lower()

    @pytest.mark.asyncio
    async def test_improve_with_length_max(self):
        """Test improvement with maximum length."""
        text = " ".join(["This is a very long text"] * 20)  # ~100 words
        result = await improve_with_length(text, max_length=100)

        assert isinstance(result, str)
        assert len(result) <= 100

    @pytest.mark.asyncio
    async def test_improve_with_length_range(self):
        """Test improvement with length range."""
        text = "AI helps businesses."
        result = await improve_with_length(text, min_length=50, max_length=150)

        assert isinstance(result, str)
        assert 50 <= len(result) <= 150

    @pytest.mark.asyncio
    async def test_improve_with_tone(self):
        """Test improvement with tone requirements."""
        text = "Your service sucks and I want a refund immediately!"

        professional = await improve_with_tone(text, tone="professional")
        friendly = await improve_with_tone(text, tone="friendly")

        assert isinstance(professional, str)
        assert isinstance(friendly, str)
        assert "sucks" not in professional
        assert professional != friendly

    def test_validators_integration(self):
        """Test that validators work with improve_sync."""
        text = "Short."

        # Should pass with no validators
        result1 = improve_sync(text)
        assert isinstance(result1, str)

        # Should expand to meet minimum length
        validator = LengthValidator(min_length=50)
        result2 = improve_sync(text, validators=[validator])
        assert len(result2) >= 50


class TestErrorHandling:
    """Test error handling in simple API."""

    def test_invalid_use_case(self):
        """Test handling of invalid use case."""
        # Should use default for invalid use case
        result = improve_sync("Test text", use_case="invalid_use_case")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_empty_text(self):
        """Test handling of empty text."""
        result = await quick_improve("")
        assert isinstance(result, str)
        assert len(result) > 0  # Should generate something

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test handling of very long text."""
        # Create text that's just under the limit
        long_text = "a" * 40000
        result = await quick_improve(long_text)
        assert isinstance(result, str)
