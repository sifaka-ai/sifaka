"""Integration tests for validators with real LLM calls."""

import pytest
from sifaka import improve
from sifaka.validators import (
    LengthValidator,
    FormatValidator,
    ContentValidator,
    CompositeValidator,
)


@pytest.mark.integration
def test_length_validator_integration(api_key, llm_provider, integration_timeout):
    """Test length validator with real improvements."""
    original_text = "This is a very short text that needs to be expanded significantly to meet the minimum length requirements."
    
    # Require longer text
    validator = LengthValidator(min_length=500, max_length=1000)
    
    result = improve(
        original_text,
        validators=[validator],
        max_iterations=3,
        timeout=integration_timeout,
        llm_provider=llm_provider,
        api_key=api_key,
    )
    
    # Check that final text meets length requirements
    assert 500 <= len(result.improved_text) <= 1000
    assert result.improved_text != original_text
    
    # Verify validation was applied
    assert "validation" in result.metadata


@pytest.mark.integration
def test_format_validator_integration(api_key, llm_provider, integration_timeout):
    """Test format validator with structured output requirements."""
    original_text = "Write a summary with three bullet points about AI safety."
    
    # Require bullet point format
    validator = FormatValidator(
        required_format="bullet_points",
        format_spec={"min_points": 3, "max_points": 5}
    )
    
    result = improve(
        original_text,
        validators=[validator],
        max_iterations=3,
        timeout=integration_timeout,
        llm_provider=llm_provider,
        api_key=api_key,
    )
    
    # Check format
    lines = result.improved_text.strip().split('\n')
    bullet_lines = [l for l in lines if l.strip().startswith(('•', '-', '*'))]
    assert 3 <= len(bullet_lines) <= 5


@pytest.mark.integration
def test_content_validator_integration(api_key, llm_provider, integration_timeout):
    """Test content validator for specific requirements."""
    original_text = "Explain machine learning."
    
    # Require specific content elements
    validator = ContentValidator(
        required_elements=["supervised learning", "unsupervised learning", "examples"],
        prohibited_elements=["jargon", "complex mathematics"]
    )
    
    result = improve(
        original_text,
        validators=[validator],
        max_iterations=3,
        timeout=integration_timeout,
        llm_provider=llm_provider,
        api_key=api_key,
    )
    
    # Check content requirements
    lower_text = result.improved_text.lower()
    assert "supervised learning" in lower_text
    assert "unsupervised learning" in lower_text
    assert "example" in lower_text


@pytest.mark.integration
def test_composite_validator_integration(api_key, llm_provider, integration_timeout):
    """Test multiple validators working together."""
    original_text = "Write about Python programming."
    
    # Combine multiple validators
    validators = [
        LengthValidator(min_length=200, max_length=500),
        FormatValidator(required_format="paragraphs", format_spec={"min_paragraphs": 2}),
        ContentValidator(required_elements=["functions", "classes", "modules"]),
    ]
    
    composite = CompositeValidator(validators=validators)
    
    result = improve(
        original_text,
        validators=[composite],
        max_iterations=4,
        timeout=integration_timeout,
        llm_provider=llm_provider,
        api_key=api_key,
    )
    
    # Check all requirements are met
    assert 200 <= len(result.improved_text) <= 500
    paragraphs = [p for p in result.improved_text.split('\n\n') if p.strip()]
    assert len(paragraphs) >= 2
    
    lower_text = result.improved_text.lower()
    assert all(term in lower_text for term in ["function", "class", "module"])


@pytest.mark.integration
def test_validator_rejection_handling(api_key, llm_provider, integration_timeout):
    """Test handling of validator rejections."""
    original_text = "Short."
    
    # Very strict validator that's hard to satisfy
    validator = LengthValidator(min_length=1000, max_length=1001)
    
    result = improve(
        original_text,
        validators=[validator],
        max_iterations=5,  # Give it several attempts
        timeout=integration_timeout,
        llm_provider=llm_provider,
        api_key=api_key,
    )
    
    # Should eventually satisfy the validator or use best attempt
    assert len(result.improved_text) > len(original_text)
    
    # Check that validation attempts were made
    assert result.metadata.get("validation_attempts", 0) > 0