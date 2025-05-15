"""
Tests for the safety content validation rules and validators.
"""

import pytest
from sifaka.rules.content.safety import HarmfulContentConfig


def test_harmful_content_config():
    """Test that HarmfulContentConfig works correctly."""
    # Test with default parameters
    config = HarmfulContentConfig()
    assert config.threshold == 0.0
    assert config.fail_if_any is True
    assert "violence" in config.categories
    assert "hate_speech" in config.categories

    # Test with custom parameters
    custom_config = HarmfulContentConfig(
        categories={"test_category": ["indicator1", "indicator2"]},
        threshold=0.5,
        fail_if_any=False,
    )
    assert custom_config.threshold == 0.5
    assert custom_config.fail_if_any is False
    assert "test_category" in custom_config.categories
    assert custom_config.categories["test_category"] == ["indicator1", "indicator2"]

    # Test validation of empty categories
    with pytest.raises(ValueError):
        HarmfulContentConfig(categories={})

    # Test validation of empty indicators
    with pytest.raises(ValueError):
        HarmfulContentConfig(categories={"empty_category": []})
