"""Tests for reflector rules."""

import pytest

from sifaka.reflector import (
    DEFAULT_PATTERN_CONFIGS,
    DEFAULT_REFLECTION_CONFIGS,
    PatternConfig,
    PatternRule,
    ReflectionConfig,
    ReflectionRule,
    create_pattern_rule,
    create_reflection_rule,
)


@pytest.fixture
def reflection_config() -> ReflectionConfig:
    """Create a test reflection configuration."""
    return ReflectionConfig(
        mirror_mode="horizontal",
        preserve_whitespace=True,
        preserve_case=True,
        ignore_punctuation=False,
        symmetry_threshold=1.0,
        cache_size=100,
        priority=1,
        cost=1.0,
    )

@pytest.fixture
def pattern_config() -> PatternConfig:
    """Create a test pattern configuration."""
    return PatternConfig(
        pattern_type="repeat",
        pattern_length=2,
        custom_pattern=None,
        case_sensitive=True,
        allow_overlap=False,
        cache_size=100,
        priority=1,
        cost=1.0,
    )

def test_reflection_config_validation():
    """Test reflection configuration validation."""
    # Valid configurations
    config = ReflectionConfig(mirror_mode="horizontal")
    assert config.mirror_mode == "horizontal"
    assert config.preserve_whitespace is True

    config = ReflectionConfig(mirror_mode="vertical")
    assert config.mirror_mode == "vertical"

    config = ReflectionConfig(mirror_mode="both")
    assert config.mirror_mode == "both"

    # Invalid mirror mode
    with pytest.raises(ValueError, match="Mirror mode must be one of"):
        ReflectionConfig(mirror_mode="invalid")

    # Invalid symmetry threshold
    with pytest.raises(ValueError, match="Symmetry threshold must be between"):
        ReflectionConfig(symmetry_threshold=1.5)

    # Invalid cache size
    with pytest.raises(ValueError, match="Cache size must be non-negative"):
        ReflectionConfig(cache_size=-1)

    # Invalid priority
    with pytest.raises(ValueError, match="Priority must be non-negative"):
        ReflectionConfig(priority=-1)

    # Invalid cost
    with pytest.raises(ValueError, match="Cost must be non-negative"):
        ReflectionConfig(cost=-1)

def test_pattern_config_validation():
    """Test pattern configuration validation."""
    # Valid configurations
    config = PatternConfig(pattern_type="repeat")
    assert config.pattern_type == "repeat"
    assert config.pattern_length == 2

    config = PatternConfig(pattern_type="alternate")
    assert config.pattern_type == "alternate"

    config = PatternConfig(pattern_type="custom", custom_pattern="abc")
    assert config.pattern_type == "custom"
    assert config.custom_pattern == "abc"

    # Invalid pattern type
    with pytest.raises(ValueError, match="Pattern type must be one of"):
        PatternConfig(pattern_type="invalid")

    # Invalid pattern length
    with pytest.raises(ValueError, match="Pattern length must be positive"):
        PatternConfig(pattern_length=0)

    # Missing custom pattern
    with pytest.raises(ValueError, match="Custom pattern must be provided"):
        PatternConfig(pattern_type="custom")

    # Invalid cache size
    with pytest.raises(ValueError, match="Cache size must be non-negative"):
        PatternConfig(cache_size=-1)

    # Invalid priority
    with pytest.raises(ValueError, match="Priority must be non-negative"):
        PatternConfig(priority=-1)

    # Invalid cost
    with pytest.raises(ValueError, match="Cost must be non-negative"):
        PatternConfig(cost=-1)

def test_horizontal_reflection_validation():
    """Test horizontal reflection validation."""
    rule = create_reflection_rule(
        name="test_horizontal",
        description="Test horizontal reflection",
        mirror_mode="horizontal",
        preserve_whitespace=True,
        preserve_case=True,
    )

    # Perfect symmetry
    result = rule._validate_impl("radar")
    assert result.passed
    assert result.metadata["symmetry_score"] == 1.0

    # Imperfect symmetry
    result = rule._validate_impl("almost")
    assert not result.passed
    assert result.metadata["symmetry_score"] < 1.0

    # Multi-line symmetry
    result = rule._validate_impl("level\nracecar\nlevel")
    assert result.passed
    assert result.metadata["symmetry_score"] == 1.0

    # Case sensitivity
    rule_case_insensitive = create_reflection_rule(
        name="test_case",
        description="Test case insensitive",
        mirror_mode="horizontal",
        preserve_case=False,
    )
    result = rule_case_insensitive._validate_impl("RaDaR")
    assert result.passed
    assert result.metadata["symmetry_score"] == 1.0

def test_vertical_reflection_validation():
    """Test vertical reflection validation."""
    rule = create_reflection_rule(
        name="test_vertical",
        description="Test vertical reflection",
        mirror_mode="vertical",
    )

    # Perfect vertical symmetry
    result = rule._validate_impl("abc\ndef\nabc")
    assert result.passed
    assert result.metadata["symmetry_score"] == 1.0

    # Imperfect vertical symmetry
    result = rule._validate_impl("abc\ndef\nghi")
    assert not result.passed
    assert result.metadata["symmetry_score"] < 1.0

    # Single line (should pass)
    result = rule._validate_impl("single")
    assert result.passed
    assert result.metadata["symmetry_score"] == 1.0

def test_both_reflection_validation():
    """Test both horizontal and vertical reflection validation."""
    rule = create_reflection_rule(
        name="test_both",
        description="Test both reflections",
        mirror_mode="both",
    )

    # Perfect symmetry both ways
    result = rule._validate_impl("wow\naba\nwow")
    assert result.passed
    assert result.metadata["symmetry_score"] == 1.0

    # Perfect horizontal but not vertical
    result = rule._validate_impl("wow\ndad\nmon")
    assert not result.passed
    assert result.metadata["symmetry_score"] < 1.0

def test_pattern_repeat_validation():
    """Test repeat pattern validation."""
    rule = create_pattern_rule(
        name="test_repeat",
        description="Test repeat pattern",
        pattern_type="repeat",
        pattern_length=3,
    )

    # Simple repeat
    result = rule._validate_impl("abcabcabc")
    assert result.passed
    assert len(result.metadata["matches_found"]) > 0
    assert result.metadata["match_count"] > 0

    # No repeat
    result = rule._validate_impl("abcdef")
    assert not result.passed
    assert len(result.metadata["matches_found"]) == 0

    # Overlapping patterns
    rule_overlap = create_pattern_rule(
        name="test_overlap",
        description="Test overlapping patterns",
        pattern_type="repeat",
        pattern_length=2,
        allow_overlap=True,
    )
    result = rule_overlap._validate_impl("aaaa")
    assert result.passed
    assert len(result.metadata["matches_found"]) > 1

def test_pattern_alternate_validation():
    """Test alternate pattern validation."""
    rule = create_pattern_rule(
        name="test_alternate",
        description="Test alternate pattern",
        pattern_type="alternate",
        pattern_length=1,
    )

    # Simple alternation
    result = rule._validate_impl("ababab")
    assert result.passed
    assert len(result.metadata["matches_found"]) > 0

    # No alternation
    result = rule._validate_impl("aaaaaa")
    assert not result.passed
    assert len(result.metadata["matches_found"]) == 0

    # Case sensitivity
    rule_case_insensitive = create_pattern_rule(
        name="test_case",
        description="Test case insensitive",
        pattern_type="alternate",
        pattern_length=1,
        case_sensitive=False,
    )
    result = rule_case_insensitive._validate_impl("AbAbAb")
    assert result.passed

def test_pattern_custom_validation():
    """Test custom pattern validation."""
    rule = create_pattern_rule(
        name="test_custom",
        description="Test custom pattern",
        pattern_type="custom",
        custom_pattern="hello",
    )

    # Pattern present
    result = rule._validate_impl("say hello world hello there")
    assert result.passed
    assert "hello" in result.metadata["matches_found"]

    # Pattern absent
    result = rule._validate_impl("no pattern here")
    assert not result.passed
    assert len(result.metadata["matches_found"]) == 0

    # Case sensitivity
    rule_case_insensitive = create_pattern_rule(
        name="test_case",
        description="Test case insensitive",
        pattern_type="custom",
        custom_pattern="Hello",
        case_sensitive=False,
    )
    result = rule_case_insensitive._validate_impl("say HELLO world")
    assert result.passed

def test_default_configs():
    """Test default configurations."""
    # Test reflection configs
    assert "palindrome" in DEFAULT_REFLECTION_CONFIGS
    palindrome_config = DEFAULT_REFLECTION_CONFIGS["palindrome"]
    assert palindrome_config["mirror_mode"] == "horizontal"
    assert palindrome_config["preserve_case"] is False

    assert "visual_mirror" in DEFAULT_REFLECTION_CONFIGS
    mirror_config = DEFAULT_REFLECTION_CONFIGS["visual_mirror"]
    assert mirror_config["mirror_mode"] == "both"
    assert mirror_config["symmetry_threshold"] == 0.8

    # Test pattern configs
    assert "word_repeat" in DEFAULT_PATTERN_CONFIGS
    word_config = DEFAULT_PATTERN_CONFIGS["word_repeat"]
    assert word_config["pattern_type"] == "repeat"
    assert word_config["pattern_length"] == 4

    assert "character_alternate" in DEFAULT_PATTERN_CONFIGS
    char_config = DEFAULT_PATTERN_CONFIGS["character_alternate"]
    assert char_config["pattern_type"] == "alternate"
    assert char_config["pattern_length"] == 1

def test_invalid_input_type():
    """Test invalid input type handling."""
    reflection_rule = create_reflection_rule(
        name="test_type",
        description="Test invalid type",
    )
    with pytest.raises(ValueError, match="Input must be a string"):
        reflection_rule._validate_impl(123)  # type: ignore

    pattern_rule = create_pattern_rule(
        name="test_type",
        description="Test invalid type",
        pattern_type="repeat",
    )
    with pytest.raises(ValueError, match="Input must be a string"):
        pattern_rule._validate_impl(123)  # type: ignore

def test_factory_functions():
    """Test factory functions for creating rules."""
    # Test reflection rule factory
    rule = create_reflection_rule(
        name="test_factory",
        description="Test factory creation",
        mirror_mode="horizontal",
        symmetry_threshold=0.8,
    )
    assert isinstance(rule, ReflectionRule)
    assert rule.name == "test_factory"
    assert rule.description == "Test factory creation"

    # Test pattern rule factory
    rule = create_pattern_rule(
        name="test_factory",
        description="Test factory creation",
        pattern_type="repeat",
        pattern_length=3,
    )
    assert isinstance(rule, PatternRule)
    assert rule.name == "test_factory"
    assert rule.description == "Test factory creation"
