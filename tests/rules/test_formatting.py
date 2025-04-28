"""Tests for formatting rules."""


import pytest

from sifaka.rules.formatting import (
    DefaultFormattingValidator,
    DefaultLengthValidator,
    DefaultParagraphValidator,
    DefaultStyleValidator,
    FormattingConfig,
    FormattingRule,
    FormattingValidator,
    LengthConfig,
    LengthRule,
    LengthValidator,
    ParagraphConfig,
    ParagraphRule,
    ParagraphValidator,
    StyleConfig,
    StyleRule,
    StyleValidator,
    create_formatting_rule,
    create_length_rule,
    create_paragraph_rule,
    create_style_rule,
)

@pytest.fixture
def length_config() -> LengthConfig:
    """Create a test length configuration."""
    return LengthConfig(
        min_length=10,
        max_length=100,
        unit="characters",
        cache_size=10,
        priority=2,
        cost=1.5,
    )

@pytest.fixture
def paragraph_config() -> ParagraphConfig:
    """Create a test paragraph configuration."""
    return ParagraphConfig(
        min_sentences=2,
        max_sentences=5,
        min_words_per_sentence=5,
        max_words_per_sentence=20,
        cache_size=10,
        priority=2,
        cost=1.5,
    )

@pytest.fixture
def style_config() -> StyleConfig:
    """Create a test style configuration."""
    return StyleConfig(
        style_indicators={"formal", "technical", "concise"},
        min_style_score=0.7,
        cache_size=10,
        priority=2,
        cost=1.5,
    )

@pytest.fixture
def formatting_config() -> FormattingConfig:
    """Create a test formatting configuration."""
    return FormattingConfig(
        patterns=[r"\b[A-Z][a-z]+\b", r"\b\d+\b"],
        min_matches=1,
        max_matches=10,
        cache_size=10,
        priority=2,
        cost=1.5,
    )

@pytest.fixture
def length_validator(length_config: LengthConfig) -> LengthValidator:
    """Create a test length validator."""
    return DefaultLengthValidator(length_config)

@pytest.fixture
def paragraph_validator(paragraph_config: ParagraphConfig) -> ParagraphValidator:
    """Create a test paragraph validator."""
    return DefaultParagraphValidator(paragraph_config)

@pytest.fixture
def style_validator(style_config: StyleConfig) -> StyleValidator:
    """Create a test style validator."""
    return DefaultStyleValidator(style_config)

@pytest.fixture
def formatting_validator(formatting_config: FormattingConfig) -> FormattingValidator:
    """Create a test formatting validator."""
    return DefaultFormattingValidator(formatting_config)

def test_length_config_validation():
    """Test length configuration validation."""
    # Test valid configuration
    config = LengthConfig(min_length=10, max_length=100)
    assert config.min_length == 10
    assert config.max_length == 100

    # Test invalid configurations
    with pytest.raises(ValueError, match="min_length must be non-negative"):
        LengthConfig(min_length=-1)

    with pytest.raises(ValueError, match="max_length must be greater than min_length"):
        LengthConfig(min_length=100, max_length=10)

    with pytest.raises(ValueError, match="unit must be one of"):
        LengthConfig(unit="invalid")

def test_paragraph_config_validation():
    """Test paragraph configuration validation."""
    # Test valid configuration
    config = ParagraphConfig(
        min_sentences=2,
        max_sentences=5,
        min_words_per_sentence=5,
        max_words_per_sentence=20,
    )
    assert config.min_sentences == 2
    assert config.max_sentences == 5

    # Test invalid configurations
    with pytest.raises(ValueError, match="min_sentences must be positive"):
        ParagraphConfig(min_sentences=0)

    with pytest.raises(ValueError, match="max_sentences must be greater than min_sentences"):
        ParagraphConfig(min_sentences=5, max_sentences=2)

def test_style_config_validation():
    """Test style configuration validation."""
    # Test valid configuration
    config = StyleConfig(
        style_indicators={"formal", "technical"},
        min_style_score=0.7,
    )
    assert "formal" in config.style_indicators
    assert config.min_style_score == 0.7

    # Test invalid configurations
    with pytest.raises(ValueError, match="style_indicators must be a set"):
        StyleConfig(style_indicators=["invalid"])  # type: ignore

    with pytest.raises(ValueError, match="min_style_score must be between 0 and 1"):
        StyleConfig(min_style_score=1.5)

def test_length_validation(length_validator: LengthValidator):
    """Test length validation."""
    rule = LengthRule(
        name="Test Length Rule",
        description="Test length validation",
        validator=length_validator,
    )

    # Test valid length
    text = "This is a valid length text for testing."
    result = rule.validate(text)
    assert result.passed
    assert result.metadata["length"] >= rule.validator.config.min_length

    # Test too short
    text = "Too short"
    result = rule.validate(text)
    assert not result.passed
    assert "Text length below minimum" in result.message

    # Test too long
    text = "x" * (rule.validator.config.max_length + 1)
    result = rule.validate(text)
    assert not result.passed
    assert "Text length exceeds maximum" in result.message

def test_paragraph_validation(paragraph_validator: ParagraphValidator):
    """Test paragraph validation."""
    rule = ParagraphRule(
        name="Test Paragraph Rule",
        description="Test paragraph validation",
        validator=paragraph_validator,
    )

    # Test valid paragraph structure
    text = """This is a good first sentence. This is a good second sentence with enough words.
    The third sentence maintains proper structure. The fourth sentence concludes well."""
    result = rule.validate(text)
    assert result.passed
    assert result.metadata["sentence_count"] >= rule.validator.config.min_sentences

    # Test too few sentences
    text = "This is a single sentence."
    result = rule.validate(text)
    assert not result.passed
    assert "Too few sentences" in result.message

    # Test too many words per sentence
    text = "This sentence has way too many words and goes on and on and on and on and on and on and on and on and on and on and on making it much longer than it should be according to our configuration."
    result = rule.validate(text)
    assert not result.passed
    assert "Sentence exceeds maximum words" in result.message

def test_style_validation(style_validator: StyleValidator):
    """Test style validation."""
    rule = StyleRule(
        name="Test Style Rule",
        description="Test style validation",
        validator=style_validator,
    )

    # Test formal style
    text = "The implementation demonstrates a sophisticated approach to solving complex technical challenges."
    result = rule.validate(text)
    assert result.passed
    assert result.metadata["style_score"] >= rule.validator.config.min_style_score

    # Test informal style
    text = "Hey guys! Check out this cool thing I made!"
    result = rule.validate(text)
    assert not result.passed
    assert "Style score below minimum" in result.message

def test_formatting_validation(formatting_validator: FormattingValidator):
    """Test formatting validation."""
    rule = FormattingRule(
        name="Test Formatting Rule",
        description="Test formatting validation",
        validator=formatting_validator,
    )

    # Test valid formatting
    text = "John has 42 apples and Mary has 15 oranges."
    result = rule.validate(text)
    assert result.passed
    assert len(result.metadata["matches"]) >= rule.validator.config.min_matches

    # Test no matches
    text = "no proper names or numbers here"
    result = rule.validate(text)
    assert not result.passed
    assert "No pattern matches found" in result.message

    # Test too many matches
    text = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    result = rule.validate(text)
    assert not result.passed
    assert "Too many pattern matches" in result.message

def test_factory_functions():
    """Test factory functions for creating rules."""
    # Test length rule creation
    length_rule = create_length_rule(
        name="Length Rule",
        description="Test length validation",
    )
    assert isinstance(length_rule.validator, DefaultLengthValidator)

    # Test paragraph rule creation
    paragraph_rule = create_paragraph_rule(
        name="Paragraph Rule",
        description="Test paragraph validation",
    )
    assert isinstance(paragraph_rule.validator, DefaultParagraphValidator)

    # Test style rule creation
    style_rule = create_style_rule(
        name="Style Rule",
        description="Test style validation",
    )
    assert isinstance(style_rule.validator, DefaultStyleValidator)

    # Test formatting rule creation
    formatting_rule = create_formatting_rule(
        name="Formatting Rule",
        description="Test formatting validation",
    )
    assert isinstance(formatting_rule.validator, DefaultFormattingValidator)

def test_edge_cases():
    """Test edge cases and error handling."""
    rule = create_length_rule(
        name="Test Rule",
        description="Test validation",
    )

    # Test empty text
    result = rule.validate("")
    assert not result.passed
    assert "Empty text" in result.message

    # Test invalid input type
    with pytest.raises(ValueError, match="Text must be a string"):
        rule.validate(123)  # type: ignore

    # Test None input
    with pytest.raises(ValueError, match="Text must be a string"):
        rule.validate(None)  # type: ignore

def test_consistent_results():
    """Test that validation results are consistent."""
    rule = create_paragraph_rule(
        name="Test Rule",
        description="Test validation",
    )
    text = """This is a test paragraph. It has multiple sentences.
    The sentences are well-formed. The structure is consistent."""

    # Multiple validations should yield the same result
    result1 = rule.validate(text)
    result2 = rule.validate(text)
    assert result1.passed == result2.passed
    assert result1.message == result2.message
    assert result1.metadata == result2.metadata
