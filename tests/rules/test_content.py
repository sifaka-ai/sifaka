"""Tests for the content rules."""

from typing import (
    Any,
    Dict,
    List
)

import pytest

from sifaka.rules.content import (
    ConfigurationError,
    ContentAnalyzer,
    ProhibitedContentRule,
    ProhibitedTerms,
    RuleConfig,
    ToneAnalyzer,
    ToneConfig,
    ToneConsistencyRule,
    ToneIndicators,
)

class MockContentAnalyzer:
    """Mock content analyzer for testing."""

    def analyze(self, text: str) -> Dict[str, Any]:
        """Mock analysis."""
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "mock": True,
        }

    def can_analyze(self, text: str) -> bool:
        """Check if text can be analyzed."""
        return isinstance(text, str)

class MockToneAnalyzer:
    """Mock tone analyzer for testing."""

    def analyze_tone(self, text: str) -> Dict[str, float]:
        """Mock tone analysis."""
        return {
            "formality": 0.8,
            "complexity": 0.6,
            "mock": True,
        }

    def get_supported_tones(self) -> List[str]:
        """Get supported tones."""
        return ["formal", "informal"]

@pytest.fixture
def content_analyzer() -> MockContentAnalyzer:
    """Fixture for creating a mock content analyzer."""
    return MockContentAnalyzer()

@pytest.fixture
def tone_analyzer() -> MockToneAnalyzer:
    """Fixture for creating a mock tone analyzer."""
    return MockToneAnalyzer()

@pytest.fixture
def prohibited_terms() -> ProhibitedTerms:
    """Fixture for creating prohibited terms."""
    return ProhibitedTerms(
        terms=frozenset(["bad", "inappropriate"]),
        case_sensitive=False,
    )

@pytest.fixture
def tone_indicators() -> Dict[str, ToneIndicators]:
    """Fixture for creating tone indicators."""
    return {
        "formal": ToneIndicators(
            positive=frozenset(
                [
                    "therefore",
                    "furthermore",
                    "consequently",
                    "analysis",
                    "proceed",
                ]
            ),
            negative=frozenset(
                [
                    "hey",
                    "cool",
                    "awesome",
                    "wow",
                    "yeah",
                ]
            ),
        )
    }

@pytest.fixture
def tone_config(tone_indicators: Dict[str, ToneIndicators]) -> ToneConfig:
    """Fixture for creating tone config."""
    return ToneConfig(
        expected_tone="formal",
        indicators=tone_indicators,
        threshold=0.7,
        priority=RuleConfig().priority,
        cache_size=10,
        cost=2,
        metadata={"test": True},
    )

@pytest.fixture
def prohibited_rule(
    prohibited_terms: ProhibitedTerms,
    content_analyzer: MockContentAnalyzer,
) -> ProhibitedContentRule:
    """Fixture for creating a prohibited content rule."""
    return ProhibitedContentRule(
        name="test_prohibited",
        description="Test prohibited content rule",
        terms=list(prohibited_terms.terms),
        case_sensitive=prohibited_terms.case_sensitive,
        analyzer=content_analyzer,
        config=RuleConfig(cache_size=10, cost=2),
    )

@pytest.fixture
def tone_rule(
    tone_config: ToneConfig,
    tone_analyzer: MockToneAnalyzer,
) -> ToneConsistencyRule:
    """Fixture for creating a tone consistency rule."""
    return ToneConsistencyRule(
        name="test_tone",
        description="Test tone consistency rule",
        expected_tone=tone_config.expected_tone,
        indicators={
            tone: {
                "positive": list(indicators.positive),
                "negative": list(indicators.negative),
            }
            for tone, indicators in tone_config.indicators.items()
        },
        threshold=tone_config.threshold,
        analyzer=tone_analyzer,
        config=RuleConfig(cache_size=10, cost=2),
    )

def test_content_analyzer_protocol():
    """Test ContentAnalyzer protocol implementation."""
    analyzer = MockContentAnalyzer()
    assert isinstance(analyzer, ContentAnalyzer)

    # Test non-compliant object
    class BadAnalyzer:
        def analyze(self, text: str) -> str:
            return text

    bad_analyzer = BadAnalyzer()
    assert not isinstance(bad_analyzer, ContentAnalyzer)

def test_tone_analyzer_protocol():
    """Test ToneAnalyzer protocol implementation."""
    analyzer = MockToneAnalyzer()
    assert isinstance(analyzer, ToneAnalyzer)

    # Test non-compliant object
    class BadAnalyzer:
        def analyze_tone(self, text: str) -> str:
            return text

    bad_analyzer = BadAnalyzer()
    assert not isinstance(bad_analyzer, ToneAnalyzer)

def test_prohibited_terms():
    """Test ProhibitedTerms initialization and behavior."""
    # Test valid initialization
    terms = ProhibitedTerms(terms=frozenset(["bad", "inappropriate"]))
    assert "bad" in terms.terms
    assert not terms.case_sensitive

    # Test empty terms
    with pytest.raises(ConfigurationError):
        ProhibitedTerms(terms=frozenset())

    # Test with_terms
    new_terms = terms.with_terms(["terrible", "horrible"])
    assert "terrible" in new_terms.terms
    assert "bad" not in new_terms.terms
    assert new_terms.case_sensitive == terms.case_sensitive

    # Test with_case_sensitivity
    case_sensitive = terms.with_case_sensitivity(True)
    assert case_sensitive.case_sensitive
    assert case_sensitive.terms == terms.terms

def test_tone_indicators():
    """Test ToneIndicators initialization and behavior."""
    # Test valid initialization
    indicators = ToneIndicators(
        positive=frozenset(["good", "great"]),
        negative=frozenset(["bad", "poor"]),
    )
    assert "good" in indicators.positive
    assert "bad" in indicators.negative

    # Test empty indicators
    with pytest.raises(ConfigurationError):
        ToneIndicators(positive=frozenset(), negative=frozenset())

    # Test one empty set
    indicators = ToneIndicators(
        positive=frozenset(["good"]),
        negative=frozenset(),
    )
    assert "good" in indicators.positive
    assert not indicators.negative

def test_tone_config():
    """Test ToneConfig initialization and behavior."""
    # Test valid initialization
    indicators = {
        "formal": ToneIndicators(
            positive=frozenset(["therefore"]),
            negative=frozenset(["hey"]),
        )
    }
    config = ToneConfig(
        expected_tone="formal",
        indicators=indicators,
        threshold=0.7,
    )
    assert config.expected_tone == "formal"
    assert config.threshold == 0.7
    assert "formal" in config.indicators

    # Test invalid threshold
    with pytest.raises(ConfigurationError):
        ToneConfig(
            expected_tone="formal",
            indicators=indicators,
            threshold=1.5,
        )

    # Test empty tone
    with pytest.raises(ConfigurationError):
        ToneConfig(
            expected_tone="",
            indicators=indicators,
        )

    # Test empty indicators
    with pytest.raises(ConfigurationError):
        ToneConfig(
            expected_tone="formal",
            indicators={},
        )

    # Test with_tone
    new_config = config.with_tone("informal")
    assert new_config.expected_tone == "informal"
    assert new_config.threshold == config.threshold
    assert new_config.indicators == config.indicators

    # Test with_threshold
    new_config = config.with_threshold(0.8)
    assert new_config.threshold == 0.8
    assert new_config.expected_tone == config.expected_tone
    assert new_config.indicators == config.indicators

def test_prohibited_content_validation(prohibited_rule: ProhibitedContentRule):
    """Test prohibited content rule validation."""
    # Test clean text
    clean_text = "This is a good and appropriate text."
    result = prohibited_rule.validate(clean_text)
    assert result.passed
    assert "found_terms" in result.metadata
    assert len(result.metadata["found_terms"]) == 0
    assert result.metadata["analysis"]["mock"]  # Check analyzer was used

    # Test text with prohibited terms
    bad_text = "This is a bad and inappropriate text."
    result = prohibited_rule.validate(bad_text)
    assert not result.passed
    assert "found_terms" in result.metadata
    assert len(result.metadata["found_terms"]) == 2
    assert "bad" in result.metadata["found_terms"]
    assert "inappropriate" in result.metadata["found_terms"]
    assert result.metadata["analysis"]["mock"]  # Check analyzer was used

    # Test case sensitivity
    mixed_case_text = "This is BAD and InAppropriate text."
    result = prohibited_rule.validate(mixed_case_text)
    assert not result.passed
    assert len(result.metadata["found_terms"]) == 2

    # Test invalid input
    with pytest.raises(TypeError):
        prohibited_rule.validate(123)  # type: ignore

def test_tone_consistency_validation(tone_rule: ToneConsistencyRule):
    """Test tone consistency rule validation."""
    # Test formal tone
    formal_text = "Therefore, we must proceed. Furthermore, the analysis shows."
    result = tone_rule.validate(formal_text)
    assert result.passed
    assert "tone_scores" in result.metadata
    assert result.metadata["tone_scores"]["mock"]  # Check analyzer was used
    assert "positive_indicators" in result.metadata
    assert "negative_indicators" in result.metadata
    assert "consistency_score" in result.metadata
    assert result.metadata["threshold"] == 0.7

    # Test informal tone
    informal_text = "Hey! What's up? This is cool!"
    result = tone_rule.validate(informal_text)
    assert not result.passed
    assert "tone_scores" in result.metadata
    assert result.metadata["tone_scores"]["mock"]  # Check analyzer was used

    # Test invalid input
    with pytest.raises(TypeError):
        tone_rule.validate(123)  # type: ignore

    # Test unknown tone
    rule = ToneConsistencyRule(
        name="test",
        description="test",
        expected_tone="unknown",
        indicators={"formal": {"positive": [], "negative": []}},
    )
    result = rule.validate("test")
    assert not result.passed
    assert "available_tones" in result.metadata

def test_error_handling():
    """Test error handling in content rules."""
    # Test invalid prohibited terms
    with pytest.raises(ConfigurationError):
        ProhibitedTerms(terms=frozenset())

    # Test invalid tone config
    with pytest.raises(ConfigurationError):
        ToneConfig(
            expected_tone="",
            indicators={},
            threshold=1.5,
        )

    # Test invalid analyzer
    with pytest.raises(ConfigurationError):
        ProhibitedContentRule(
            name="test",
            description="test",
            terms=["bad"],
            analyzer="not an analyzer",  # type: ignore
        )

def test_consistent_results(
    prohibited_rule: ProhibitedContentRule,
    tone_rule: ToneConsistencyRule,
):
    """Test consistency of validation results."""
    # Test prohibited content rule
    text = "This is a test text."
    results = [prohibited_rule.validate(text) for _ in range(3)]
    first = results[0]
    for result in results[1:]:
        assert result.passed == first.passed
        assert result.message == first.message
        assert result.metadata == first.metadata

    # Test tone consistency rule
    text = "Therefore, we proceed with the analysis."
    results = [tone_rule.validate(text) for _ in range(3)]
    first = results[0]
    for result in results[1:]:
        assert result.passed == first.passed
        assert result.message == first.message
        assert result.metadata == first.metadata
