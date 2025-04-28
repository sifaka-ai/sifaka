"""Tests for factual rules."""

from dataclasses import dataclass

import pytest

from sifaka.rules.base import RuleResult
from sifaka.rules.factual import (
    CitationConfig,
    CitationRule,
    ConfidenceConfig,
    ConfidenceRule,
    FactualAccuracyConfig,
    FactualAccuracyRule,
    FactualConsistencyConfig,
    FactualConsistencyRule,
)

@dataclass
class MockFactualConsistencyValidator:
    """Mock validator for testing factual consistency."""

    config: FactualConsistencyConfig

    def validate(self, text: str) -> RuleResult:
        """Mock validation that checks for contradictions."""
        for indicator in self.config.contradiction_indicators:
            if indicator in text.lower():
                return RuleResult(
                    passed=False,
                    message="Found contradiction",
                    metadata={"indicator": indicator},
                )
        return RuleResult(passed=True, message="No contradictions found")

@dataclass
class MockConfidenceValidator:
    """Mock validator for testing confidence levels."""

    config: ConfidenceConfig

    def validate(self, text: str) -> RuleResult:
        """Mock validation that checks confidence levels."""
        text_lower = text.lower()
        found_levels = {}
        for level, indicators in self.config.confidence_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    found_levels.setdefault(level, []).append(indicator)
        return RuleResult(
            passed=True,
            message="Confidence check complete",
            metadata={"levels": found_levels},
        )

@dataclass
class MockCitationValidator:
    """Mock validator for testing citations."""

    config: CitationConfig

    def validate(self, text: str) -> RuleResult:
        """Mock validation that checks for citations."""
        import re

        citations = []
        for pattern in self.config.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)

        if self.config.required_citations and not citations:
            return RuleResult(
                passed=False,
                message="No citations found",
                metadata={"required": True},
            )
        return RuleResult(
            passed=True,
            message="Citations check complete",
            metadata={"citations": citations},
        )

@dataclass
class MockFactualAccuracyValidator:
    """Mock validator for testing factual accuracy."""

    config: FactualAccuracyConfig

    def validate(self, text: str) -> RuleResult:
        """Mock validation that checks facts against knowledge base."""
        text_lower = text.lower()
        inaccuracies = []
        for fact, variations in self.config.knowledge_base.items():
            if not any(variation.lower() in text_lower for variation in variations):
                inaccuracies.append(fact)
        return RuleResult(
            passed=not inaccuracies,
            message="Factual check complete",
            metadata={"inaccuracies": inaccuracies},
        )

@pytest.fixture
def factual_consistency_config() -> FactualConsistencyConfig:
    """Create a test configuration for factual consistency."""
    return FactualConsistencyConfig(
        contradiction_indicators=["but", "however"],
        confidence_threshold=0.8,
        cache_size=50,
        priority=2,
        cost=1.5,
    )

@pytest.fixture
def confidence_config() -> ConfidenceConfig:
    """Create a test configuration for confidence levels."""
    return ConfidenceConfig(
        confidence_indicators={
            "high": ["definitely", "certainly"],
            "low": ["maybe", "possibly"],
        },
        cache_size=50,
        priority=2,
        cost=1.5,
    )

@pytest.fixture
def citation_config() -> CitationConfig:
    """Create a test configuration for citations."""
    return CitationConfig(
        citation_patterns=[r"\[[\d]+\]", r"\([A-Za-z]+, \d{4}\)"],
        required_citations=True,
        cache_size=50,
        priority=2,
        cost=1.5,
    )

@pytest.fixture
def factual_accuracy_config() -> FactualAccuracyConfig:
    """Create a test configuration for factual accuracy."""
    return FactualAccuracyConfig(
        knowledge_base={
            "earth": {"round", "spherical"},
            "water": {"H2O", "dihydrogen monoxide"},
        },
        cache_size=50,
        priority=2,
        cost=1.5,
    )

def test_factual_consistency_config_validation():
    """Test factual consistency configuration validation."""
    with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
        FactualConsistencyConfig(confidence_threshold=1.5)

    with pytest.raises(ValueError, match="cache_size must be positive"):
        FactualConsistencyConfig(cache_size=0)

    with pytest.raises(ValueError, match="priority must be non-negative"):
        FactualConsistencyConfig(priority=-1)

    with pytest.raises(ValueError, match="cost must be non-negative"):
        FactualConsistencyConfig(cost=-1.0)

def test_confidence_config_validation():
    """Test confidence configuration validation."""
    with pytest.raises(ValueError, match="confidence_indicators must be a Dict"):
        ConfidenceConfig(confidence_indicators={"high": "not a list"})  # type: ignore

    with pytest.raises(ValueError, match="cache_size must be positive"):
        ConfidenceConfig(cache_size=0)

def test_citation_config_validation():
    """Test citation configuration validation."""
    with pytest.raises(ValueError, match="citation_patterns must be a List"):
        CitationConfig(citation_patterns={"not": "a list"})  # type: ignore

    with pytest.raises(ValueError, match="cache_size must be positive"):
        CitationConfig(cache_size=0)

def test_factual_accuracy_config_validation():
    """Test factual accuracy configuration validation."""
    with pytest.raises(ValueError, match="knowledge_base must be a Dict"):
        FactualAccuracyConfig(knowledge_base={"key": "not a set"})  # type: ignore

    with pytest.raises(ValueError, match="cache_size must be positive"):
        FactualAccuracyConfig(cache_size=0)

def test_factual_consistency_rule(factual_consistency_config: FactualConsistencyConfig):
    """Test factual consistency rule validation."""
    validator = MockFactualConsistencyValidator(config=factual_consistency_config)
    rule = FactualConsistencyRule(
        name="test",
        description="test rule",
        config=factual_consistency_config,
        validator=validator,
    )

    # Test with contradiction
    result = rule.validate("This is true, but that is false")
    assert not result.passed
    assert "Found contradiction" in result.message
    assert result.metadata["indicator"] == "but"

    # Test without contradiction
    result = rule.validate("This is consistently true")
    assert result.passed
    assert "No contradictions found" in result.message

def test_confidence_rule(confidence_config: ConfidenceConfig):
    """Test confidence rule validation."""
    validator = MockConfidenceValidator(config=confidence_config)
    rule = ConfidenceRule(
        name="test",
        description="test rule",
        config=confidence_config,
        validator=validator,
    )

    # Test with confidence indicators
    result = rule.validate("I am definitely sure about this")
    assert result.passed
    assert "high" in result.metadata["levels"]
    assert "definitely" in result.metadata["levels"]["high"]

    # Test without confidence indicators
    result = rule.validate("This is a statement")
    assert result.passed
    assert not result.metadata["levels"]

def test_citation_rule(citation_config: CitationConfig):
    """Test citation rule validation."""
    validator = MockCitationValidator(config=citation_config)
    rule = CitationRule(
        name="test",
        description="test rule",
        config=citation_config,
        validator=validator,
    )

    # Test with citations
    result = rule.validate("According to [1] and (Smith, 2020)")
    assert result.passed
    assert len(result.metadata["citations"]) == 2

    # Test without citations
    result = rule.validate("This has no citations")
    assert not result.passed
    assert "No citations found" in result.message

def test_factual_accuracy_rule(factual_accuracy_config: FactualAccuracyConfig):
    """Test factual accuracy rule validation."""
    validator = MockFactualAccuracyValidator(config=factual_accuracy_config)
    rule = FactualAccuracyRule(
        name="test",
        description="test rule",
        config=factual_accuracy_config,
        validator=validator,
    )

    # Test with accurate facts
    result = rule.validate("The earth is round and water is H2O")
    assert result.passed
    assert not result.metadata["inaccuracies"]

    # Test with inaccurate facts
    result = rule.validate("The earth is flat")
    assert not result.passed
    assert "earth" in result.metadata["inaccuracies"]

def test_rule_type_validation():
    """Test type validation for rules."""
    with pytest.raises(ValueError, match="Text must be a string"):
        FactualConsistencyRule(
            name="test",
            description="test",
            config=FactualConsistencyConfig(),
            validator=MockFactualConsistencyValidator(config=FactualConsistencyConfig()),
        ).validate(
            None
        )  # type: ignore

    with pytest.raises(ValueError, match="Text must be a string"):
        ConfidenceRule(
            name="test",
            description="test",
            config=ConfidenceConfig(),
            validator=MockConfidenceValidator(config=ConfidenceConfig()),
        ).validate(
            123
        )  # type: ignore
