"""
Tests for structure validation rules.

This module contains tests for the structure validation rules in
sifaka/rules/formatting/structure.py.
"""

import pytest
from pydantic import ValidationError

from sifaka.rules.formatting.structure import (
    StructureConfig,
    StructureValidator,
    StructureRule,
    create_structure_validator,
    create_structure_rule,
)
from sifaka.rules.base import RuleConfig, RuleResult


class TestStructureConfig:
    """Tests for StructureConfig."""

    def test_default_config(self):
        """Test that default config is created correctly."""
        config = StructureConfig()
        assert config.required_sections == []
        assert config.min_sections == 1
        assert config.max_sections is None
        assert config.cache_size == 100

    def test_custom_config(self):
        """Test that custom config is created correctly."""
        config = StructureConfig(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3,
            max_sections=5,
            cache_size=50,
        )
        assert config.required_sections == ["introduction", "body", "conclusion"]
        assert config.min_sections == 3
        assert config.max_sections == 5
        assert config.cache_size == 50

    def test_validation(self):
        """Test that config validation works correctly."""
        # Test min_sections validation
        with pytest.raises(ValidationError):
            StructureConfig(min_sections=-1)

        # Test max_sections validation
        with pytest.raises(ValidationError):
            StructureConfig(max_sections=-1)

        # Test cache_size validation
        with pytest.raises(ValidationError):
            StructureConfig(cache_size=0)


class TestStructureValidator:
    """Tests for StructureValidator."""

    def test_initialization(self):
        """Test that validator initializes correctly with state management."""
        validator = StructureValidator()
        
        # Check that state is initialized
        assert validator._state_manager.get("config") is not None
        assert validator._state_manager.get("cache") == {}
        assert validator._state_manager.get_metadata("validator_type") == "StructureValidator"
        assert validator._state_manager.get_metadata("creation_time") is not None

    def test_config_property(self):
        """Test that config property returns the correct value."""
        config = StructureConfig(min_sections=3)
        validator = StructureValidator(config=config)
        
        # Check that config is stored in state
        assert validator.config == config
        assert validator._state_manager.get("config") == config

    def test_validate_empty_text(self):
        """Test validation of empty text."""
        validator = StructureValidator()
        result = validator.validate("")
        
        # Check that result is correct
        assert not result.passed
        assert "empty" in result.message.lower()
        
        # Check that state is updated
        assert validator._state_manager.get_metadata("validation_count") == 1

    def test_validate_valid_text(self):
        """Test validation of valid text."""
        validator = create_structure_validator(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3,
        )
        text = "# Introduction\nContent\n# Body\nMore content\n# Conclusion\nFinal content"
        result = validator.validate(text)
        
        # Check that result is correct
        assert result.passed
        assert "valid" in result.message.lower()
        
        # Check that state is updated
        assert validator._state_manager.get_metadata("validation_count") == 1
        assert len(validator._state_manager.get("cache")) == 1

    def test_validate_invalid_text(self):
        """Test validation of invalid text."""
        validator = create_structure_validator(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3,
        )
        text = "# Introduction\nContent"
        result = validator.validate(text)
        
        # Check that result is correct
        assert not result.passed
        assert "sections" in result.message.lower()
        
        # Check that state is updated
        assert validator._state_manager.get_metadata("validation_count") == 1
        assert len(validator._state_manager.get("cache")) == 1

    def test_cache_functionality(self):
        """Test that caching works correctly."""
        validator = create_structure_validator(cache_size=10)
        text = "# Introduction\nContent\n# Body\nMore content"
        
        # First validation should not use cache
        result1 = validator.validate(text)
        assert validator._state_manager.get_metadata("cache_hit") is False
        
        # Second validation should use cache
        result2 = validator.validate(text)
        assert validator._state_manager.get_metadata("cache_hit") is True
        
        # Results should be identical
        assert result1 == result2
        
        # Cache should contain the result
        cache = validator._state_manager.get("cache")
        assert text in cache
        assert cache[text] == result1


class TestStructureRule:
    """Tests for StructureRule."""

    def test_initialization(self):
        """Test that rule initializes correctly with state management."""
        rule = create_structure_rule(
            name="test_rule",
            description="Test rule",
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3,
        )
        
        # Check that state is initialized
        assert rule._state_manager.get("structure_validator") is not None
        assert rule._state_manager.get("validator_config") is not None
        assert rule._state_manager.get_metadata("rule_id") is not None

    def test_create_default_validator(self):
        """Test that _create_default_validator works correctly."""
        # Create rule with params but no validator
        config = RuleConfig(
            name="test_rule",
            description="Test rule",
            rule_id="test_rule_id",
            params={
                "required_sections": ["introduction", "body", "conclusion"],
                "min_sections": 3,
                "max_sections": 5,
            },
        )
        rule = StructureRule(
            name="test_rule",
            description="Test rule",
            config=config,
            validator=None,
        )
        
        # Create default validator
        validator = rule._create_default_validator()
        
        # Check that validator is created correctly
        assert isinstance(validator, StructureValidator)
        assert validator.config.required_sections == ["introduction", "body", "conclusion"]
        assert validator.config.min_sections == 3
        assert validator.config.max_sections == 5
        
        # Check that state is updated
        assert rule._state_manager.get("validator_config") == validator.config

    def test_validate(self):
        """Test that validate works correctly."""
        rule = create_structure_rule(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3,
        )
        
        # Valid text
        valid_text = "# Introduction\nContent\n# Body\nMore content\n# Conclusion\nFinal content"
        valid_result = rule.validate(valid_text)
        assert valid_result.passed
        
        # Invalid text
        invalid_text = "# Introduction\nContent"
        invalid_result = rule.validate(invalid_text)
        assert not invalid_result.passed
        
        # Check that state is updated
        assert rule._state_manager.get("execution_count") == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_structure_validator(self):
        """Test that create_structure_validator works correctly."""
        validator = create_structure_validator(
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3,
            max_sections=5,
            cache_size=50,
        )
        
        # Check that validator is created correctly
        assert isinstance(validator, StructureValidator)
        assert validator.config.required_sections == ["introduction", "body", "conclusion"]
        assert validator.config.min_sections == 3
        assert validator.config.max_sections == 5
        assert validator.config.cache_size == 50
        
        # Check that state is initialized
        assert validator._state_manager.get("config") is not None
        assert validator._state_manager.get("cache") == {}

    def test_create_structure_rule(self):
        """Test that create_structure_rule works correctly."""
        rule = create_structure_rule(
            name="test_rule",
            description="Test rule",
            required_sections=["introduction", "body", "conclusion"],
            min_sections=3,
            max_sections=5,
            rule_id="test_rule_id",
            severity="warning",
            category="formatting",
            tags=["structure", "formatting"],
        )
        
        # Check that rule is created correctly
        assert isinstance(rule, StructureRule)
        assert rule.name == "test_rule"
        assert rule.description == "Test rule"
        assert rule.config.rule_id == "test_rule_id"
        assert rule.config.severity == "warning"
        assert rule.config.category == "formatting"
        assert rule.config.tags == ["structure", "formatting"]
        
        # Check that validator is created correctly
        validator_config = rule._state_manager.get("validator_config")
        assert validator_config.required_sections == ["introduction", "body", "conclusion"]
        assert validator_config.min_sections == 3
        assert validator_config.max_sections == 5
        
        # Check that state is initialized
        assert rule._state_manager.get("structure_validator") is not None
