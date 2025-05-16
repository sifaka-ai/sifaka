"""
Tests for the registry system.

This module tests the registry system in sifaka.registry.
"""

import pytest
from unittest.mock import Mock

from sifaka.interfaces import Model, Validator, Improver
from sifaka.registry import (
    Registry,
    get_registry,
    register_model,
    register_validator,
    register_improver,
    get_model_factory,
    get_validator_factory,
    get_improver_factory,
)


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    def generate(self, prompt, **options):
        return f"Generated text for: {prompt}"
    
    def count_tokens(self, text):
        return len(text.split())


class MockValidator:
    """Mock validator for testing."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def validate(self, text):
        # Simple mock validation result
        return type("ValidationResult", (), {
            "passed": True,
            "message": "Validation passed",
            "details": {}
        })


class MockImprover:
    """Mock improver for testing."""
    
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
    
    def improve(self, text):
        improved_text = f"Improved: {text}"
        # Simple mock improvement result
        result = type("ImprovementResult", (), {
            "passed": True,
            "message": "Improvement successful",
            "original_text": text,
            "improved_text": improved_text,
            "changes_made": True,
            "details": {}
        })
        return improved_text, result


def test_registry_singleton():
    """Test that the registry is a singleton."""
    registry1 = get_registry()
    registry2 = get_registry()
    assert registry1 is registry2


def test_register_and_get_model():
    """Test registering and retrieving a model factory."""
    # Register a test model factory
    @register_model("test_model")
    def create_test_model(model_name, **options):
        return MockModel(model_name, **options)
    
    # Get the factory
    factory = get_model_factory("test_model")
    assert factory is not None
    
    # Create a model using the factory
    model = factory("test", temperature=0.7)
    assert isinstance(model, MockModel)
    assert model.model_name == "test"
    assert model.kwargs["temperature"] == 0.7


def test_register_and_get_validator():
    """Test registering and retrieving a validator factory."""
    # Register a test validator factory
    @register_validator("test_validator")
    def create_test_validator(**options):
        return MockValidator(**options)
    
    # Get the factory
    factory = get_validator_factory("test_validator")
    assert factory is not None
    
    # Create a validator using the factory
    validator = factory(min_length=10, max_length=100)
    assert isinstance(validator, MockValidator)
    assert validator.kwargs["min_length"] == 10
    assert validator.kwargs["max_length"] == 100


def test_register_and_get_improver():
    """Test registering and retrieving an improver factory."""
    # Register a test improver factory
    @register_improver("test_improver")
    def create_test_improver(model, **options):
        return MockImprover(model, **options)
    
    # Get the factory
    factory = get_improver_factory("test_improver")
    assert factory is not None
    
    # Create an improver using the factory
    model = MockModel("test")
    improver = factory(model, level="high")
    assert isinstance(improver, MockImprover)
    assert improver.model is model
    assert improver.kwargs["level"] == "high"


def test_case_insensitivity():
    """Test that component names are case-insensitive."""
    # Register with uppercase
    @register_model("TEST_CASE")
    def create_test_model(model_name, **options):
        return MockModel(model_name, **options)
    
    # Get with lowercase
    factory = get_model_factory("test_case")
    assert factory is not None
    
    # Create a model
    model = factory("test")
    assert isinstance(model, MockModel)


def test_lazy_loading():
    """Test that lazy loading works correctly."""
    # This test is more of a placeholder since we can't easily test
    # the actual lazy loading behavior in a unit test
    registry = get_registry()
    
    # Add a test module to lazy imports
    component_type = "test_component"
    registry._lazy_imports[component_type] = ["nonexistent.module"]
    
    # Initialize the component type
    registry._initialize_component_type(component_type)
    
    # Check that the component type is marked as initialized
    assert component_type in registry._initialized_types


def test_initialize_all():
    """Test initializing all component types."""
    from sifaka.registry import initialize_all
    
    # Call initialize_all
    initialize_all()
    
    # Check that all component types in lazy_imports are initialized
    registry = get_registry()
    for component_type in registry._lazy_imports:
        assert component_type in registry._initialized_types
