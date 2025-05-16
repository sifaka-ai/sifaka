"""
Tests for component creation and registration.

This module tests the creation and registration of components in Sifaka.
"""

import pytest
from unittest.mock import Mock, patch

from sifaka.interfaces import Model, Validator, Improver
from sifaka.registry import (
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
        return {"passed": True, "message": "Validation passed"}


class MockImprover:
    """Mock improver for testing."""
    
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
    
    def improve(self, text):
        return f"Improved: {text}", {"changes_made": True, "message": "Improved"}


def test_model_registration_and_retrieval():
    """Test registering and retrieving a model factory."""
    # Register a model factory
    @register_model("test_component_model")
    def create_test_model(model_name, **options):
        return MockModel(model_name, **options)
    
    # Get the factory
    factory = get_model_factory("test_component_model")
    
    # Check that the factory was registered correctly
    assert factory is not None
    
    # Create a model using the factory
    model = factory("test-model", temperature=0.7)
    
    # Check that the model was created correctly
    assert isinstance(model, MockModel)
    assert model.model_name == "test-model"
    assert model.kwargs["temperature"] == 0.7


def test_validator_registration_and_retrieval():
    """Test registering and retrieving a validator factory."""
    # Register a validator factory
    @register_validator("test_component_validator")
    def create_test_validator(**options):
        return MockValidator(**options)
    
    # Get the factory
    factory = get_validator_factory("test_component_validator")
    
    # Check that the factory was registered correctly
    assert factory is not None
    
    # Create a validator using the factory
    validator = factory(min_length=10, max_length=100)
    
    # Check that the validator was created correctly
    assert isinstance(validator, MockValidator)
    assert validator.kwargs["min_length"] == 10
    assert validator.kwargs["max_length"] == 100


def test_improver_registration_and_retrieval():
    """Test registering and retrieving an improver factory."""
    # Register an improver factory
    @register_improver("test_component_improver")
    def create_test_improver(model, **options):
        return MockImprover(model, **options)
    
    # Get the factory
    factory = get_improver_factory("test_component_improver")
    
    # Check that the factory was registered correctly
    assert factory is not None
    
    # Create a model
    model = MockModel("test-model")
    
    # Create an improver using the factory
    improver = factory(model, level="high")
    
    # Check that the improver was created correctly
    assert isinstance(improver, MockImprover)
    assert improver.model == model
    assert improver.kwargs["level"] == "high"


def test_model_registration_overwrite():
    """Test that registering a model factory with the same name overwrites the previous one."""
    # Register a model factory
    @register_model("test_overwrite_model")
    def create_test_model_1(model_name, **options):
        return MockModel(model_name, **options)
    
    # Register another model factory with the same name
    @register_model("test_overwrite_model")
    def create_test_model_2(model_name, **options):
        return MockModel(f"overwritten-{model_name}", **options)
    
    # Get the factory
    factory = get_model_factory("test_overwrite_model")
    
    # Create a model using the factory
    model = factory("test-model")
    
    # Check that the model was created using the second factory
    assert model.model_name == "overwritten-test-model"


def test_validator_registration_overwrite():
    """Test that registering a validator factory with the same name overwrites the previous one."""
    # Register a validator factory
    @register_validator("test_overwrite_validator")
    def create_test_validator_1(**options):
        return MockValidator(**options, source="original")
    
    # Register another validator factory with the same name
    @register_validator("test_overwrite_validator")
    def create_test_validator_2(**options):
        return MockValidator(**options, source="overwritten")
    
    # Get the factory
    factory = get_validator_factory("test_overwrite_validator")
    
    # Create a validator using the factory
    validator = factory()
    
    # Check that the validator was created using the second factory
    assert validator.kwargs["source"] == "overwritten"


def test_improver_registration_overwrite():
    """Test that registering an improver factory with the same name overwrites the previous one."""
    # Register an improver factory
    @register_improver("test_overwrite_improver")
    def create_test_improver_1(model, **options):
        return MockImprover(model, **options, source="original")
    
    # Register another improver factory with the same name
    @register_improver("test_overwrite_improver")
    def create_test_improver_2(model, **options):
        return MockImprover(model, **options, source="overwritten")
    
    # Get the factory
    factory = get_improver_factory("test_overwrite_improver")
    
    # Create a model
    model = MockModel("test-model")
    
    # Create an improver using the factory
    improver = factory(model)
    
    # Check that the improver was created using the second factory
    assert improver.kwargs["source"] == "overwritten"
