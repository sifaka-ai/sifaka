"""
Tests for the component registry system.

These tests ensure that the registry system correctly manages factory functions
and handles circular dependencies between modules. The registry is a key part
of the solution to circular import issues in the codebase.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import registry functionality
from sifaka.core import registry


# Mock factory functions for testing
def mock_factory_1(**kwargs):
    return {"type": "mock_1", **kwargs}


def mock_factory_2(**kwargs):
    return {"type": "mock_2", **kwargs}


def mock_factory_3(**kwargs):
    return {"type": "mock_3", **kwargs}


class TestRegistryBasic:
    """Basic tests for the registry functionality."""

    def test_register_and_get_factory(self):
        """Test basic registration and retrieval of factory functions."""
        # Register a factory function
        registry.register_factory("test_component", "test_factory", mock_factory_1)

        # Retrieve the factory function
        factory = registry.get_factory("test_component", "test_factory")

        # Check that we got the correct factory function
        assert factory is mock_factory_1

        # Verify the factory works as expected
        result = factory(param="value")
        assert result["type"] == "mock_1"
        assert result["param"] == "value"

    def test_get_nonexistent_factory(self):
        """Test retrieving a factory that doesn't exist."""
        # Try to get a factory that doesn't exist
        factory = registry.get_factory("nonexistent", "factory")

        # Should return None
        assert factory is None

    def test_register_invalid_component_type(self):
        """Test registering with an invalid component type."""
        # Try to register with an invalid component type
        with pytest.raises(ValueError) as excinfo:
            registry.register_factory("invalid_type", "test_factory", mock_factory_1)

        # Check the error message
        assert "Unsupported component type" in str(excinfo.value)
        assert "invalid_type" in str(excinfo.value)

    def test_get_available_factories(self):
        """Test getting all available factories for a component type."""
        # Register some factories
        registry.register_factory("test_component", "factory1", mock_factory_1)
        registry.register_factory("test_component", "factory2", mock_factory_2)

        # Get all available factories
        factories = registry.get_available_factories("test_component")

        # Check that we got the correct factories
        assert "factory1" in factories
        assert "factory2" in factories
        assert factories["factory1"] is mock_factory_1
        assert factories["factory2"] is mock_factory_2

    def test_get_available_factories_nonexistent_type(self):
        """Test getting factories for a component type that doesn't exist."""
        # Get factories for a component type that doesn't exist
        factories = registry.get_available_factories("nonexistent")

        # Should return an empty dict
        assert factories == {}

    def test_register_duplicate_factory(self):
        """Test registering a factory with the same name twice."""
        # Register a factory
        registry.register_factory("test_component", "duplicate", mock_factory_1)

        # Register another factory with the same name
        registry.register_factory("test_component", "duplicate", mock_factory_2)

        # Get the factory - should be the latest one registered
        factory = registry.get_factory("test_component", "duplicate")
        assert factory is mock_factory_2


class TestRegistryHelperFunctions:
    """Tests for the helper functions in the registry module."""

    def test_register_critic_factory(self):
        """Test registering a critic factory."""
        # Register a critic factory
        registry.register_critic_factory("test_critic", mock_factory_1)

        # Get the factory using the general getter
        factory = registry.get_factory("critic", "test_critic")
        assert factory is mock_factory_1

        # Get the factory using the specialized getter
        factory = registry.get_critic_factory("test_critic")
        assert factory is mock_factory_1

    def test_register_rule_factory(self):
        """Test registering a rule factory."""
        # Register a rule factory
        registry.register_rule_factory("test_rule", mock_factory_1)

        # Get the factory using the general getter
        factory = registry.get_factory("rule", "test_rule")
        assert factory is mock_factory_1

        # Get the factory using the specialized getter
        factory = registry.get_rule_factory("test_rule")
        assert factory is mock_factory_1

    def test_register_classifier_factory(self):
        """Test registering a classifier factory."""
        # Register a classifier factory
        registry.register_classifier_factory("test_classifier", mock_factory_1)

        # Get the factory using the general getter
        factory = registry.get_factory("classifier", "test_classifier")
        assert factory is mock_factory_1

        # Get the factory using the specialized getter
        factory = registry.get_classifier_factory("test_classifier")
        assert factory is mock_factory_1

    def test_register_model_provider_factory(self):
        """Test registering a model provider factory."""
        # Register a model provider factory
        registry.register_model_provider_factory("test_provider", mock_factory_1)

        # Get the factory using the general getter
        factory = registry.get_factory("model_provider", "test_provider")
        assert factory is mock_factory_1

        # Get the factory using the specialized getter
        factory = registry.get_model_provider_factory("test_provider")
        assert factory is mock_factory_1


class TestRegistryEdgeCases:
    """Tests for edge cases in the registry system."""

    def test_registry_empty_initially(self):
        """Test that the registry is empty for a new component type."""
        # Create a test component type that's supported but empty
        registry._factory_registry["test_empty"] = {}

        # Get factories for this component type
        factories = registry.get_available_factories("test_empty")

        # Should be an empty dict
        assert factories == {}

        # Get a specific factory
        factory = registry.get_factory("test_empty", "nonexistent")

        # Should be None
        assert factory is None

        # Register a factory
        registry.register_factory("test_empty", "test_factory", mock_factory_1)

        # Now it should be available
        factory = registry.get_factory("test_empty", "test_factory")
        assert factory is mock_factory_1

    def test_registry_internal_structure(self):
        """Test the internal structure of the registry."""
        # Check that the registry has the expected component types
        assert "critic" in registry._factory_registry
        assert "rule" in registry._factory_registry
        assert "classifier" in registry._factory_registry
        assert "retriever" in registry._factory_registry
        assert "adapter" in registry._factory_registry
        assert "model_provider" in registry._factory_registry
        assert "chain" in registry._factory_registry

        # Each component type should be a dict
        for component_type in registry._factory_registry:
            assert isinstance(registry._factory_registry[component_type], dict)


@pytest.fixture
def clean_test_registry():
    """Fixture to provide a clean registry for testing."""
    # Save the original registry
    original_registry = registry._factory_registry.copy()

    # Create a clean registry for testing
    registry._factory_registry = {
        "test_component": {},  # Include this for test cases
        "critic": {},
        "rule": {},
        "classifier": {},
        "retriever": {},
        "adapter": {},
        "model_provider": {},
        "chain": {},
    }

    # Run the test
    yield

    # Restore the original registry
    registry._factory_registry = original_registry


class TestRegistryIsolation:
    """Tests that run with a clean registry to avoid side effects."""

    def test_register_and_get_factory_isolated(self, clean_test_registry):
        """Test basic registration and retrieval in isolation."""
        # Register a factory function
        registry.register_factory("test_component", "isolated", mock_factory_1)

        # Retrieve the factory function
        factory = registry.get_factory("test_component", "isolated")

        # Check that we got the correct factory function
        assert factory is mock_factory_1

        # Other component types should be empty
        assert registry.get_factory("critic", "isolated") is None
        assert registry.get_factory("rule", "isolated") is None
