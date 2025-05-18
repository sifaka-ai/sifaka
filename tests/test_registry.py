"""
Tests for the registry module.

This module contains tests for the registry system in the Sifaka framework.
"""

from typing import Any

from sifaka.registry import (
    Registry,
    get_improver_factory,
    get_model_factory,
    get_registry,
    get_validator_factory,
    initialize_all,
    initialize_registry,
    register_improver,
    register_model,
    register_validator,
)


class TestRegistry:
    """Tests for the Registry class."""

    def test_singleton_pattern(self) -> None:
        """Test that Registry uses the singleton pattern."""
        registry1 = Registry()
        registry2 = Registry()
        assert registry1 is registry2

    def test_register_and_get_model_factory(self) -> None:
        """Test registering and retrieving a model factory."""
        registry = Registry()

        # Define a mock factory
        def mock_factory(model_name: str, **options: Any) -> str:
            return f"Mock model: {model_name}"

        # Register the factory
        registry.register_model("mock", mock_factory)

        # Retrieve the factory
        factory = registry.get_model_factory("mock")
        assert factory is not None
        assert factory("test-model") == "Mock model: test-model"

    def test_register_and_get_validator_factory(self) -> None:
        """Test registering and retrieving a validator factory."""
        registry = Registry()

        # Define a mock factory
        def mock_factory(**options: Any) -> str:
            return "Mock validator"

        # Register the factory
        registry.register_validator("mock", mock_factory)

        # Retrieve the factory
        factory = registry.get_validator_factory("mock")
        assert factory is not None
        assert factory() == "Mock validator"

    def test_register_and_get_improver_factory(self) -> None:
        """Test registering and retrieving an improver factory."""
        registry = Registry()

        # Define a mock factory
        def mock_factory(model: Any, **options: Any) -> str:
            return f"Mock improver with model: {model}"

        # Register the factory
        registry.register_improver("mock", mock_factory)

        # Retrieve the factory
        factory = registry.get_improver_factory("mock")
        assert factory is not None
        assert factory("test-model") == "Mock improver with model: test-model"

    def test_get_nonexistent_factory(self) -> None:
        """Test retrieving a nonexistent factory."""
        registry = Registry()

        # Try to retrieve a nonexistent factory
        factory = registry.get_model_factory("nonexistent")
        assert factory is None

    def test_register_duplicate_factory(self) -> None:
        """Test registering a duplicate factory."""
        registry = Registry()

        # Define mock factories
        def mock_factory1(model_name: str, **options: Any) -> str:
            return f"Mock model 1: {model_name}"

        def mock_factory2(model_name: str, **options: Any) -> str:
            return f"Mock model 2: {model_name}"

        # Register the first factory
        registry.register_model("mock", mock_factory1)

        # Register the second factory with the same name
        registry.register_model("mock", mock_factory2)

        # Retrieve the factory (should be the second one)
        factory = registry.get_model_factory("mock")
        assert factory is not None
        assert factory("test-model") == "Mock model 2: test-model"

    def test_get_all_factories(self) -> None:
        """Test retrieving all factories."""
        registry = Registry()

        # Define mock factories
        def mock_model_factory1(model_name: str, **options: Any) -> str:
            return f"Mock model 1: {model_name}"

        def mock_model_factory2(model_name: str, **options: Any) -> str:
            return f"Mock model 2: {model_name}"

        def mock_validator_factory(**options: Any) -> str:
            return "Mock validator"

        def mock_improver_factory(model: Any, **options: Any) -> str:
            return f"Mock improver with model: {model}"

        # Register the factories
        registry.register_model("mock1", mock_model_factory1)
        registry.register_model("mock2", mock_model_factory2)
        registry.register_validator("mock", mock_validator_factory)
        registry.register_improver("mock", mock_improver_factory)

        # Retrieve all factories
        model_factories = registry.get_all_model_factories()
        validator_factories = registry.get_all_validator_factories()
        improver_factories = registry.get_all_improver_factories()

        # Check the results
        assert len(model_factories) >= 2
        assert "mock1" in model_factories
        assert "mock2" in model_factories

        assert len(validator_factories) >= 1
        assert "mock" in validator_factories

        assert len(improver_factories) >= 1
        assert "mock" in improver_factories


class TestPublicAPI:
    """Tests for the public API functions."""

    def test_get_registry(self) -> None:
        """Test the get_registry function."""
        registry = get_registry()
        assert isinstance(registry, Registry)

    def test_register_model_decorator(self) -> None:
        """Test the register_model decorator."""

        # Define a mock factory
        @register_model("mock_decorator")
        def mock_factory(model_name: str, **options: Any) -> str:
            return f"Mock model: {model_name}"

        # Retrieve the factory
        factory = get_model_factory("mock_decorator")
        assert factory is not None
        assert factory("test-model") == "Mock model: test-model"

    def test_register_validator_decorator(self) -> None:
        """Test the register_validator decorator."""

        # Define a mock factory
        @register_validator("mock_decorator")
        def mock_factory(**options: Any) -> str:
            return "Mock validator"

        # Retrieve the factory
        factory = get_validator_factory("mock_decorator")
        assert factory is not None
        assert factory() == "Mock validator"

    def test_register_improver_decorator(self) -> None:
        """Test the register_improver decorator."""

        # Define a mock factory
        @register_improver("mock_decorator")
        def mock_factory(model: Any, **options: Any) -> str:
            return f"Mock improver with model: {model}"

        # Retrieve the factory
        factory = get_improver_factory("mock_decorator")
        assert factory is not None
        assert factory("test-model") == "Mock improver with model: test-model"

    def test_initialize_registry(self) -> None:
        """Test the initialize_registry function."""
        # Initialize the registry
        initialize_registry()

        # Check that some default factories are registered
        model_factories = get_registry().get_all_model_factories()
        assert len(model_factories) > 0

    def test_initialize_all(self, monkeypatch) -> None:
        """Test the initialize_all function."""
        # Mock the _initialize_component_type method to track calls
        original_method = Registry._initialize_component_type
        calls = []

        def mock_initialize_component_type(self, component_type: str) -> None:
            calls.append(component_type)
            original_method(self, component_type)

        monkeypatch.setattr(Registry, "_initialize_component_type", mock_initialize_component_type)

        # Initialize all components
        initialize_all()

        # Check that all component types were initialized
        assert "model" in calls
        assert "validator" in calls
        assert "improver" in calls
        assert "retriever" in calls
