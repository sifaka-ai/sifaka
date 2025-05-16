"""
Integration tests for the registry system.

These tests ensure that the registry system correctly handles factory function
registration and retrieval in real usage scenarios, particularly focusing on
how it handles circular dependencies between modules.
"""

import os
import pytest
import importlib
from unittest.mock import patch, MagicMock

# Import registry functionality to test against
from sifaka.core import registry
from sifaka.core.initialize_registry import initialize_registry, reset_initialization_state


@pytest.fixture
def clean_environment():
    """Fixture to reset environment variables that might affect tests."""
    # Save original environment variable
    original_value = os.environ.get("SIFAKA_SKIP_REGISTRY_INIT", "")

    # Set environment variable to skip automatic initialization
    os.environ["SIFAKA_SKIP_REGISTRY_INIT"] = "true"

    # Run the test
    yield

    # Restore the original environment variable
    if original_value:
        os.environ["SIFAKA_SKIP_REGISTRY_INIT"] = original_value
    else:
        del os.environ["SIFAKA_SKIP_REGISTRY_INIT"]


@pytest.fixture
def reset_registry_and_initialization():
    """Fixture to reset registry and initialization state."""
    # Save original registry state
    original_registry = registry._factory_registry.copy()

    # Create a clean registry for testing with standard component types
    registry._factory_registry = {
        "critic": {},
        "rule": {},
        "classifier": {},
        "retriever": {},
        "adapter": {},
        "model_provider": {},
        "chain": {},
        "test_component": {},
    }

    # Reset initialization flag
    reset_initialization_state()

    # Run the test
    yield

    # Restore original registry state
    registry._factory_registry = original_registry

    # Reset initialization flag again to avoid affecting other tests
    reset_initialization_state()


class TestCircularImportHandling:
    """Tests specifically for handling circular imports."""

    @patch("importlib.import_module")
    def test_factory_retrieval_without_import(
        self, mock_import_module, clean_environment, reset_registry_and_initialization
    ):
        """Test retrieving factory functions without importing implementation modules."""
        # Register a factory function directly
        registry.register_factory(
            "critic", "test_critic", lambda **kwargs: f"test_critic: {kwargs}"
        )

        # Get the factory function using the helper function
        factory = registry.get_critic_factory("test_critic")

        # Verify it works
        assert factory is not None
        assert factory(param="value") == "test_critic: {'param': 'value'}"

        # Verify that no imports were made
        mock_import_module.assert_not_called()

    def test_registry_reduces_circular_imports(
        self, clean_environment, reset_registry_and_initialization
    ):
        """Test that the registry system reduces the need for circular imports."""
        # In a real scenario, we would have:
        # 1. Factory module that imports implementation modules
        # 2. Implementation modules that import factory functions

        # With the registry, implementation modules register their factories
        # and factory modules get factories from the registry without imports

        # Simulate an implementation module registering its factory
        registry.register_factory("critic", "reflexion", lambda **kwargs: f"reflexion: {kwargs}")

        # Simulate a factory module getting the factory without importing
        factory = registry.get_critic_factory("reflexion")

        # Verify it works
        assert factory is not None
        assert factory(param="value") == "reflexion: {'param': 'value'}"


class TestIntegrationWithImportedModules:
    """Tests for integration with actual imported modules."""

    @pytest.mark.skip(
        reason="This test requires actual modules to be imported, which may cause circular imports in tests"
    )
    def test_initialize_with_actual_modules(
        self, clean_environment, reset_registry_and_initialization
    ):
        """Test initialization with actual modules."""
        # Initialize with a subset of the actual modules
        test_modules = [
            "sifaka.critics.implementations.prompt",
            "sifaka.rules.formatting.length",
        ]

        initialize_registry(modules=test_modules)

        # Check that factories were registered
        critic_factory = registry.get_critic_factory("prompt")
        rule_factory = registry.get_rule_factory("length")

        # These should be defined if the modules were properly imported
        assert critic_factory is not None
        assert rule_factory is not None


class TestRegistryUsagePatterns:
    """Tests for common usage patterns of the registry system."""

    def test_factory_pattern_usage(self, clean_environment, reset_registry_and_initialization):
        """Test using the registry in a factory pattern."""
        # Register factory functions
        registry.register_factory("critic", "prompt", lambda **kwargs: f"prompt: {kwargs}")
        registry.register_factory("critic", "reflexion", lambda **kwargs: f"reflexion: {kwargs}")

        # Create a factory function that uses the registry
        def create_critic(critic_type, **kwargs):
            factory = registry.get_critic_factory(critic_type)
            if factory is None:
                raise ValueError(f"Unknown critic type: {critic_type}")
            return factory(**kwargs)

        # Use the factory function
        critic1 = create_critic("prompt", param="value1")
        critic2 = create_critic("reflexion", param="value2")

        # Verify the results
        assert critic1 == "prompt: {'param': 'value1'}"
        assert critic2 == "reflexion: {'param': 'value2'}"

        # Verify error handling
        with pytest.raises(ValueError) as excinfo:
            create_critic("nonexistent")
        assert "Unknown critic type: nonexistent" in str(excinfo.value)

    def test_dependency_injection_with_registry(
        self, clean_environment, reset_registry_and_initialization
    ):
        """Test combining dependency injection with the registry."""
        # Register factory functions
        registry.register_factory("model_provider", "openai", lambda **kwargs: f"openai: {kwargs}")
        registry.register_factory(
            "model_provider", "anthropic", lambda **kwargs: f"anthropic: {kwargs}"
        )

        # Create a component that uses a model provider
        class Critic:
            def __init__(self, model_provider=None, **kwargs):
                self.model_provider = model_provider
                self.kwargs = kwargs

                # If no model provider is provided, get the default one from the registry
                if self.model_provider is None:
                    factory = registry.get_model_provider_factory("openai")
                    if factory:
                        self.model_provider = factory()

            def generate(self, prompt):
                return f"{self.model_provider} - {prompt} - {self.kwargs}"

        # Create a critic with explicit model provider
        critic1 = Critic(model_provider="explicit_provider", param="value1")

        # Create a critic with registry-based model provider
        critic2 = Critic(param="value2")

        # Verify the results
        assert critic1.generate("test") == "explicit_provider - test - {'param': 'value1'}"
        assert critic2.generate("test") == "openai: {} - test - {'param': 'value2'}"

    def test_registry_as_service_locator(
        self, clean_environment, reset_registry_and_initialization
    ):
        """Test using the registry as a service locator."""
        # Register services as factory functions
        registry.register_factory("model_provider", "openai", lambda: "openai_service")
        registry.register_factory("critic", "prompt", lambda: "prompt_service")
        registry.register_factory("rule", "length", lambda: "length_service")

        # Create a function that uses the registry to locate services
        def get_service(service_type, service_name):
            factory = registry.get_factory(service_type, service_name)
            if factory is None:
                raise ValueError(f"Service not found: {service_type}/{service_name}")
            return factory()

        # Get services
        openai_service = get_service("model_provider", "openai")
        prompt_service = get_service("critic", "prompt")
        length_service = get_service("rule", "length")

        # Verify the results
        assert openai_service == "openai_service"
        assert prompt_service == "prompt_service"
        assert length_service == "length_service"

    def test_dynamic_registration(self, clean_environment, reset_registry_and_initialization):
        """Test dynamic registration of factories at runtime."""

        # Create a dynamically loaded module
        class DynamicModule:
            @staticmethod
            def register_factories():
                registry.register_factory(
                    "critic", "dynamic", lambda **kwargs: f"dynamic: {kwargs}"
                )
                registry.register_factory(
                    "rule", "dynamic", lambda **kwargs: f"dynamic_rule: {kwargs}"
                )

        # Register factories from the dynamic module
        DynamicModule.register_factories()

        # Get the factories
        critic_factory = registry.get_critic_factory("dynamic")
        rule_factory = registry.get_rule_factory("dynamic")

        # Verify the results
        assert critic_factory is not None
        assert rule_factory is not None
        assert critic_factory(param="value") == "dynamic: {'param': 'value'}"
        assert rule_factory(param="value") == "dynamic_rule: {'param': 'value'}"
