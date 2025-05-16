"""
Tests for the registry initialization module.

These tests ensure that the registry initialization system correctly imports
modules and registers factory functions with the registry. This is a
critical part of the solution to circular import issues in the codebase.
"""

import importlib
import logging
import pytest
from unittest.mock import patch, MagicMock, call

# Import initialization functionality
from sifaka.core.initialize_registry import (
    initialize_registry,
    REGISTRY_MODULES,
    _initialized,
    reset_initialization_state,
)


@pytest.fixture
def reset_registry_state():
    """Fixture to reset the initialization state before each test."""
    # Reset the initialization state
    reset_initialization_state()

    # Run the test
    yield

    # Reset again after the test
    reset_initialization_state()


class TestInitializeRegistry:
    """Tests for the registry initialization functionality."""

    def test_initialization_modules_exist(self):
        """Test that all modules in REGISTRY_MODULES exist."""
        # All modules in REGISTRY_MODULES should be valid module paths
        for module_path in REGISTRY_MODULES:
            try:
                # Try to import the module
                mod = importlib.util.find_spec(module_path)
                assert mod is not None, f"Module {module_path} does not exist"
            except ImportError:
                # If an ImportError occurs, the module likely doesn't exist
                pytest.fail(f"Module {module_path} is not a valid module path")

    @patch("importlib.import_module")
    def test_initialize_registry_imports_modules(self, mock_import_module, reset_registry_state):
        """Test that initialize_registry imports all modules in REGISTRY_MODULES."""
        # Mock successful imports
        mock_import_module.return_value = MagicMock()

        # Call the function
        initialize_registry()

        # Check that importlib.import_module was called for each module
        for module_path in REGISTRY_MODULES:
            mock_import_module.assert_any_call(module_path)

    @patch("importlib.import_module")
    def test_initialize_registry_handles_import_errors(
        self, mock_import_module, reset_registry_state
    ):
        """Test that initialize_registry handles import errors gracefully."""

        # Mock importlib.import_module to raise ImportError for specific modules
        def side_effect(module_path):
            if module_path == "sifaka.critics.implementations.reflexion":
                raise ImportError(f"Mocked import error for {module_path}")
            return MagicMock()

        mock_import_module.side_effect = side_effect

        # Call the function
        initialize_registry()

        # The function should not raise an exception
        # It should log warnings, which we can't easily check in this test

        # Check that importlib.import_module was called for each module
        assert mock_import_module.call_count == len(REGISTRY_MODULES)

    @patch("importlib.import_module")
    def test_initialize_registry_logs_results(
        self, mock_import_module, reset_registry_state, caplog
    ):
        """Test that initialize_registry logs results correctly."""
        # Enable logging capture
        caplog.set_level(logging.INFO)

        # Mock importlib.import_module to raise ImportError for specific modules
        def side_effect(module_path):
            if module_path == "sifaka.critics.implementations.reflexion":
                raise ImportError(f"Mocked import error for {module_path}")
            return MagicMock()

        mock_import_module.side_effect = side_effect

        # Call the function
        initialize_registry()

        # Check that success and failure messages were logged
        assert "Initialized registry with" in caplog.text
        assert (
            "Failed to import" in caplog.text
            or len(REGISTRY_MODULES) - 1 == mock_import_module.call_count
        )

    @patch("importlib.import_module")
    def test_initialize_registry_uses_custom_modules(
        self, mock_import_module, reset_registry_state
    ):
        """Test that initialize_registry can use custom modules."""
        # Mock successful imports
        mock_import_module.return_value = MagicMock()

        # Custom modules list
        custom_modules = [
            "custom.module1",
            "custom.module2",
            "custom.module3",
        ]

        # Call the function with custom modules
        initialize_registry(modules=custom_modules)

        # Check that importlib.import_module was called for each custom module
        assert mock_import_module.call_count == len(custom_modules)
        for module_path in custom_modules:
            mock_import_module.assert_any_call(module_path)

    @patch("importlib.import_module")
    def test_initialize_registry_skips_if_already_initialized(
        self, mock_import_module, reset_registry_state
    ):
        """Test that initialize_registry skips if already initialized."""
        # Mock successful imports
        mock_import_module.return_value = MagicMock()

        # Call the function first time
        initialize_registry()

        # Reset the mock to check if it's called again
        mock_import_module.reset_mock()

        # Call the function second time
        initialize_registry()

        # Check that importlib.import_module was not called
        mock_import_module.assert_not_called()

        # But it should be called if we force it
        initialize_registry(force=True)
        assert mock_import_module.call_count > 0

    @patch("importlib.import_module")
    def test_initialize_registry_skips_duplicate_modules(
        self, mock_import_module, reset_registry_state
    ):
        """Test that initialize_registry skips duplicate modules."""
        # Mock successful imports
        mock_import_module.return_value = MagicMock()

        # Custom modules list with duplicates
        custom_modules = [
            "custom.module1",
            "custom.module1",  # Duplicate
            "custom.module2",
            "custom.module2",  # Duplicate
        ]

        # Call the function with custom modules
        initialize_registry(modules=custom_modules)

        # Check that importlib.import_module was called only once for each unique module
        assert mock_import_module.call_count == 2
        mock_import_module.assert_has_calls(
            [
                call("custom.module1"),
                call("custom.module2"),
            ],
            any_order=True,
        )


class TestIntegrationWithRegistry:
    """Tests for the integration between registry initialization and registry functionality."""

    @patch("importlib.import_module")
    @patch("sifaka.core.registry.register_factory")
    def test_initialize_registry_registers_factories(
        self, mock_register_factory, mock_import_module, reset_registry_state
    ):
        """Test that imported modules register factory functions."""
        # Mock module with register_factory calls
        mock_module = MagicMock()

        def mock_import_side_effect(module_path):
            """Simulate a module that registers factory functions."""
            # When the module is imported, it should register its factory functions
            from sifaka.core import registry

            # We'll simulate this behavior by calling register_factory directly
            registry.register_factory("critic", "test_critic", lambda: "test_critic_result")
            registry.register_factory("rule", "test_rule", lambda: "test_rule_result")

            return mock_module

        # Set up the mock to simulate module imports that register factories
        mock_import_module.side_effect = mock_import_side_effect

        # Call the function with a single test module
        initialize_registry(modules=["test.module"])

        # Check that register_factory was called
        assert mock_register_factory.call_count >= 2

        # Check that the first call was for the critic
        critic_call = any(
            call_args[0][0] == "critic" and call_args[0][1] == "test_critic"
            for call_args in mock_register_factory.call_args_list
        )
        assert critic_call, "Expected critic registration not found"

        # Check that the second call was for the rule
        rule_call = any(
            call_args[0][0] == "rule" and call_args[0][1] == "test_rule"
            for call_args in mock_register_factory.call_args_list
        )
        assert rule_call, "Expected rule registration not found"
