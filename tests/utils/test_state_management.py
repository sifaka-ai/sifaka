"""
Tests for state management patterns in the Sifaka framework.

This module contains tests that verify the proper usage of state management patterns
across the Sifaka framework, including factory function usage, initialization patterns,
and state access.
"""

import unittest
from unittest.mock import MagicMock, patch

from sifaka.utils.state import (
    StateManager,
    create_state_manager,
    create_classifier_state,
    create_manager_state,
    create_chain_state,
    create_adapter_state,
    create_engine_state,
    ComponentState,
    ClassifierState,
    ManagerState,
    ChainState,
    AdapterState,
)


class TestStateFactoryFunctions(unittest.TestCase):
    """Tests for state factory functions."""

    def test_create_state_manager_returns_state_manager(self):
        """Test that create_state_manager returns a StateManager instance."""
        manager = create_state_manager(ComponentState)
        self.assertIsInstance(manager, StateManager)

    def test_create_state_manager_initializes_with_component_state(self):
        """Test that create_state_manager initializes with the provided state."""
        manager = create_state_manager(ComponentState, initialized=True)
        self.assertTrue(manager.get("initialized"))

    def test_specialized_factory_functions_create_correct_state(self):
        """Test that specialized factory functions create the correct state."""
        # Test classifier state
        classifier_manager = create_classifier_state(model="test_model")
        self.assertEqual(classifier_manager.get("model"), "test_model")

        # Test manager state
        manager_manager = create_manager_state(resources=["resource1", "resource2"])
        self.assertEqual(manager_manager.get("resources"), ["resource1", "resource2"])

        # Test chain state
        chain_manager = create_chain_state(model="test_model")
        self.assertEqual(chain_manager.get("model"), "test_model")

        # Test adapter state
        adapter_manager = create_adapter_state(adaptee="test_adaptee")
        self.assertEqual(adapter_manager.get("adaptee"), "test_adaptee")

        # Test engine state
        engine_manager = create_engine_state(cache={})
        self.assertEqual(engine_manager.get("cache"), {})


class TestStateManagerMethods(unittest.TestCase):
    """Tests for StateManager methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = create_state_manager(ComponentState)

    def test_update_adds_to_state(self):
        """Test that update adds to state."""
        self.manager.update("test_key", "test_value")
        self.assertEqual(self.manager.get("test_key"), "test_value")

    def test_get_returns_default_value_if_key_not_in_state(self):
        """Test that get returns the default value if the key is not in state."""
        self.assertEqual(self.manager.get("non_existent_key", "default"), "default")

    def test_set_metadata_adds_to_metadata(self):
        """Test that set_metadata adds to metadata."""
        self.manager.set_metadata("test_key", "test_value")
        self.assertEqual(self.manager.get_metadata("test_key"), "test_value")

    def test_get_metadata_returns_default_value_if_key_not_in_metadata(self):
        """Test that get_metadata returns the default value if the key is not in metadata."""
        self.assertEqual(self.manager.get_metadata("non_existent_key", "default"), "default")

    def test_rollback_reverts_to_previous_state(self):
        """Test that rollback reverts to the previous state."""
        # Set initial state
        self.manager.update("test_key", "initial_value")
        # Update state
        self.manager.update("test_key", "updated_value")
        # Rollback
        self.manager.rollback()
        # Check that state is reverted
        self.assertEqual(self.manager.get("test_key"), "initial_value")


class TestComponentWithStateManager:
    """A test component that uses a state manager."""

    def __init__(self):
        """Initialize the component."""
        self._state_manager = create_manager_state()
        self._initialize_state()

    def _initialize_state(self):
        """Initialize the component state."""
        self._state_manager.update("initialized", True)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", "test_component")

    def is_initialized(self):
        """Check if the component is initialized."""
        return self._state_manager.get("initialized", False)

    def get_cache(self):
        """Get the component cache."""
        return self._state_manager.get("cache", {})

    def get_component_type(self):
        """Get the component type."""
        return self._state_manager.get_metadata("component_type")


class TestComponentWithStateManagerUsage(unittest.TestCase):
    """Tests for components that use a state manager."""

    def test_component_initializes_state_correctly(self):
        """Test that a component initializes state correctly."""
        component = TestComponentWithStateManager()
        self.assertTrue(component.is_initialized())
        self.assertEqual(component.get_cache(), {})
        self.assertEqual(component.get_component_type(), "test_component")


if __name__ == "__main__":
    unittest.main()
