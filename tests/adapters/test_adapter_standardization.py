"""
Tests for standardized adapters in the Sifaka codebase.

This module contains simple tests to verify that adapters follow the standardized
state management pattern and have consistent lifecycle management.
"""

import pytest
from pydantic import BaseModel

from sifaka.adapters.base import BaseAdapter, create_adapter_state
from sifaka.utils.state import StateManager


class TestAdapterStandardization:
    """Test suite for adapter standardization."""

    def test_base_adapter_state_manager(self):
        """Test that BaseAdapter uses the standardized state manager."""
        # Create a simple adapter
        adapter = BaseAdapter()
        
        # Check that it has a _state_manager attribute
        assert hasattr(adapter, "_state_manager")
        
        # Check that _state_manager is an instance of StateManager
        assert isinstance(adapter._state_manager, StateManager)
        
        # Check that _state_manager was created with create_adapter_state
        state = adapter._state_manager.get_state()
        assert hasattr(state, "adaptee")
        assert hasattr(state, "initialized")
        assert hasattr(state, "execution_count")
        assert hasattr(state, "error_count")
        assert hasattr(state, "last_execution_time")
        assert hasattr(state, "avg_execution_time")
        assert hasattr(state, "cache")
        assert hasattr(state, "config_cache")
