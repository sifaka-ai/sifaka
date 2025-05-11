"""
Tests for standardized adapters in the Sifaka codebase.

This module contains tests to verify that adapters follow the standardized
state management pattern and have consistent lifecycle management.
"""

import pytest
from pydantic import BaseModel

from sifaka.adapters.base import BaseAdapter, create_adapter_state
from sifaka.adapters.guardrails.adapter import GuardrailsAdapter, GuardrailsRule, GuardrailsValidatorAdapter
from sifaka.adapters.pydantic_ai.adapter import SifakaPydanticAdapter
from sifaka.adapters.pydantic_ai.factory import create_pydantic_adapter, create_pydantic_adapter_with_critic
from sifaka.core.factories import create_adapter
from sifaka.rules.formatting.length import create_length_rule
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

    def test_guardrails_adapter_state_manager(self):
        """Test that GuardrailsAdapter uses the standardized state manager."""
        # Skip if guardrails is not installed
        try:
            from guardrails.validators import RegexMatch
            validator = RegexMatch(regex=r"\d{3}-\d{3}-\d{4}")
        except ImportError:
            pytest.skip("Guardrails not installed")
            
        # Create a guardrails adapter
        adapter = GuardrailsAdapter(validator)
        
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
        
        # Check that adaptee was set correctly
        assert state.adaptee == validator

    def test_guardrails_rule_state_manager(self):
        """Test that GuardrailsRule uses the standardized state manager."""
        # Skip if guardrails is not installed
        try:
            from guardrails.validators import RegexMatch
            validator = RegexMatch(regex=r"\d{3}-\d{3}-\d{4}")
        except ImportError:
            pytest.skip("Guardrails not installed")
            
        # Create a guardrails rule
        rule = GuardrailsRule(validator)
        
        # Check that it has a _state_manager attribute
        assert hasattr(rule, "_state_manager")
        
        # Check that _state_manager is an instance of StateManager
        assert isinstance(rule._state_manager, StateManager)
        
        # Check that _state_manager was created with create_adapter_state
        state = rule._state_manager.get_state()
        assert hasattr(state, "adaptee")
        assert hasattr(state, "initialized")
        assert hasattr(state, "execution_count")
        assert hasattr(state, "error_count")
        assert hasattr(state, "last_execution_time")
        assert hasattr(state, "avg_execution_time")
        assert hasattr(state, "cache")
        assert hasattr(state, "config_cache")
        
        # Check that adaptee was set correctly
        assert state.adaptee == validator

    def test_pydantic_adapter_state_manager(self):
        """Test that SifakaPydanticAdapter uses the standardized state manager."""
        # Create a simple model
        class TestModel(BaseModel):
            content: str
            
        # Create rules
        rules = [create_length_rule(min_chars=10, max_chars=100)]
        
        # Create a pydantic adapter
        adapter = SifakaPydanticAdapter(rules=rules, output_model=TestModel)
        
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
        
        # Check that config_cache was set correctly
        assert "rules" in state.config_cache
        assert "output_model" in state.config_cache
        assert "critic" in state.config_cache

    def test_factory_functions(self):
        """Test that factory functions create adapters with standardized state management."""
        # Skip if guardrails is not installed
        try:
            from guardrails.validators import RegexMatch
            validator = RegexMatch(regex=r"\d{3}-\d{3}-\d{4}")
        except ImportError:
            pytest.skip("Guardrails not installed")
            
        # Create a guardrails adapter using the factory function
        adapter = create_adapter(
            adapter_type="guardrails",
            adaptee=validator,
            name="test_adapter",
            description="Test adapter"
        )
        
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
        
        # Check that adaptee was set correctly
        assert state.adaptee == validator
        
        # Check that metadata was set correctly
        assert adapter._state_manager.get_metadata("name") == "test_adapter"
        assert adapter._state_manager.get_metadata("description") == "Test adapter"
