"""Tests for SifakaContainer dependency injection system."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from sifaka.core.container import SifakaContainer, NodeInterface, get_container, reset_container


class MockNode(NodeInterface):
    """Mock node for testing."""
    
    def __init__(self, name: str = "mock_node"):
        self._name = name
    
    async def run(self, state, deps):
        return state
    
    @property
    def name(self) -> str:
        return self._name


class TestSifakaContainer:
    """Test cases for SifakaContainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.container = SifakaContainer()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.container.cleanup()
    
    def test_container_initialization(self):
        """Test container initializes correctly."""
        assert isinstance(self.container._created_at, datetime)
        assert len(self.container._nodes) > 0  # Should have default nodes
        assert "generate" in self.container._nodes
        assert "validate" in self.container._nodes
        assert "critique" in self.container._nodes
    
    def test_register_node(self):
        """Test node registration."""
        mock_node = MockNode("test_node")
        self.container.register_node("test", MockNode)
        
        assert "test" in self.container._nodes
        assert self.container._nodes["test"] == MockNode
    
    def test_get_node_instance(self):
        """Test node instance retrieval (singleton pattern)."""
        self.container.register_node("test", MockNode)
        
        instance1 = self.container.get_node_instance("test")
        instance2 = self.container.get_node_instance("test")
        
        assert isinstance(instance1, MockNode)
        assert instance1 is instance2  # Should be same instance (singleton)
    
    def test_register_plugin(self):
        """Test plugin registration."""
        plugin = {"name": "test_plugin", "version": "1.0.0"}
        self.container.register_plugin("test_plugin", plugin)
        
        retrieved_plugin = self.container.get_plugin("test_plugin")
        assert retrieved_plugin == plugin
    
    def test_get_plugin_not_found(self):
        """Test getting non-existent plugin raises error."""
        with pytest.raises(ValueError, match="Plugin 'nonexistent' is not registered"):
            self.container.get_plugin("nonexistent")
    
    def test_register_singleton(self):
        """Test singleton registration."""
        singleton_value = "test_singleton"
        self.container.register_singleton("test", singleton_value)
        
        retrieved = self.container.get_singleton("test")
        assert retrieved == singleton_value
    
    def test_get_singleton_not_found(self):
        """Test getting non-existent singleton raises error."""
        with pytest.raises(ValueError, match="Singleton 'nonexistent' is not registered"):
            self.container.get_singleton("nonexistent")
    
    def test_get_all_nodes(self):
        """Test getting all registered nodes."""
        nodes = self.container.get_all_nodes()
        assert isinstance(nodes, list)
        assert "generate" in nodes
        assert "validate" in nodes
        assert "critique" in nodes
    
    def test_get_all_plugins(self):
        """Test getting all registered plugins."""
        self.container.register_plugin("plugin1", {"name": "plugin1"})
        self.container.register_plugin("plugin2", {"name": "plugin2"})
        
        plugins = self.container.get_all_plugins()
        assert "plugin1" in plugins
        assert "plugin2" in plugins
    
    def test_clear_instances(self):
        """Test clearing cached instances."""
        self.container.register_node("test", MockNode)
        instance = self.container.get_node_instance("test")
        
        assert "test" in self.container._node_instances
        
        self.container.clear_instances()
        assert len(self.container._node_instances) == 0
    
    def test_get_container_stats(self):
        """Test container statistics."""
        self.container.register_plugin("test_plugin", {"name": "test"})
        self.container.register_singleton("test_singleton", "value")
        
        stats = self.container.get_container_stats()
        
        assert "created_at" in stats
        assert "registered_nodes" in stats
        assert "registered_plugins" in stats
        assert "registered_singletons" in stats
        assert stats["registered_plugins"] >= 1
        assert stats["registered_singletons"] >= 1
    
    def test_context_manager(self):
        """Test container as context manager."""
        with SifakaContainer() as container:
            container.register_plugin("test", {"name": "test"})
            assert "test" in container._plugins
        
        # After context exit, container should be cleaned up
        assert len(container._plugins) == 0
    
    def test_cleanup(self):
        """Test container cleanup."""
        self.container.register_plugin("test_plugin", {"name": "test"})
        self.container.register_singleton("test_singleton", "value")
        self.container.register_node("test_node", MockNode)
        self.container.get_node_instance("test_node")  # Create instance
        
        assert len(self.container._plugins) > 0
        assert len(self.container._singletons) > 0
        assert len(self.container._node_instances) > 0
        
        self.container.cleanup()
        
        assert len(self.container._node_instances) == 0
        assert len(self.container._plugins) == 0
        assert len(self.container._singletons) == 0


class TestGlobalContainer:
    """Test cases for global container functions."""
    
    def teardown_method(self):
        """Clean up global container after each test."""
        reset_container()
    
    def test_get_container_singleton(self):
        """Test global container is singleton."""
        container1 = get_container()
        container2 = get_container()
        
        assert container1 is container2
    
    def test_reset_container(self):
        """Test resetting global container."""
        container1 = get_container()
        container1.register_plugin("test", {"name": "test"})
        
        reset_container()
        
        container2 = get_container()
        assert container1 is not container2
        assert len(container2._plugins) == 0
    
    @patch('sifaka.core.container.importlib.import_module')
    def test_lazy_node_loading_failure(self, mock_import):
        """Test handling of node loading failures."""
        mock_import.side_effect = ImportError("Module not found")
        
        container = SifakaContainer()
        
        with pytest.raises(ValueError, match="Failed to load node generate"):
            container.get_node("generate")
    
    def test_node_interface_abstract_methods(self):
        """Test NodeInterface abstract methods."""
        with pytest.raises(TypeError):
            # Should not be able to instantiate abstract class
            NodeInterface()
    
    def test_mock_node_implementation(self):
        """Test MockNode implements NodeInterface correctly."""
        node = MockNode("test")
        assert node.name == "test"
        
        # Test async run method
        import asyncio
        
        async def test_run():
            state = {"test": "state"}
            result = await node.run(state, None)
            return result
        
        result = asyncio.run(test_run())
        assert result == {"test": "state"}
