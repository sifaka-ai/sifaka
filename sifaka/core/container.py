"""SifakaContainer: Dependency injection container for Sifaka.

This module implements a dependency injection container that resolves circular
dependencies and provides clean separation of concerns.

Key features:
- Lazy loading of graph nodes to avoid circular imports
- Interface-based dependency resolution
- Plugin registry integration
- Memory management and cleanup
"""

from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
import importlib
from datetime import datetime

from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from sifaka.graph.dependencies import SifakaDependencies
    from sifaka.core.thought import SifakaThought

logger = get_logger(__name__)


class NodeInterface(ABC):
    """Abstract interface for graph nodes."""
    
    @abstractmethod
    async def run(self, state: "SifakaThought", deps: "SifakaDependencies") -> "SifakaThought":
        """Execute the node logic."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the node name."""
        pass


class SifakaContainer:
    """Dependency injection container for Sifaka components.
    
    This container manages the lifecycle and dependencies of all Sifaka components,
    providing lazy loading and interface-based resolution to avoid circular dependencies.
    
    Features:
    - Lazy loading of graph nodes
    - Plugin registry for extensibility
    - Memory management and cleanup
    - Interface-based dependency resolution
    """
    
    def __init__(self):
        """Initialize the container."""
        self._nodes: Dict[str, Type[NodeInterface]] = {}
        self._node_instances: Dict[str, NodeInterface] = {}
        self._plugins: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._created_at = datetime.now()
        
        # Register default nodes lazily
        self._register_default_nodes()
        
        logger.info("SifakaContainer initialized")
    
    def _register_default_nodes(self) -> None:
        """Register default graph nodes with lazy loading."""
        # Register node types without importing them yet
        self._nodes = {
            "generate": None,  # Will be loaded lazily
            "validate": None,  # Will be loaded lazily
            "critique": None,  # Will be loaded lazily
        }
    
    def get_node(self, node_name: str) -> Type[NodeInterface]:
        """Get a node class by name with lazy loading.
        
        Args:
            node_name: Name of the node to retrieve
            
        Returns:
            Node class
            
        Raises:
            ValueError: If node is not registered
        """
        if node_name not in self._nodes:
            raise ValueError(f"Node '{node_name}' is not registered")
        
        # Lazy load the node if not already loaded
        if self._nodes[node_name] is None:
            self._nodes[node_name] = self._load_node(node_name)
        
        return self._nodes[node_name]
    
    def get_node_instance(self, node_name: str) -> NodeInterface:
        """Get a node instance by name (singleton pattern).
        
        Args:
            node_name: Name of the node to retrieve
            
        Returns:
            Node instance
        """
        if node_name not in self._node_instances:
            node_class = self.get_node(node_name)
            self._node_instances[node_name] = node_class()
            logger.debug(f"Created node instance: {node_name}")
        
        return self._node_instances[node_name]
    
    def _load_node(self, node_name: str) -> Type[NodeInterface]:
        """Dynamically load a node class to avoid circular imports.
        
        Args:
            node_name: Name of the node to load
            
        Returns:
            Loaded node class
        """
        node_mappings = {
            "generate": ("sifaka.graph.nodes", "GenerateNode"),
            "validate": ("sifaka.graph.nodes", "ValidateNode"),
            "critique": ("sifaka.graph.nodes", "CritiqueNode"),
        }
        
        if node_name not in node_mappings:
            raise ValueError(f"Unknown node: {node_name}")
        
        module_name, class_name = node_mappings[node_name]
        
        try:
            module = importlib.import_module(module_name)
            node_class = getattr(module, class_name)
            logger.debug(f"Loaded node: {node_name} from {module_name}.{class_name}")
            return node_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load node {node_name}: {e}")
            raise ValueError(f"Failed to load node {node_name}: {e}") from e
    
    def register_node(self, name: str, node_class: Type[NodeInterface]) -> None:
        """Register a custom node.
        
        Args:
            name: Name to register the node under
            node_class: Node class to register
        """
        self._nodes[name] = node_class
        logger.info(f"Registered custom node: {name}")
    
    def register_plugin(self, name: str, plugin: Any) -> None:
        """Register a plugin component.
        
        Args:
            name: Name to register the plugin under
            plugin: Plugin instance to register
        """
        self._plugins[name] = plugin
        logger.info(f"Registered plugin: {name}")
    
    def get_plugin(self, name: str) -> Any:
        """Get a registered plugin.
        
        Args:
            name: Name of the plugin to retrieve
            
        Returns:
            Plugin instance
            
        Raises:
            ValueError: If plugin is not registered
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' is not registered")
        
        return self._plugins[name]
    
    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton instance.
        
        Args:
            name: Name to register the singleton under
            instance: Instance to register
        """
        self._singletons[name] = instance
        logger.info(f"Registered singleton: {name}")
    
    def get_singleton(self, name: str) -> Any:
        """Get a registered singleton.
        
        Args:
            name: Name of the singleton to retrieve
            
        Returns:
            Singleton instance
            
        Raises:
            ValueError: If singleton is not registered
        """
        if name not in self._singletons:
            raise ValueError(f"Singleton '{name}' is not registered")
        
        return self._singletons[name]
    
    def get_all_nodes(self) -> List[str]:
        """Get names of all registered nodes.
        
        Returns:
            List of node names
        """
        return list(self._nodes.keys())
    
    def get_all_plugins(self) -> List[str]:
        """Get names of all registered plugins.
        
        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())
    
    def clear_instances(self) -> None:
        """Clear all cached instances (useful for testing)."""
        self._node_instances.clear()
        logger.debug("Cleared all cached node instances")
    
    def get_container_stats(self) -> Dict[str, Any]:
        """Get container statistics.
        
        Returns:
            Dictionary with container statistics
        """
        return {
            "created_at": self._created_at,
            "registered_nodes": len(self._nodes),
            "loaded_nodes": sum(1 for node in self._nodes.values() if node is not None),
            "node_instances": len(self._node_instances),
            "registered_plugins": len(self._plugins),
            "registered_singletons": len(self._singletons),
            "node_names": list(self._nodes.keys()),
            "plugin_names": list(self._plugins.keys()),
            "singleton_names": list(self._singletons.keys())
        }
    
    def cleanup(self) -> None:
        """Clean up container resources."""
        self._node_instances.clear()
        self._plugins.clear()
        self._singletons.clear()
        logger.info("SifakaContainer cleaned up")
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with cleanup."""
        self.cleanup()


# Global container instance
_container: Optional[SifakaContainer] = None


def get_container() -> SifakaContainer:
    """Get the global container instance.
    
    Returns:
        Global SifakaContainer instance
    """
    global _container
    if _container is None:
        _container = SifakaContainer()
    return _container


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _container
    if _container:
        _container.cleanup()
    _container = None
