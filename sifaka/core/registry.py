"""SifakaRegistry: Plugin registry system for Sifaka.

This module implements a plugin registry system that enables dynamic registration
and discovery of critics, validators, and other components.

Key features:
- Dynamic critic and validator registration
- Plugin discovery and validation
- Version compatibility checking
- Plugin metadata management
"""

from typing import Any, Dict, List, Optional, Type, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import importlib
import inspect
import warnings

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a registered plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: str  # "critic", "validator", "retriever", etc.
    dependencies: List[str]
    min_sifaka_version: str
    registered_at: datetime
    instance: Any


class CriticInterface(ABC):
    """Abstract interface for critics."""
    
    @abstractmethod
    async def critique(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide critique feedback for the given text."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the critic name."""
        pass


class ValidatorInterface(ABC):
    """Abstract interface for validators."""
    
    @abstractmethod
    async def validate(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the given text."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the validator name."""
        pass


class SifakaRegistry:
    """Plugin registry for Sifaka components.
    
    This registry manages the registration and discovery of plugins including
    critics, validators, retrievers, and other extensible components.
    
    Features:
    - Type-safe plugin registration
    - Automatic plugin discovery
    - Version compatibility checking
    - Plugin metadata management
    - Conflict resolution
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._plugins: Dict[str, PluginMetadata] = {}
        self._critics: Dict[str, CriticInterface] = {}
        self._validators: Dict[str, ValidatorInterface] = {}
        self._factories: Dict[str, Callable] = {}
        self._created_at = datetime.now()
        
        # Register built-in components
        self._register_builtin_components()
        
        logger.info("SifakaRegistry initialized")
    
    def _register_builtin_components(self) -> None:
        """Register built-in critics and validators."""
        # This will be populated with built-in components
        builtin_critics = [
            "reflexion", "constitutional", "self_refine", "n_critics",
            "self_consistency", "prompt", "meta_rewarding", "self_rag"
        ]
        
        for critic_name in builtin_critics:
            self._register_builtin_critic(critic_name)
        
        logger.debug(f"Registered {len(builtin_critics)} built-in critics")
    
    def _register_builtin_critic(self, name: str) -> None:
        """Register a built-in critic with lazy loading."""
        metadata = PluginMetadata(
            name=name,
            version="1.0.0",
            description=f"Built-in {name} critic",
            author="Sifaka Team",
            plugin_type="critic",
            dependencies=[],
            min_sifaka_version="0.1.0",
            registered_at=datetime.now(),
            instance=None  # Will be loaded lazily
        )
        self._plugins[f"critic:{name}"] = metadata
    
    def register_critic(
        self, 
        name: str, 
        critic: Union[CriticInterface, Type[CriticInterface], Callable],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a critic plugin.
        
        Args:
            name: Name to register the critic under
            critic: Critic instance, class, or factory function
            metadata: Optional metadata for the plugin
        """
        if metadata is None:
            metadata = {}
        
        # Validate the critic
        if not self._validate_critic(critic):
            raise ValueError(f"Invalid critic: {name}")
        
        # Create metadata
        plugin_metadata = PluginMetadata(
            name=name,
            version=metadata.get("version", "1.0.0"),
            description=metadata.get("description", f"Custom critic: {name}"),
            author=metadata.get("author", "Unknown"),
            plugin_type="critic",
            dependencies=metadata.get("dependencies", []),
            min_sifaka_version=metadata.get("min_sifaka_version", "0.1.0"),
            registered_at=datetime.now(),
            instance=critic
        )
        
        plugin_key = f"critic:{name}"
        
        # Check for conflicts
        if plugin_key in self._plugins:
            logger.warning(f"Overriding existing critic: {name}")
        
        self._plugins[plugin_key] = plugin_metadata
        
        # Store in critics dict for quick access
        if isinstance(critic, CriticInterface):
            self._critics[name] = critic
        
        logger.info(f"Registered critic: {name}")
    
    def register_validator(
        self, 
        name: str, 
        validator: Union[ValidatorInterface, Type[ValidatorInterface], Callable],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a validator plugin.
        
        Args:
            name: Name to register the validator under
            validator: Validator instance, class, or factory function
            metadata: Optional metadata for the plugin
        """
        if metadata is None:
            metadata = {}
        
        # Validate the validator
        if not self._validate_validator(validator):
            raise ValueError(f"Invalid validator: {name}")
        
        # Create metadata
        plugin_metadata = PluginMetadata(
            name=name,
            version=metadata.get("version", "1.0.0"),
            description=metadata.get("description", f"Custom validator: {name}"),
            author=metadata.get("author", "Unknown"),
            plugin_type="validator",
            dependencies=metadata.get("dependencies", []),
            min_sifaka_version=metadata.get("min_sifaka_version", "0.1.0"),
            registered_at=datetime.now(),
            instance=validator
        )
        
        plugin_key = f"validator:{name}"
        
        # Check for conflicts
        if plugin_key in self._plugins:
            logger.warning(f"Overriding existing validator: {name}")
        
        self._plugins[plugin_key] = plugin_metadata
        
        # Store in validators dict for quick access
        if isinstance(validator, ValidatorInterface):
            self._validators[name] = validator
        
        logger.info(f"Registered validator: {name}")
    
    def register_factory(self, name: str, factory: Callable, plugin_type: str) -> None:
        """Register a factory function for creating plugin instances.
        
        Args:
            name: Name to register the factory under
            factory: Factory function
            plugin_type: Type of plugin this factory creates
        """
        factory_key = f"{plugin_type}:{name}"
        self._factories[factory_key] = factory
        logger.info(f"Registered {plugin_type} factory: {name}")
    
    def get_critic(self, name: str) -> CriticInterface:
        """Get a critic by name with lazy loading.
        
        Args:
            name: Name of the critic to retrieve
            
        Returns:
            Critic instance
            
        Raises:
            ValueError: If critic is not registered
        """
        plugin_key = f"critic:{name}"
        
        if plugin_key not in self._plugins:
            raise ValueError(f"Critic '{name}' is not registered")
        
        # Check if already instantiated
        if name in self._critics:
            return self._critics[name]
        
        # Lazy load the critic
        metadata = self._plugins[plugin_key]
        if metadata.instance is None:
            # Load built-in critic
            critic = self._load_builtin_critic(name)
        else:
            critic = metadata.instance
        
        # Instantiate if it's a class
        if inspect.isclass(critic):
            critic = critic()
        elif callable(critic) and not isinstance(critic, CriticInterface):
            # It's a factory function
            critic = critic()
        
        self._critics[name] = critic
        return critic
    
    def get_validator(self, name: str) -> ValidatorInterface:
        """Get a validator by name with lazy loading.
        
        Args:
            name: Name of the validator to retrieve
            
        Returns:
            Validator instance
            
        Raises:
            ValueError: If validator is not registered
        """
        plugin_key = f"validator:{name}"
        
        if plugin_key not in self._plugins:
            raise ValueError(f"Validator '{name}' is not registered")
        
        # Check if already instantiated
        if name in self._validators:
            return self._validators[name]
        
        # Lazy load the validator
        metadata = self._plugins[plugin_key]
        validator = metadata.instance
        
        # Instantiate if it's a class
        if inspect.isclass(validator):
            validator = validator()
        elif callable(validator) and not isinstance(validator, ValidatorInterface):
            # It's a factory function
            validator = validator()
        
        self._validators[name] = validator
        return validator
    
    def _load_builtin_critic(self, name: str) -> Any:
        """Load a built-in critic by name.
        
        Args:
            name: Name of the built-in critic
            
        Returns:
            Critic class or instance
        """
        try:
            from sifaka.critics import (
                ReflexionCritic, ConstitutionalCritic, SelfRefineCritic,
                NCriticsCritic, SelfConsistencyCritic, PromptCritic,
                MetaEvaluationCritic, SelfRAGCritic
            )
            
            critic_classes = {
                "reflexion": ReflexionCritic,
                "constitutional": ConstitutionalCritic,
                "self_refine": SelfRefineCritic,
                "n_critics": NCriticsCritic,
                "self_consistency": SelfConsistencyCritic,
                "prompt": PromptCritic,
                "meta_rewarding": MetaEvaluationCritic,
                "self_rag": SelfRAGCritic,
            }
            
            if name in critic_classes:
                return critic_classes[name]
            else:
                raise ValueError(f"Unknown built-in critic: {name}")
                
        except ImportError as e:
            logger.error(f"Failed to import built-in critic {name}: {e}")
            raise ValueError(f"Failed to load built-in critic {name}: {e}") from e
    
    def _validate_critic(self, critic: Any) -> bool:
        """Validate that a critic implements the required interface."""
        if isinstance(critic, CriticInterface):
            return True
        
        if inspect.isclass(critic):
            return issubclass(critic, CriticInterface)
        
        if callable(critic):
            # Check if it's a factory function
            return True
        
        return False
    
    def _validate_validator(self, validator: Any) -> bool:
        """Validate that a validator implements the required interface."""
        if isinstance(validator, ValidatorInterface):
            return True
        
        if inspect.isclass(validator):
            return issubclass(validator, ValidatorInterface)
        
        if callable(validator):
            # Check if it's a factory function
            return True
        
        return False
    
    def discover_plugins(self, package_name: str) -> List[str]:
        """Discover plugins in a package.
        
        Args:
            package_name: Name of the package to search
            
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        try:
            package = importlib.import_module(package_name)
            
            # Look for plugin classes and functions
            for name in dir(package):
                obj = getattr(package, name)
                
                if (inspect.isclass(obj) and 
                    (issubclass(obj, CriticInterface) or issubclass(obj, ValidatorInterface))):
                    discovered.append(name)
                    logger.debug(f"Discovered plugin: {name} in {package_name}")
                    
        except ImportError as e:
            logger.warning(f"Failed to discover plugins in {package_name}: {e}")
        
        return discovered
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[PluginMetadata]:
        """List registered plugins.
        
        Args:
            plugin_type: Optional filter by plugin type
            
        Returns:
            List of plugin metadata
        """
        if plugin_type:
            return [meta for meta in self._plugins.values() if meta.plugin_type == plugin_type]
        else:
            return list(self._plugins.values())
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        plugin_types = {}
        for meta in self._plugins.values():
            plugin_types[meta.plugin_type] = plugin_types.get(meta.plugin_type, 0) + 1
        
        return {
            "created_at": self._created_at,
            "total_plugins": len(self._plugins),
            "plugin_types": plugin_types,
            "instantiated_critics": len(self._critics),
            "instantiated_validators": len(self._validators),
            "registered_factories": len(self._factories)
        }
    
    def clear(self) -> None:
        """Clear all registered plugins (useful for testing)."""
        self._plugins.clear()
        self._critics.clear()
        self._validators.clear()
        self._factories.clear()
        logger.info("Registry cleared")


# Global registry instance
_registry: Optional[SifakaRegistry] = None


def get_registry() -> SifakaRegistry:
    """Get the global registry instance.
    
    Returns:
        Global SifakaRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = SifakaRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    if _registry:
        _registry.clear()
    _registry = None
