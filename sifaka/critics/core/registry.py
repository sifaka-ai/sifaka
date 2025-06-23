"""Dynamic critic registry for extensibility."""

from typing import Dict, Type, Optional, List, Callable
import importlib.metadata

from .base import BaseCritic
from ..core.interfaces import Critic
from ..core.constants import ENTRY_POINT_CRITICS


class CriticRegistry:
    """Registry for dynamically registering and discovering critics."""
    
    _critics: Dict[str, Type[Critic]] = {}
    _aliases: Dict[str, str] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        critic_class: Type[Critic],
        aliases: Optional[List[str]] = None
    ) -> None:
        """Register a critic with the registry.
        
        Args:
            name: Primary name for the critic
            critic_class: The critic class to register
            aliases: Optional alternative names
        """
        cls._critics[name.lower()] = critic_class
        
        if aliases:
            for alias in aliases:
                cls._aliases[alias.lower()] = name.lower()
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Critic]]:
        """Get a critic class by name.
        
        Args:
            name: Name or alias of the critic
            
        Returns:
            Critic class or None if not found
        """
        name_lower = name.lower()
        
        # Check direct registration
        if name_lower in cls._critics:
            return cls._critics[name_lower]
        
        # Check aliases
        if name_lower in cls._aliases:
            return cls._critics[cls._aliases[name_lower]]
        
        return None
    
    @classmethod
    def list(cls) -> List[str]:
        """List all registered critic names."""
        return sorted(cls._critics.keys())
    
    @classmethod
    def discover_plugins(cls) -> None:
        """Discover and register critics from entry points."""
        try:
            # Python 3.10+ with importlib.metadata
            entry_points = importlib.metadata.entry_points()
            critic_eps = entry_points.select(group=ENTRY_POINT_CRITICS)
        except AttributeError:
            # Python 3.9 compatibility
            entry_points = importlib.metadata.entry_points()
            critic_eps = entry_points.get(ENTRY_POINT_CRITICS, [])
        
        for ep in critic_eps:
            try:
                critic_class = ep.load()
                cls.register(ep.name, critic_class)
            except Exception as e:
                # Log but don't fail on individual plugin errors
                print(f"Failed to load critic plugin {ep.name}: {e}")
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (mainly for testing)."""
        cls._critics.clear()
        cls._aliases.clear()


# Pre-register built-in critics
def register_builtin_critics():
    """Register all built-in critics."""
    from ..reflexion import ReflexionCritic
    from ..constitutional import ConstitutionalCritic
    from ..self_refine import SelfRefineCritic
    from ..n_critics import NCriticsCritic
    from ..self_rag import SelfRAGCritic
    from ..meta_rewarding import MetaRewardingCritic
    from ..self_consistency import SelfConsistencyCritic
    from ..prompt import PromptCritic
    
    CriticRegistry.register("reflexion", ReflexionCritic, ["reflection"])
    CriticRegistry.register("constitutional", ConstitutionalCritic, ["constitutional_ai"])
    CriticRegistry.register("self_refine", SelfRefineCritic, ["self-refine", "selfrefine"])
    CriticRegistry.register("n_critics", NCriticsCritic, ["ncritics", "ensemble"])
    CriticRegistry.register("self_rag", SelfRAGCritic, ["selfrag", "self-rag"])
    CriticRegistry.register("meta_rewarding", MetaRewardingCritic, ["meta-rewarding", "metarewarding"])
    CriticRegistry.register("self_consistency", SelfConsistencyCritic, ["selfconsistency", "self-consistency"])
    CriticRegistry.register("prompt", PromptCritic, ["custom"])


# Register built-in critics on import
register_builtin_critics()

# Discover plugins
CriticRegistry.discover_plugins()