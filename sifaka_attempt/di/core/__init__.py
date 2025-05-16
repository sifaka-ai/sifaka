"""
Core components of the dependency injection system.
"""

from sifaka.di.core.container import DependencyContainer
from sifaka.di.core.protocols import (
    DependencyContainerProtocol,
    DependencyHealthStatus,
    DependencyScope,
    RegistryProtocol,
    ResolutionContext,
    ResolverProtocol,
    ScopeContextProtocol,
    ScopeManagerProtocol,
)
from sifaka.di.core.registry import DependencyRegistry
from sifaka.di.core.resolver import DependencyResolver
from sifaka.di.core.scope_manager import ScopeContext, ScopeManager

__all__ = [
    "DependencyContainer",
    "DependencyContainerProtocol",
    "DependencyHealthStatus",
    "DependencyRegistry",
    "DependencyResolver",
    "DependencyScope",
    "RegistryProtocol",
    "ResolutionContext",
    "ResolverProtocol",
    "ScopeContext",
    "ScopeContextProtocol",
    "ScopeManager",
    "ScopeManagerProtocol",
]
