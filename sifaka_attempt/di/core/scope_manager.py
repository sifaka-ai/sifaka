"""
Scope manager for dependency injection.

The scope manager is responsible for managing scoped dependencies,
including creating scope contexts and clearing scopes.
"""

import logging
import threading
import uuid
from typing import Any, Dict, Optional

from sifaka.di.core.protocols import DependencyScope, ScopeContextProtocol, ScopeManagerProtocol

logger = logging.getLogger(__name__)


class ScopeContext(ScopeContextProtocol):
    """
    Context manager for a dependency scope.

    This class provides a context manager for dependency scopes,
    ensuring that scopes are properly cleaned up when no longer needed.
    """

    def __init__(self, scope_manager: "ScopeManager", scope_type: DependencyScope, scope_id: str):
        """
        Initialize a scope context.

        Args:
            scope_manager: The scope manager
            scope_type: The scope type
            scope_id: The scope ID
        """
        self._scope_manager = scope_manager
        self._scope_type = scope_type
        self._scope_id = scope_id

    def __enter__(self) -> Any:
        """
        Enter the scope context.

        Returns:
            The current scope context
        """
        logger.debug(f"Entering {self._scope_type.value} scope with ID {self._scope_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the scope context.

        This method cleans up the scope by clearing all dependencies
        registered in this scope.
        """
        logger.debug(f"Exiting {self._scope_type.value} scope with ID {self._scope_id}")
        self.clear()

    def clear(self) -> "ScopeContext":
        """
        Clear dependencies on exit.

        Returns:
            The current scope context
        """
        self._scope_manager.clear_scope(self._scope_type, self._scope_id)
        return self


class ScopeManager(ScopeManagerProtocol):
    """Manager for dependency scopes."""

    def __init__(self):
        """Initialize a new scope manager."""
        self._singleton_instances: Dict[str, Any] = {}
        self._session_instances: Dict[str, Dict[str, Any]] = {}
        self._request_instances: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create_scope(
        self, scope_type: DependencyScope, scope_id: Optional[str] = None
    ) -> ScopeContextProtocol:
        """
        Create a scope context.

        Args:
            scope_type: The type of scope to create
            scope_id: Optional scope ID

        Returns:
            A scope context manager
        """
        with self._lock:
            scope_id = scope_id or str(uuid.uuid4())

            if scope_type == DependencyScope.SESSION:
                if scope_id not in self._session_instances:
                    self._session_instances[scope_id] = {}
            elif scope_type == DependencyScope.REQUEST:
                if scope_id not in self._request_instances:
                    self._request_instances[scope_id] = {}

            return ScopeContext(self, scope_type, scope_id)

    def session_scope(self, session_id: Optional[str] = None) -> ScopeContextProtocol:
        """
        Create a session scope context.

        Args:
            session_id: Optional session ID

        Returns:
            A session scope context manager
        """
        return self.create_scope(DependencyScope.SESSION, session_id)

    def request_scope(
        self, request_id: Optional[str] = None, session_id: Optional[str] = None
    ) -> ScopeContextProtocol:
        """
        Create a request scope context.

        Args:
            request_id: Optional request ID
            session_id: Optional session ID

        Returns:
            A request scope context manager with a reference to the session
        """
        # If no request ID is provided, generate one
        request_id = request_id or str(uuid.uuid4())

        # Store the session ID with the request scope for reference
        if session_id:
            request_entry = self._request_instances.get(request_id, {})
            request_entry["__session_id__"] = session_id
            self._request_instances[request_id] = request_entry

        return self.create_scope(DependencyScope.REQUEST, request_id)

    def clear_scope(self, scope_type: DependencyScope, scope_id: Optional[str] = None) -> None:
        """
        Clear dependencies in a scope.

        Args:
            scope_type: The type of scope to clear
            scope_id: Optional scope ID
        """
        with self._lock:
            if scope_type == DependencyScope.SINGLETON:
                self._singleton_instances.clear()
            elif scope_type == DependencyScope.SESSION:
                if scope_id:
                    self._session_instances.pop(scope_id, None)
                else:
                    self._session_instances.clear()
            elif scope_type == DependencyScope.REQUEST:
                if scope_id:
                    self._request_instances.pop(scope_id, None)
                else:
                    self._request_instances.clear()

    def clear_all_scopes(self) -> None:
        """Clear all dependencies in all scopes."""
        with self._lock:
            self._singleton_instances.clear()
            self._session_instances.clear()
            self._request_instances.clear()

    def get_instance(
        self,
        name: str,
        scope: DependencyScope,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get an instance from a scope.

        Args:
            name: The dependency name
            scope: The dependency scope
            session_id: Optional session ID
            request_id: Optional request ID

        Returns:
            The dependency instance or None if the instance is not found
        """
        with self._lock:
            if scope == DependencyScope.SINGLETON:
                return self._singleton_instances.get(name)
            elif scope == DependencyScope.SESSION:
                if not session_id:
                    return None
                session_instances = self._session_instances.get(session_id, {})
                return session_instances.get(name)
            elif scope == DependencyScope.REQUEST:
                if not request_id:
                    return None
                request_instances = self._request_instances.get(request_id, {})
                return request_instances.get(name)
            else:  # TRANSIENT
                return None

    def set_instance(
        self,
        name: str,
        instance: Any,
        scope: DependencyScope,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Set an instance in a scope.

        Args:
            name: The dependency name
            instance: The dependency instance
            scope: The dependency scope
            session_id: Optional session ID
            request_id: Optional request ID
        """
        with self._lock:
            if scope == DependencyScope.SINGLETON:
                self._singleton_instances[name] = instance
            elif scope == DependencyScope.SESSION:
                if not session_id:
                    logger.warning(
                        f"Attempted to set session-scoped dependency '{name}' without a session ID"
                    )
                    return
                if session_id not in self._session_instances:
                    self._session_instances[session_id] = {}
                self._session_instances[session_id][name] = instance
            elif scope == DependencyScope.REQUEST:
                if not request_id:
                    logger.warning(
                        f"Attempted to set request-scoped dependency '{name}' without a request ID"
                    )
                    return
                if request_id not in self._request_instances:
                    self._request_instances[request_id] = {}
                self._request_instances[request_id][name] = instance
            # No storage for TRANSIENT scope

    def get_session_id_for_request(self, request_id: str) -> Optional[str]:
        """
        Get the session ID associated with a request.

        Args:
            request_id: The request ID

        Returns:
            The session ID or None if the request is not found
        """
        with self._lock:
            request_entry = self._request_instances.get(request_id, {})
            return request_entry.get("__session_id__")
