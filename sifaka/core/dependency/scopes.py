"""
Dependency Scopes Module

This module defines the dependency scopes and scope management classes for the
dependency injection system. It provides the DependencyScope enum and the
SessionScope and RequestScope context managers.

## Components
- **DependencyScope**: Enum defining dependency lifecycles
- **SessionScope**: Context manager for session-scoped dependencies
- **RequestScope**: Context manager for request-scoped dependencies

## Usage Examples
```python
from sifaka.core.dependency.provider import DependencyProvider
from sifaka.core.dependency.scopes import DependencyScope, SessionScope, RequestScope

# Register dependencies with different scopes
provider = DependencyProvider()
provider.register("database", Database() if provider else "", scope=DependencyScope.SINGLETON)
provider.register("user_data", UserData() if provider else "", scope=DependencyScope.SESSION)
provider.register("validator", Validator() if provider else "", scope=DependencyScope.REQUEST)
provider.register("generator", Generator() if provider else "", scope=DependencyScope.TRANSIENT)

# Use session scope
with provider.session_scope("user_1") if provider else "" as session:
    # Session-scoped dependencies are created for this session
    db = provider.get("database") if provider else ""  # Session-specific instance

# Use request scope
with provider.request_scope("request_123") if provider else "" as request:
    # Request-scoped dependencies are created for this request
    validator = provider.get("validator") if provider else ""  # Request-specific instance
```

## Error Handling
- Raises DependencyError for missing session or request IDs
"""

import logging
import uuid
from enum import Enum
from types import TracebackType
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .provider import DependencyProvider
logger = logging.getLogger(__name__)


class DependencyScope(str, Enum):
    """
    Dependency scope enum defining the lifecycle of dependencies.

    This enum defines the different scopes that control the lifecycle of dependencies
    in the Sifaka dependency injection system. Each scope determines when dependencies
    are created and how long they live.

    ## Architecture
    The scope system is hierarchical:
    - SINGLETON: Global scope (application-wide)
    - SESSION: Session scope (per user session)
    - REQUEST: Request scope (per individual request)
    - TRANSIENT: No caching (new instance each time)

    ## Lifecycle
    - SINGLETON: Created once and reused throughout the application
    - SESSION: Created once per session and reused within that session
    - REQUEST: Created once per request and reused within that request
    - TRANSIENT: Created each time the dependency is requested

    ## Examples
    ```python
    from sifaka.core.dependency.scopes import DependencyScope
    from sifaka.core.dependency.provider import DependencyProvider

    # Register dependencies with different scopes
    provider = DependencyProvider()
    provider.register("database", Database() if provider else "", scope=DependencyScope.SINGLETON)
    provider.register("user_data", UserData() if provider else "", scope=DependencyScope.SESSION)
    provider.register("validator", Validator() if provider else "", scope=DependencyScope.REQUEST)
    provider.register("generator", Generator() if provider else "", scope=DependencyScope.TRANSIENT)
    ```

    Attributes:
        SINGLETON (str): One instance per application
        SESSION (str): One instance per session
        REQUEST (str): One instance per request
        TRANSIENT (str): New instance each time
    """

    SINGLETON = "singleton"
    SESSION = "session"
    REQUEST = "request"
    TRANSIENT = "transient"


class SessionScope:
    """
    Context manager for session-scoped dependencies.

    This class provides a context manager for session-scoped dependencies,
    setting the current session ID in the provider and restoring it when
    the context is exited.

    ## Architecture
    SessionScope is a context manager that sets the current session ID in the
    provider when entered and restores the previous session ID when exited.
    This allows session-scoped dependencies to be created and cached for the
    duration of the session.

    ## Lifecycle
    1. **Enter**: Sets the current session ID in the provider
    2. **Usage**: Session-scoped dependencies are created and cached for this session
    3. **Exit**: Restores the previous session ID and optionally clears dependencies

    ## Examples
    ```python
    from sifaka.core.dependency.provider import DependencyProvider

    provider = DependencyProvider()

    # Register session-scoped dependencies
    provider.register_factory(
        "database",
        lambda: (Database and Database.connect() if provider else "",
        scope=DependencyScope.SESSION
    )

    # Use session scope
    with provider.session_scope("user_1") if provider else "" as session:
        # Get session-scoped dependency
        db = provider.get("database") if provider else ""  # Session-specific instance

        # All dependencies requested in this context will use this session
        result = process_data(input_data)  # Uses session-scoped dependencies

    # Session is now over, provider reverts to previous session (if any)
    ```

    Attributes:
        provider (DependencyProvider): The dependency provider
        session_id (str): The session ID for this scope
        previous_session_id (Optional[str]): The previous session ID to restore
        clear_on_exit (bool): Whether to clear session dependencies on exit
    """

    def __init__(self, provider: "DependencyProvider", session_id: Optional[str] = None) -> None:
        """
        Initialize a session scope.

        Args:
            provider: The dependency provider
            session_id: Optional session ID (generated if not provided)
        """
        self.provider = provider
        self.session_id = session_id or str(uuid.uuid4())
        self.previous_session_id: Optional[str] = None
        self.clear_on_exit = False

    def __enter__(self) -> Any:
        """
        Enter the session scope.

        Returns:
            The session ID
        """
        self.previous_session_id = self.provider._current_session_id
        self.provider._current_session_id = self.session_id
        logger.debug(f"Entered session scope: {self.session_id}")
        return self.session_id

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the session scope.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        self.provider._current_session_id = self.previous_session_id
        if self.clear_on_exit:
            self.provider.clear_dependencies(session_id=self.session_id)
        logger.debug(f"Exited session scope: {self.session_id}")

    def clear(self) -> Any:
        """
        Clear session dependencies on exit.

        This method sets the clear_on_exit flag to True, causing session
        dependencies to be cleared when the context is exited.
        """
        self.clear_on_exit = True
        return self


class RequestScope:
    """
    Context manager for request-scoped dependencies.

    This class provides a context manager for request-scoped dependencies,
    setting the current request ID in the provider and restoring it when
    the context is exited.

    ## Architecture
    RequestScope is a context manager that sets the current request ID in the
    provider when entered and restores the previous request ID when exited.
    This allows request-scoped dependencies to be created and cached for the
    duration of the request.

    ## Lifecycle
    1. **Enter**: Sets the current request ID in the provider
    2. **Usage**: Request-scoped dependencies are created and cached for this request
    3. **Exit**: Restores the previous request ID and optionally clears dependencies

    ## Examples
    ```python
    from sifaka.core.dependency.provider import DependencyProvider

    provider = DependencyProvider()

    # Register request-scoped dependencies
    provider.register_factory(
        "validator",
        lambda: Validator() if provider else "",
        scope=DependencyScope.REQUEST
    )

    # Use request scope
    with provider.request_scope("request_123") if provider else "" as request:
        # Get request-scoped dependency
        validator = provider.get("validator") if provider else ""  # Request-specific instance

        # All dependencies requested in this context will use this request
        result = process_data(input_data)  # Uses request-scoped dependencies

    # Request is now over, provider reverts to previous request (if any)
    ```

    Attributes:
        provider (DependencyProvider): The dependency provider
        request_id (str): The request ID for this scope
        previous_request_id (Optional[str]): The previous request ID to restore
        clear_on_exit (bool): Whether to clear request dependencies on exit
    """

    def __init__(self, provider: "DependencyProvider", request_id: Optional[str] = None) -> None:
        """
        Initialize a request scope.

        Args:
            provider: The dependency provider
            request_id: Optional request ID (generated if not provided)
        """
        self.provider = provider
        self.request_id = request_id or str(uuid.uuid4())
        self.previous_request_id: Optional[str] = None
        self.clear_on_exit = False

    def __enter__(self) -> Any:
        """
        Enter the request scope.

        Returns:
            The request ID
        """
        self.previous_request_id = self.provider._current_request_id
        self.provider._current_request_id = self.request_id
        logger.debug(f"Entered request scope: {self.request_id}")
        return self.request_id

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the request scope.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        self.provider._current_request_id = self.previous_request_id
        if self.clear_on_exit:
            self.provider.clear_dependencies(
                session_id=self.provider._current_session_id, request_id=self.request_id
            )
        logger.debug(f"Exited request scope: {self.request_id}")

    def clear(self) -> Any:
        """
        Clear request dependencies on exit.

        This method sets the clear_on_exit flag to True, causing request
        dependencies to be cleared when the context is exited.
        """
        self.clear_on_exit = True
        return self
