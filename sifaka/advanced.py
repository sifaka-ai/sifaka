"""Advanced Sifaka API for complex use cases.

This module contains the advanced components for users who need full control
over Sifaka's configuration and behavior. Most users should use the simple
preset functions in the main sifaka module instead.

Use this module when:
- You need custom validators or critics
- You want to configure memory management
- You need persistence or storage backends
- You're building integrations or extensions

For 90% of use cases, use the simple presets instead:
    ```python
    import sifaka
    result = await sifaka.academic_writing("Your prompt")
    ```

Advanced usage:
    ```python
    from sifaka.advanced import SifakaEngine, SifakaConfig

    config = SifakaConfig(
        model="openai:gpt-4",
        max_iterations=5,
        critics=["reflexion", "constitutional"],
        validators=[custom_validator],
        enable_persistence=True,
    )
    engine = SifakaEngine(config=config)
    result = await engine.think("Your prompt")
    ```
"""

# Re-export advanced components
from sifaka.core.engine import SifakaEngine
from sifaka.core.thought import SifakaThought
from sifaka.graph.dependencies import SifakaDependencies
from sifaka.utils.config import SifakaConfig

# Re-export error types for advanced error handling
from sifaka.utils import (
    SifakaError,
    ValidationError,
    CritiqueError,
    GraphExecutionError,
    ConfigurationError,
)

# Re-export validators for custom validation
from sifaka import validators

# Re-export critics for custom critics
from sifaka import critics

# Re-export storage backends
from sifaka import storage

__all__ = [
    # Core advanced components
    "SifakaEngine",
    "SifakaThought",
    "SifakaDependencies",
    "SifakaConfig",
    # Error types
    "SifakaError",
    "ValidationError",
    "CritiqueError",
    "GraphExecutionError",
    "ConfigurationError",
    # Component modules
    "validators",
    "critics",
    "storage",
]
