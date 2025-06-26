# Tool-Aware Critics: Hybrid Design

## Overview

This document outlines the design for making Sifaka critics tool-aware while maintaining backward compatibility and a clean separation of concerns. Critics will remain evaluation-focused by default but can optionally use tools when enabled.

## Architecture

### Hybrid Approach

We will use a hybrid approach with:
1. **Core interfaces** in the main Sifaka package
2. **Tool implementations** in a separate `sifaka-tools` package

```
sifaka/                    # Main package
  tools/
    base.py                # Tool interface only
    registry.py            # Tool registry only

sifaka-tools/              # Separate package
  websearch/               # Web search implementations
    duckduckgo.py
  storage/                 # Storage backend implementations
    redis.py
    mem0.py
    postgresql.py
```

## Core Design

### 1. Tool-Aware Base Critic

```python
class BaseCritic:
    def __init__(self, ..., enable_tools: bool = False):
        self.enable_tools = enable_tools
        self.tools = []
        if enable_tools:
            self.tools = self._get_available_tools()

    def _get_available_tools(self) -> List[Tool]:
        """Override to specify which tools this critic can use."""
        return []
```

### 2. Tool Interface

```python
# In sifaka/tools/base.py
from typing import Protocol, List, Dict, Any, Optional

class ToolInterface(Protocol):
    """Protocol for PydanticAI-compatible tools."""

    async def __call__(
        self,
        query: str,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Execute tool query and return results."""
        ...

    @property
    def name(self) -> str:
        """Tool name for identification."""
        ...

    @property
    def description(self) -> str:
        """Tool description for LLM context."""
        ...
```

### 3. Storage Backend Interface

```python
# In sifaka/tools/base.py
class StorageInterface(Protocol):
    """Protocol for tool storage backends."""

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key."""
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Store value with optional TTL."""
        ...

    async def delete(self, key: str) -> None:
        """Delete value by key."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...
```

### 4. Tool Registry

```python
# In sifaka/tools/registry.py
from typing import Dict, Type, Optional

class ToolRegistry:
    """Registry for tool implementations."""

    _tools: Dict[str, Type[ToolInterface]] = {}

    @classmethod
    def register(cls, name: str, tool_class: Type[ToolInterface]) -> None:
        """Register a tool implementation."""
        cls._tools[name] = tool_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[ToolInterface]]:
        """Get a tool by name."""
        return cls._tools.get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered tools."""
        return list(cls._tools.keys())
```

## Implementation Plan

### Phase 1: Core Interface (In Sifaka)
1. Create `sifaka/tools/` directory
2. Implement `base.py` with interfaces
3. Implement `registry.py` for tool registration
4. Update `BaseCritic` to support `enable_tools` flag
5. Add tool configuration to `Config` class

### Phase 2: Basic Tool Implementation (In sifaka-tools)
1. Create separate `sifaka-tools` package
2. Implement `DuckDuckGoTool` for web search
3. Implement basic in-memory caching
4. Create installation extras: `pip install sifaka[tools]`

### Phase 3: Storage Backends (In sifaka-tools)
1. Implement Redis storage backend
2. Implement mem0 storage backend
3. Implement PostgreSQL storage backend
4. Add backend selection logic

## Configuration

```python
# In sifaka/core/config.py
class Config:
    # Tool settings
    enable_tools: bool = Field(
        default=False,
        description="Enable tool usage for critics"
    )

    tool_timeout: float = Field(
        default=5.0,
        description="Maximum time for tool calls"
    )

    tool_cache_ttl: int = Field(
        default=3600,
        description="Tool result cache TTL in seconds"
    )

    # Per-critic tool settings
    critic_tool_settings: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "self_rag": {"enable_tools": True},      # On by default
            "constitutional": {"enable_tools": False},
            "meta_rewarding": {"enable_tools": False},
            # ... other critics
        },
        description="Per-critic tool configuration"
    )
```

## Usage Examples

### Basic Usage (No Tools)
```python
# Works exactly as before
critic = SelfRAGCritic()
result = await critic.critique(text, result)
```

### With Tools Enabled
```python
# Enable tools for a specific critic
critic = SelfRAGCritic(enable_tools=True)
result = await critic.critique(text, result)
# Now includes verified claims from web search

# Or via config
config = Config(
    enable_tools=True,
    critic_tool_settings={
        "self_rag": {"enable_tools": True}
    }
)
```

### Custom Tool Registration
```python
# In sifaka-tools or user code
from sifaka.tools import ToolRegistry

class CustomSearchTool:
    async def __call__(self, query: str, **kwargs):
        # Custom implementation
        return [{"content": "result", "source": "custom"}]

    @property
    def name(self):
        return "custom_search"

# Register the tool
ToolRegistry.register("custom_search", CustomSearchTool)
```

## Benefits

1. **Backward Compatible**: Critics work without any tools
2. **Progressive Enhancement**: Add tools as needed
3. **Clean Separation**: Tools don't bloat core package
4. **Extensible**: Easy to add new tools and storage backends
5. **Type Safe**: Full typing support with protocols
6. **Performance**: Caching and timeout controls

## Latency Management

### Caching Strategy
- **L1 Cache**: In-memory (instant)
- **L2 Cache**: Redis/mem0 (1-5ms)
- **L3 Cache**: Tool execution (100-1000ms)

### Async Optimization
```python
# Critics can start tool calls early
async def critique_with_tools(self, text):
    # Start tool calls in background
    tool_futures = self._start_tool_calls(text)

    # Do normal critique while tools run
    base_critique = await self._base_critique(text)

    # Gather tool results with timeout
    if tool_futures:
        tool_results = await asyncio.wait_for(
            asyncio.gather(*tool_futures, return_exceptions=True),
            timeout=self.config.tool_timeout
        )
        return self._enhance_critique(base_critique, tool_results)

    return base_critique
```

## Next Steps

1. **Implement Core Interface**: Add tool support to Sifaka core
2. **Create sifaka-tools Package**: Set up separate package structure
3. **Implement DuckDuckGo Tool**: Start with web search
4. **Add Storage Backends**: Redis, mem0, PostgreSQL
5. **Update Critics**: Make Self-RAG use tools by default
6. **Documentation**: Usage guides and examples

This design provides a clean, extensible foundation for tool-aware critics while maintaining simplicity and backward compatibility.
