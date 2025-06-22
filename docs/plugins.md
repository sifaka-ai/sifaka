# Sifaka Plugin System

Sifaka uses a plugin architecture to keep the core lightweight while supporting multiple storage backends. This document explains how to create and use storage plugins.

## Core vs Plugin Storage

**Core Storage** (built into Sifaka):
- `MemoryStorage` - Fast, non-persistent, no dependencies
- `FileStorage` - JSON files on disk, minimal dependencies

**Plugin Storage** (separate packages):
- `sifaka-redis` - Redis and Redis Cluster support
- `sifaka-mem0` - Semantic memory with AI-powered search
- `sifaka-mcp` - Model Context Protocol for any MCP storage server

## Using Plugin Storage

### Install Plugin
```bash
pip install sifaka-redis
# or
pip install sifaka-mem0
```

### Auto-Discovery (Recommended)
```python
from sifaka import improve, create_storage_backend

# Plugin automatically discovered via entry points
storage = create_storage_backend('redis', url='redis://localhost:6379')

result = await improve(
    "Write about quantum computing",
    storage=storage
)
```

### Manual Registration
```python
from sifaka import improve, register_storage_backend
from sifaka_redis import RedisStorage

# Register the plugin manually
register_storage_backend('redis', RedisStorage)

# Use it
storage = RedisStorage(url='redis://localhost:6379')
result = await improve("Text to improve", storage=storage)
```

### List Available Storage
```python
from sifaka import list_storage_backends

print("Available storage backends:", list_storage_backends())
# Output: ['memory', 'file', 'redis', 'mem0']
```

## Creating a Storage Plugin

### 1. Create the Storage Backend

```python
# sifaka_mydb/storage.py
from typing import Optional, List
from sifaka.storage import StorageBackend
from sifaka.core.models import SifakaResult

class MyDBStorage(StorageBackend):
    """Custom database storage backend."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # Initialize your database connection
    
    async def save(self, result: SifakaResult) -> str:
        """Save result to your database."""
        # Implement save logic
        return result.id
    
    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load result from your database."""
        # Implement load logic
        return None
    
    async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List stored result IDs."""
        # Implement list logic
        return []
    
    async def delete(self, result_id: str) -> bool:
        """Delete a stored result."""
        # Implement delete logic
        return False
    
    async def search(self, query: str, limit: int = 10) -> List[str]:
        """Search results by content."""
        # Implement search logic
        return []
```

### 2. Create Entry Point

```python
# setup.py or pyproject.toml
entry_points={
    'sifaka.storage': [
        'mydb = sifaka_mydb.storage:MyDBStorage',
    ],
}
```

### 3. Package Structure

```
sifaka-mydb/
├── sifaka_mydb/
│   ├── __init__.py
│   └── storage.py
├── setup.py
└── README.md
```

### 4. Auto-Registration

```python
# sifaka_mydb/__init__.py
from .storage import MyDBStorage

# Auto-register when imported
try:
    from sifaka import register_storage_backend
    register_storage_backend('mydb', MyDBStorage)
except ImportError:
    # Sifaka not installed, skip registration
    pass

__all__ = ['MyDBStorage']
```

## Plugin Examples

### Redis Plugin Structure (sifaka-redis)
```
sifaka-redis/
├── sifaka_redis/
│   ├── __init__.py
│   ├── storage.py      # RedisStorage class
│   └── cluster.py      # RedisClusterStorage class
├── setup.py
└── README.md
```

### Mem0 Plugin Structure (sifaka-mem0)
```
sifaka-mem0/
├── sifaka_mem0/
│   ├── __init__.py
│   └── storage.py      # Mem0Storage class
├── setup.py
└── README.md
```

### MCP Plugin Structure (sifaka-mcp)
```
sifaka-mcp/
├── sifaka_mcp/
│   ├── __init__.py
│   ├── client.py       # MCP client implementation
│   └── storage.py      # MCPStorage class
├── setup.py
└── README.md
```

## Benefits of Plugin Architecture

1. **Lightweight Core**: Core Sifaka has minimal dependencies
2. **Optional Dependencies**: Only install what you need
3. **Third-Party Extensions**: Anyone can create storage plugins
4. **Security**: Avoid pulling in unnecessary dependencies
5. **Maintenance**: Plugin maintainers handle their own storage logic

## Plugin Guidelines

1. **Naming**: Use `sifaka-{storage}` pattern (e.g., `sifaka-redis`)
2. **Entry Points**: Use `sifaka.storage` group for auto-discovery
3. **Error Handling**: Gracefully handle connection failures
4. **Documentation**: Include clear setup and usage examples
5. **Testing**: Test against core Sifaka interfaces
6. **Dependencies**: Declare all required dependencies in setup.py