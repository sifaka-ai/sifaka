# Storage Architecture

Sifaka provides a flexible storage system supporting multiple backends for different use cases. The storage architecture follows a unified protocol that allows you to choose from in-memory, file-based, Redis, or Milvus storage, or combine them in layered configurations.

## Storage Backends

### Memory Storage
In-memory storage for development and testing:
```python
from sifaka.storage import MemoryStorage

storage = MemoryStorage()
```

### File Storage
JSON file persistence for simple deployments:
```python
from sifaka.storage import FileStorage

storage = FileStorage("./thoughts.json")
```

### Redis Storage (via MCP)
High-performance caching and persistence using Redis through Model Context Protocol:
```python
from sifaka.storage import RedisStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

redis_config = MCPServerConfig(
    name="redis-server",
    transport_type=MCPTransportType.STDIO,
    url="cd mcp/mcp-redis && python -m main.py"
)
storage = RedisStorage(redis_config)
```

### Milvus Storage (via MCP)
Vector storage for semantic search using Milvus through Model Context Protocol:
```python
from sifaka.storage import MilvusStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

milvus_config = MCPServerConfig(
    name="milvus-server",
    transport_type=MCPTransportType.STDIO,
    url="cd mcp/mcp-server-milvus && python -m mcp_server_milvus"
)
storage = MilvusStorage(milvus_config, collection_name="thoughts")
```

## 3-Tier Storage Architecture

Sifaka supports a 3-tier storage system that combines multiple backends for optimal performance:

```
Memory (L1) → Redis (L2) → Milvus (L3)
```

### Benefits
- **Memory**: Ultra-fast access for frequently used data
- **Redis**: Fast persistent cache for session data
- **Milvus**: Long-term vector storage for semantic search

### Configuration
```python
from sifaka.storage import CachedStorage, MemoryStorage, RedisStorage, MilvusStorage

# 3-tier configuration
storage = CachedStorage(
    cache=MemoryStorage(),
    persistence=CachedStorage(
        cache=RedisStorage(redis_config),
        persistence=MilvusStorage(milvus_config)
    )
)
```

## Installation and Setup

### Redis Setup

#### Docker Installation
```bash
# Pull and run Redis container
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:alpine

# Verify Redis is running
docker ps | grep redis
```

For more configuration options, see the [official Redis Docker documentation](https://hub.docker.com/_/redis).

#### MCP Server Setup
The Redis MCP server is included locally in the Sifaka repository:

```bash
# Navigate to the Redis MCP server directory
cd mcp/mcp-redis

# Install dependencies (if needed)
pip install -e .

# Run the MCP server
python -m main.py
```

For more details, see the local [Redis MCP server](../../mcp/mcp-redis/README.md) or the [upstream repository](https://github.com/redis/mcp-redis).

### Milvus Setup

#### Docker Installation
```bash
# Download Milvus standalone Docker Compose
curl -sfL https://github.com/milvus-io/milvus/releases/download/v2.4.15/milvus-standalone-docker-compose.yml -o docker-compose.yml

# Start Milvus
docker-compose up -d

# Verify Milvus is running
docker-compose ps
```

For detailed installation instructions, see the [official Milvus Docker documentation](https://milvus.io/docs/install_standalone-docker.md).

#### MCP Server Setup
The Milvus MCP server is included locally in the Sifaka repository:

```bash
# Navigate to the Milvus MCP server directory
cd mcp/mcp-server-milvus

# Install dependencies (requires Python 3.10+)
pip install -e .

# Run the MCP server
python -m mcp_server_milvus
```

For more details, see the local [Milvus MCP server](../../mcp/mcp-server-milvus/README.md) or the [upstream repository](https://github.com/zilliztech/mcp-server-milvus).

## Configuration Examples

### Basic Redis Configuration
```python
from sifaka.core.chain import Chain
from sifaka.storage import RedisStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

# Configure Redis MCP (using local server)
redis_config = MCPServerConfig(
    name="redis-server",
    transport_type=MCPTransportType.STDIO,
    url="cd mcp/mcp-redis && python -m main.py"
)

# Create chain with Redis storage
chain = Chain(
    model=model,
    storage=RedisStorage(redis_config)
)
```

### Basic Milvus Configuration
```python
from sifaka.core.chain import Chain
from sifaka.storage import MilvusStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

# Configure Milvus MCP (using local server)
milvus_config = MCPServerConfig(
    name="milvus-server",
    transport_type=MCPTransportType.STDIO,
    url="cd mcp/mcp-server-milvus && python -m mcp_server_milvus"
)

# Create chain with Milvus storage
chain = Chain(
    model=model,
    storage=MilvusStorage(milvus_config, collection_name="my_thoughts")
)
```

### 3-Tier Configuration
```python
from sifaka.storage import CachedStorage, MemoryStorage, RedisStorage, MilvusStorage

# Memory → Redis → Milvus
storage = CachedStorage(
    cache=MemoryStorage(),
    persistence=CachedStorage(
        cache=RedisStorage(redis_config),
        persistence=MilvusStorage(milvus_config)
    )
)

chain = Chain(model=model, storage=storage)
```

## Environment Variables

Configure storage backends using environment variables:

```bash
# Redis configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_password  # if authentication is enabled

# Milvus configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=your_username     # if authentication is enabled
MILVUS_PASSWORD=your_password # if authentication is enabled
```

## Storage Protocol

All storage backends implement the unified `Storage` protocol:

```python
from typing import Protocol, Optional, Any

class Storage(Protocol):
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        ...

    async def set(self, key: str, value: Any) -> None:
        """Store a value with a key."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a value by key."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...

    async def clear(self) -> None:
        """Clear all stored data."""
        ...
```

This unified interface allows you to switch between storage backends without changing your application code.

## Performance Considerations

### Memory Storage
- **Pros**: Fastest access, no network overhead
- **Cons**: Data lost on restart, limited by RAM
- **Use case**: Development, testing, temporary data

### File Storage
- **Pros**: Simple, persistent, no external dependencies
- **Cons**: Slower than memory, not suitable for concurrent access
- **Use case**: Single-user applications, simple deployments

### Redis Storage
- **Pros**: Fast, persistent, supports clustering
- **Cons**: Requires Redis server, memory-based storage
- **Use case**: Production caching, session storage

### Milvus Storage
- **Pros**: Vector search capabilities, scalable, persistent
- **Cons**: More complex setup, higher resource usage
- **Use case**: Semantic search, large-scale vector storage

### 3-Tier Storage
- **Pros**: Optimal performance and persistence
- **Cons**: Complex setup, higher resource usage
- **Use case**: Production systems requiring both speed and durability

## Troubleshooting

### Redis Connection Issues
```bash
# Check if Redis is running
docker ps | grep redis

# Test Redis connection
redis-cli ping

# Check Redis logs
docker logs redis
```

### Milvus Connection Issues
```bash
# Check if Milvus is running
docker-compose ps

# Check Milvus logs
docker-compose logs milvus-standalone

# Test Milvus connection
curl http://localhost:9091/health
```

### MCP Server Issues
```bash
# Test Redis MCP server (local)
cd mcp/mcp-redis
python -m main.py

# Test Milvus MCP server (local)
cd mcp/mcp-server-milvus
python -m mcp_server_milvus
```

## Migration Guide

### From Direct Redis to MCP Redis
```python
# Old direct Redis usage (deprecated)
import redis
r = redis.Redis(host='localhost', port=6379)

# New MCP-based usage (using local server)
from sifaka.storage import RedisStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

config = MCPServerConfig(
    name="redis-server",
    transport_type=MCPTransportType.STDIO,
    url="cd mcp/mcp-redis && python -m main.py"
)
storage = RedisStorage(config)
```

### From Direct Milvus to MCP Milvus
```python
# Old direct Milvus usage (deprecated)
from pymilvus import connections, Collection
connections.connect("default", host="localhost", port="19530")

# New MCP-based usage (using local server)
from sifaka.storage import MilvusStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

config = MCPServerConfig(
    name="milvus-server",
    transport_type=MCPTransportType.STDIO,
    url="cd mcp/mcp-server-milvus && python -m mcp_server_milvus"
)
storage = MilvusStorage(config, collection_name="thoughts")
```
