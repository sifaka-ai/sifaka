# Storage Setup Guide

Learn how to configure and use Sifaka's flexible storage system for persisting thoughts, caching, and semantic search.

## Overview

Sifaka provides multiple storage backends that can be used individually or combined in a 3-tier architecture:

- **Memory Storage**: Fast, temporary storage for development
- **File Storage**: Simple JSON persistence for single-user applications
- **Redis Storage**: High-performance caching via MCP
- **Milvus Storage**: Vector storage for semantic search via MCP

## Quick Start

### Memory Storage (Development)

Perfect for development and testing:

```python
from sifaka.storage import MemoryStorage
from sifaka import Chain

# Create memory storage
storage = MemoryStorage()

# Use in chain
chain = Chain(model=model, storage=storage)
```

### File Storage (Simple Persistence)

For simple applications that need persistence:

```python
from sifaka.storage import FileStorage

# Store thoughts in JSON file
storage = FileStorage("./my_thoughts.json")

chain = Chain(model=model, storage=storage)
```

## Redis Setup (Production Caching)

Redis provides fast, persistent caching for production use.

### 1. Install and Start Redis

```bash
# Using Docker (recommended)
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:alpine

# Verify Redis is running
docker ps | grep redis
redis-cli ping  # Should return "PONG"
```

### 2. Set Up Redis MCP Server

Sifaka uses Redis through the official Redis MCP server:

```bash
# Clone the official Redis MCP server
git clone https://github.com/redis/mcp-redis.git
cd mcp-redis

# Install dependencies
uv sync

# Test the MCP server
uv run src/main.py
```

### 3. Configure Redis Storage

```python
from sifaka.storage import RedisStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

# Configure Redis MCP connection
redis_config = MCPServerConfig(
    name="redis-server",
    transport_type=MCPTransportType.STDIO,
    url="uv run --directory /path/to/mcp-redis src/main.py"
)

# Create Redis storage
storage = RedisStorage(
    mcp_config=redis_config,
    key_prefix="sifaka:",  # Optional prefix for keys
    ttl=3600  # Optional TTL in seconds
)

# Use in chain
chain = Chain(model=model, storage=storage)
```

## Milvus Setup (Vector Search)

Milvus provides vector storage for semantic search capabilities.

### 1. Install and Start Milvus

```bash
# Download Milvus installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Start Milvus
bash standalone_embed.sh start

# Verify Milvus is running
curl http://localhost:9091/health  # Should return health status
```

### 2. Set Up Milvus MCP Server

Sifaka uses Milvus through the official Milvus MCP server:

```bash
# Clone the official Milvus MCP server
git clone https://github.com/zilliztech/mcp-server-milvus.git
cd mcp-server-milvus

# Install dependencies
uv sync

# Test the MCP server
uv run src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530
```

### 3. Configure Milvus Storage

```python
from sifaka.storage import MilvusStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType

# Configure Milvus MCP connection
milvus_config = MCPServerConfig(
    name="milvus-server",
    transport_type=MCPTransportType.STDIO,
    url="uv run --directory /path/to/mcp-server-milvus src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530"
)

# Create Milvus storage
storage = MilvusStorage(
    mcp_config=milvus_config,
    collection_name="sifaka_thoughts",  # Collection for storing thoughts
    dimension=768  # Vector dimension (depends on your embedding model)
)

# Use in chain
chain = Chain(model=model, storage=storage)
```

## 3-Tier Storage Architecture

Combine multiple storage backends for optimal performance:

```
Memory (L1) → Redis (L2) → Milvus (L3)
```

### Benefits

- **L1 (Memory)**: Ultra-fast access for frequently used data
- **L2 (Redis)**: Fast persistent cache for session data
- **L3 (Milvus)**: Long-term vector storage for semantic search

### Configuration

```python
from sifaka.storage import CachedStorage, MemoryStorage, RedisStorage, MilvusStorage

# Configure each tier
memory_storage = MemoryStorage()
redis_storage = RedisStorage(redis_config, key_prefix="sifaka:cache:")
milvus_storage = MilvusStorage(milvus_config, collection_name="thoughts")

# Create 3-tier storage
storage = CachedStorage(
    cache=memory_storage,  # L1: Memory
    persistence=CachedStorage(
        cache=redis_storage,    # L2: Redis
        persistence=milvus_storage  # L3: Milvus
    )
)

# Use in chain
chain = Chain(model=model, storage=storage)
```

### How It Works

1. **Read**: Check Memory → Redis → Milvus (first hit wins)
2. **Write**: Store in all tiers simultaneously
3. **Cache warming**: Lower tiers populate higher tiers on access

## Environment Configuration

Use environment variables for flexible configuration:

```bash
# .env file
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_password  # if authentication enabled

MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=your_username     # if authentication enabled
MILVUS_PASSWORD=your_password # if authentication enabled
```

```python
import os
from sifaka.storage import RedisStorage, MilvusStorage

# Use environment variables
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
milvus_host = os.getenv("MILVUS_HOST", "localhost")
milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
```

## Advanced Configuration

### Custom Redis Configuration

```python
redis_storage = RedisStorage(
    mcp_config=redis_config,
    key_prefix="myapp:",
    ttl=7200,  # 2 hours
    max_connections=20,
    retry_on_timeout=True
)
```

### Custom Milvus Configuration

```python
milvus_storage = MilvusStorage(
    mcp_config=milvus_config,
    collection_name="custom_thoughts",
    dimension=1024,  # Custom embedding dimension
    index_type="IVF_FLAT",  # Vector index type
    metric_type="L2",  # Distance metric
    nlist=1024  # Index parameter
)
```

### Storage with Encryption

```python
from sifaka.storage import EncryptedStorage

# Wrap any storage with encryption
encrypted_storage = EncryptedStorage(
    storage=redis_storage,
    encryption_key="your-32-byte-encryption-key"
)
```

## Performance Tuning

### Memory Storage

```python
# Configure memory limits
memory_storage = MemoryStorage(
    max_size=1000,  # Maximum number of items
    eviction_policy="lru"  # Least Recently Used eviction
)
```

### Redis Optimization

```bash
# Redis configuration (redis.conf)
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1  # Save snapshot every 15 minutes if at least 1 key changed
```

### Milvus Optimization

```python
# Optimize for your use case
milvus_storage = MilvusStorage(
    mcp_config=milvus_config,
    collection_name="optimized_thoughts",
    dimension=768,
    index_type="HNSW",  # Better for high-dimensional vectors
    metric_type="COSINE",  # Good for text embeddings
    M=16,  # HNSW parameter
    efConstruction=200  # HNSW parameter
)
```

## Monitoring and Debugging

### Storage Metrics

```python
# Get storage statistics
stats = storage.get_stats()
print(f"Total items: {stats.total_items}")
print(f"Cache hit rate: {stats.cache_hit_rate:.2%}")
print(f"Memory usage: {stats.memory_usage_mb:.1f} MB")
```

### Debug Logging

```python
import logging
from sifaka.utils.logging import get_logger

# Enable debug logging for storage
logger = get_logger("sifaka.storage")
logger.setLevel(logging.DEBUG)

# This will show all storage operations
storage = RedisStorage(redis_config)
```

## Troubleshooting

### Redis Issues

```bash
# Check Redis status
docker ps | grep redis
redis-cli ping

# View Redis logs
docker logs redis

# Test Redis MCP server
cd mcp/mcp-redis
python -m main.py
```

### Milvus Issues

```bash
# Check Milvus status
docker ps | grep milvus
curl http://localhost:9091/health

# View Milvus logs
docker logs milvus

# Test Milvus MCP server
cd mcp/mcp-server-milvus
python -m mcp_server_milvus
```

### Common Problems

**Connection Refused**
- Ensure Redis/Milvus is running
- Check port bindings (6379 for Redis, 19530 for Milvus)
- Verify firewall settings

**MCP Server Errors**
- Check Python version (3.10+ for Milvus MCP)
- Ensure dependencies are installed
- Verify MCP server paths

**Performance Issues**
- Monitor memory usage
- Check network latency
- Consider using 3-tier storage for better performance

## Migration Guide

### From Direct Redis to MCP

```python
# Old (deprecated)
import redis
r = redis.Redis(host='localhost', port=6379)

# New (recommended)
from sifaka.storage import RedisStorage
storage = RedisStorage(redis_config)
```

### From Direct Milvus to MCP

```python
# Old (deprecated)
from pymilvus import connections, Collection
connections.connect("default", host="localhost", port="19530")

# New (recommended)
from sifaka.storage import MilvusStorage
storage = MilvusStorage(milvus_config)
```

## Next Steps

- **[Performance tuning guide](performance-tuning.md)** - Optimize storage performance
- **[Custom models guide](custom-models.md)** - Create models that work with storage
- **[API reference](../api/api-reference.md)** - Complete storage API documentation

Your storage system is now ready to provide fast, reliable persistence for your Sifaka applications!
