"""Flexible Hybrid Storage Example for Sifaka.

This example demonstrates the new flexible hybrid persistence system that can
combine any number of storage backends with configurable roles and priorities.

Example Architecture:
Memory (Cache) → Redis (Cache) → PostgreSQL (Primary) → File (Backup) → Milvus (Search)

Prerequisites:
1. Redis server running
2. PostgreSQL database set up (optional)
3. Milvus server running (optional)
4. Set environment variables: ANTHROPIC_API_KEY, DATABASE_URL
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from sifaka import SifakaEngine
from sifaka.storage.flexible_hybrid import (
    FlexibleHybridPersistence,
    BackendConfig,
    BackendRole
)
from sifaka.storage import (
    MemoryPersistence,
    SifakaFilePersistence,
    RedisPersistence
)
from sifaka.utils.logging import get_logger

# Import optional backends
try:
    from sifaka.storage.postgresql import PostgreSQLPersistence
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    from sifaka.storage.milvus import MilvusPersistence
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    from pydantic_ai.mcp import MCPServerStdio
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


async def create_simple_hybrid() -> FlexibleHybridPersistence:
    """Create a simple 2-tier hybrid: Memory + File."""
    print("\n🔧 Creating Simple Hybrid (Memory + File)")
    
    backends = [
        BackendConfig(
            backend=MemoryPersistence(key_prefix="simple"),
            role=BackendRole.CACHE,
            priority=0,
            name="MemoryCache"
        ),
        BackendConfig(
            backend=SifakaFilePersistence(
                storage_dir="simple_hybrid_storage",
                key_prefix="simple"
            ),
            role=BackendRole.PRIMARY,
            priority=1,
            read_repair_target=True,
            name="FilePrimary"
        ),
    ]
    
    hybrid = FlexibleHybridPersistence(
        backends=backends,
        key_prefix="simple",
        write_through=True,
        read_repair=True
    )
    
    print(f"✅ Created simple hybrid with {len(backends)} backends")
    return hybrid


async def create_complex_hybrid() -> FlexibleHybridPersistence:
    """Create a complex multi-tier hybrid with all available backends."""
    print("\n🔧 Creating Complex Hybrid (All Available Backends)")
    
    backends = []
    
    # L1: Memory Cache (highest priority)
    backends.append(BackendConfig(
        backend=MemoryPersistence(key_prefix="complex"),
        role=BackendRole.CACHE,
        priority=0,
        read_repair_target=True,
        name="L1_Memory"
    ))
    
    # L2: Redis Cache (if available)
    if REDIS_AVAILABLE:
        try:
            redis_mcp_server = MCPServerStdio(
                "uv",
                args=[
                    "run",
                    "--directory",
                    "/Users/evanvolgas/Documents/not_beam/sifaka/mcp/mcp-redis",
                    "src/main.py",
                ],
                tool_prefix="redis",
            )
            
            backends.append(BackendConfig(
                backend=RedisPersistence(
                    mcp_server=redis_mcp_server,
                    key_prefix="complex",
                    ttl_seconds=3600
                ),
                role=BackendRole.CACHE,
                priority=1,
                read_repair_target=True,
                name="L2_Redis"
            ))
            print("✅ Added Redis cache layer")
        except Exception as e:
            print(f"⚠️  Redis not available: {e}")
    
    # L3: PostgreSQL Primary (if available)
    if POSTGRESQL_AVAILABLE and os.getenv("DATABASE_URL"):
        try:
            backends.append(BackendConfig(
                backend=PostgreSQLPersistence(
                    connection_string=os.getenv("DATABASE_URL"),
                    key_prefix="complex"
                ),
                role=BackendRole.PRIMARY,
                priority=2,
                read_repair_target=True,
                name="L3_PostgreSQL"
            ))
            print("✅ Added PostgreSQL primary storage")
        except Exception as e:
            print(f"⚠️  PostgreSQL not available: {e}")
    
    # L4: File Backup
    backends.append(BackendConfig(
        backend=SifakaFilePersistence(
            storage_dir="complex_hybrid_storage",
            key_prefix="complex",
            auto_backup=True
        ),
        role=BackendRole.BACKUP,
        priority=3,
        name="L4_FileBackup"
    ))
    
    # L5: Milvus Search Index (if available)
    if MILVUS_AVAILABLE:
        try:
            backends.append(BackendConfig(
                backend=MilvusPersistence(
                    collection_name="sifaka_complex",
                    key_prefix="complex"
                ),
                role=BackendRole.SEARCH,
                priority=4,
                read_enabled=False,  # Write-only for indexing
                write_enabled=True,
                name="L5_MilvusSearch"
            ))
            print("✅ Added Milvus search index")
        except Exception as e:
            print(f"⚠️  Milvus not available: {e}")
    
    if len(backends) < 2:
        # Fallback to simple hybrid if no backends available
        print("⚠️  Not enough backends available, falling back to simple hybrid")
        return await create_simple_hybrid()
    
    hybrid = FlexibleHybridPersistence(
        backends=backends,
        key_prefix="complex",
        write_through=True,
        read_repair=True,
        max_concurrent_writes=3
    )
    
    print(f"✅ Created complex hybrid with {len(backends)} backends:")
    for backend in backends:
        print(f"   - {backend.name} ({backend.role.value}, priority={backend.priority})")
    
    return hybrid


async def demonstrate_hybrid_operations(hybrid: FlexibleHybridPersistence, name: str):
    """Demonstrate operations with the hybrid storage."""
    print(f"\n📊 Testing {name} Operations")
    
    # Create engine with hybrid storage
    engine = SifakaEngine(persistence=hybrid)
    
    # Process some thoughts
    prompts = [
        "Explain the benefits of renewable energy",
        "Describe how machine learning works",
        "What are the challenges of space exploration?"
    ]
    
    thoughts = []
    for i, prompt in enumerate(prompts):
        print(f"   Processing thought {i+1}: {prompt[:30]}...")
        thought = await engine.think(prompt, max_iterations=1)
        thoughts.append(thought)
    
    print(f"✅ Processed {len(thoughts)} thoughts")
    
    # Test retrieval
    print("   Testing thought retrieval...")
    for thought in thoughts:
        retrieved = await engine.get_thought(thought.id)
        if retrieved:
            print(f"   ✅ Retrieved thought: {thought.id}")
        else:
            print(f"   ❌ Failed to retrieve thought: {thought.id}")
    
    # Get backend statistics
    print("   Getting backend statistics...")
    stats = await hybrid.get_backend_stats()
    
    print(f"   📈 Global Stats:")
    for key, value in stats["global_stats"].items():
        print(f"      {key}: {value}")
    
    print(f"   📈 Backend Stats:")
    for backend_name, backend_stats in stats["backend_stats"].items():
        print(f"      {backend_name}:")
        print(f"         Role: {backend_stats['role']}")
        print(f"         Reads: {backend_stats['read_count']}")
        print(f"         Writes: {backend_stats['write_count']}")
        print(f"         Errors: {backend_stats['error_count']}")
    
    return thoughts


async def demonstrate_health_monitoring(hybrid: FlexibleHybridPersistence):
    """Demonstrate health monitoring capabilities."""
    print("\n🏥 Health Check")
    
    health = await hybrid.health_check()
    
    print(f"   Overall Health: {'✅ Healthy' if health['overall_healthy'] else '❌ Unhealthy'}")
    print(f"   Healthy Backends: {health['healthy_backends']}/{health['total_backends']}")
    print(f"   Health Percentage: {health['health_percentage']:.1f}%")
    
    print("   Backend Health Details:")
    for backend_name, backend_health in health["backend_health"].items():
        status = "✅" if backend_health["healthy"] else "❌"
        print(f"      {status} {backend_name} ({backend_health['role']})")
        if not backend_health["healthy"]:
            print(f"         Error: {backend_health.get('error', 'Unknown')}")


async def demonstrate_semantic_search(hybrid: FlexibleHybridPersistence):
    """Demonstrate semantic search if Milvus is available."""
    print("\n🔍 Semantic Search Demo")
    
    # Check if we have a Milvus backend
    milvus_backends = hybrid.get_backends_by_role(BackendRole.SEARCH)
    
    if not milvus_backends:
        print("   ⚠️  No search backends available")
        return
    
    milvus_backend = milvus_backends[0].backend
    
    if hasattr(milvus_backend, 'semantic_search'):
        try:
            results = await milvus_backend.semantic_search(
                query_text="renewable energy and sustainability",
                limit=5,
                score_threshold=0.5
            )
            
            print(f"   Found {len(results)} semantically similar thoughts:")
            for i, result in enumerate(results):
                print(f"      {i+1}. Score: {result['similarity_score']:.3f}")
                print(f"         Prompt: {result['prompt'][:50]}...")
                
        except Exception as e:
            print(f"   ❌ Semantic search failed: {e}")
    else:
        print("   ⚠️  Search backend doesn't support semantic search")


async def main():
    """Run the flexible hybrid storage demonstration."""
    print("🚀 FLEXIBLE HYBRID STORAGE DEMO")
    print("Demonstrating configurable multi-backend storage architecture")
    
    # Check API keys
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("❌ Missing API keys. Set ANTHROPIC_API_KEY or GOOGLE_API_KEY")
        return
    
    try:
        # Demonstrate simple hybrid
        simple_hybrid = await create_simple_hybrid()
        await demonstrate_hybrid_operations(simple_hybrid, "Simple Hybrid")
        await demonstrate_health_monitoring(simple_hybrid)
        
        # Demonstrate complex hybrid
        complex_hybrid = await create_complex_hybrid()
        await demonstrate_hybrid_operations(complex_hybrid, "Complex Hybrid")
        await demonstrate_health_monitoring(complex_hybrid)
        await demonstrate_semantic_search(complex_hybrid)
        
        print("\n" + "="*60)
        print("✅ FLEXIBLE HYBRID DEMO COMPLETED!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("• Configurable backend roles and priorities")
        print("• Automatic failover and read repair")
        print("• Per-backend statistics and monitoring")
        print("• Health checks for all backends")
        print("• Semantic search with vector storage")
        print("• Flexible architecture: Memory → Redis → PostgreSQL → File → Milvus")
        
        # Show configuration examples
        print("\n📋 Configuration Examples:")
        print("\nSimple Development Setup:")
        print("  Memory (Cache) → File (Primary)")
        
        print("\nProduction Setup:")
        print("  Memory (Cache) → Redis (Cache) → PostgreSQL (Primary) → File (Backup)")
        
        print("\nAI-Enhanced Setup:")
        print("  Memory → Redis → PostgreSQL → File → Milvus (Search)")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Flexible hybrid demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
