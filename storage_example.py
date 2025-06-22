"""Example showing how to use Sifaka with different storage backends."""

import asyncio
from sifaka import improve, FileStorage, MemoryStorage, create_storage_backend


async def memory_storage_example():
    """Example with in-memory storage (default)."""
    print("🧠 Memory Storage Example")

    # Default behavior - uses MemoryStorage
    result = await improve(
        "Write about artificial intelligence benefits",
        max_iterations=2,
        critics=["reflexion"],
    )

    print(f"Result ID: {result.id}")
    print(f"Stored in memory (non-persistent)")
    print()


async def file_storage_example():
    """Example with file-based persistent storage."""
    print("💾 File Storage Example")

    # Create file storage backend
    file_storage = FileStorage(storage_dir="./my_thoughts")

    result = await improve(
        "Explain quantum computing concepts",
        max_iterations=3,
        critics=["constitutional", "reflexion"],
        storage=file_storage,
    )

    print(f"Result ID: {result.id}")
    print(f"Saved to: ./my_thoughts/{result.id}.json")

    # Load the result back
    loaded_result = await file_storage.load(result.id)
    print(f"Loaded back: {loaded_result.final_text[:100]}...")

    # List all stored thoughts
    all_ids = await file_storage.list()
    print(f"Total stored thoughts: {len(all_ids)}")
    print()


async def plugin_storage_example():
    """Example with plugin-based storage backends."""
    print("🔌 Plugin Storage Examples")

    # Example 1: Redis plugin (if available)
    try:
        redis_storage = create_storage_backend("redis", url="redis://localhost:6379")

        result = await improve(
            "Write about sustainable energy solutions",
            max_iterations=2,
            critics=["constitutional"],
            storage=redis_storage,
        )

        print(f"✅ Redis: Stored {result.id}")

    except (KeyError, ImportError, Exception) as e:
        print(f"⚠️  Redis plugin not available: {e}")
        print("   Install with: pip install sifaka-redis")

    # Example 2: Mem0 plugin (if available)
    try:
        mem0_storage = create_storage_backend("mem0", user_id="demo_user")

        result = await improve(
            "Explain quantum computing concepts",
            max_iterations=2,
            critics=["self_rag"],
            storage=mem0_storage,
        )

        print(f"✅ Mem0: Stored {result.id} with semantic indexing")

    except (KeyError, ImportError, Exception) as e:
        print(f"⚠️  Mem0 plugin not available: {e}")
        print("   Install with: pip install sifaka-mem0")

    print()


async def storage_management_example():
    """Example showing storage management operations."""
    print("🔧 Storage Management Example")

    file_storage = FileStorage(storage_dir="./demo_thoughts")

    # Create multiple thoughts
    thoughts = []
    for i in range(3):
        result = await improve(
            f"Write about topic {i+1}",
            max_iterations=1,
            critics=["reflexion"],
            storage=file_storage,
        )
        thoughts.append(result.id)

    print(f"Created {len(thoughts)} thoughts")

    # List all thoughts
    all_ids = await file_storage.list()
    print(f"Total thoughts in storage: {len(all_ids)}")

    # Search for specific content
    search_results = await file_storage.search("topic", limit=10)
    print(f"Search results for 'topic': {len(search_results)} found")

    # Get metadata for a thought
    if all_ids:
        metadata = await file_storage.get_metadata(all_ids[0])
        print(f"Metadata example: {metadata}")

    # Cleanup - delete one thought
    if thoughts:
        deleted = await file_storage.delete(thoughts[0])
        print(f"Deleted thought: {deleted}")

    # Final count
    remaining = await file_storage.list()
    print(f"Remaining thoughts: {len(remaining)}")
    print()


async def custom_storage_example():
    """Example showing how to create a custom storage backend."""
    print("🔧 Custom Storage Backend Example")

    from sifaka import StorageBackend, register_storage_backend
    from sifaka.core.models import SifakaResult
    from typing import Optional, List

    class LoggingStorage(StorageBackend):
        """Custom storage that logs all operations."""

        def __init__(self):
            self.storage = {}

        async def save(self, result: SifakaResult) -> str:
            print(f"📝 Saving thought {result.id}: {result.final_text[:50]}...")
            self.storage[result.id] = result
            return result.id

        async def load(self, result_id: str) -> Optional[SifakaResult]:
            print(f"📖 Loading thought {result_id}")
            return self.storage.get(result_id)

        async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
            return list(self.storage.keys())[offset : offset + limit]

        async def delete(self, result_id: str) -> bool:
            if result_id in self.storage:
                print(f"🗑️  Deleting thought {result_id}")
                del self.storage[result_id]
                return True
            return False

        async def search(self, query: str, limit: int = 10) -> List[str]:
            print(f"🔍 Searching for: {query}")
            matches = []
            for result_id, result in self.storage.items():
                if query.lower() in result.final_text.lower():
                    matches.append(result_id)
                if len(matches) >= limit:
                    break
            return matches

    # Register and use the custom storage
    register_storage_backend("logging", LoggingStorage)
    custom_storage = create_storage_backend("logging")

    result = await improve(
        "Explain machine learning basics",
        max_iterations=2,
        critics=["reflexion"],
        storage=custom_storage,
    )

    print(f"Result with custom storage: {result.id}")
    print()


async def main():
    """Run all storage examples."""
    print("🎉 Sifaka Storage Examples\n")

    # Show available storage backends
    from sifaka import list_storage_backends

    print(f"📦 Available storage backends: {list_storage_backends()}\n")

    await memory_storage_example()
    await file_storage_example()
    await plugin_storage_example()
    await storage_management_example()
    await custom_storage_example()

    print("✅ All storage examples completed!")
    print("\n💡 Key Takeaways:")
    print("   • MemoryStorage: Fast, non-persistent (default)")
    print("   • FileStorage: Persistent, searchable, local files")
    print("   • Plugin system: Install only what you need")
    print("   • Custom storage: Implement StorageBackend for any system")
    print("   • All storage backends support save/load/search/delete operations")
    print("\n📚 Plugin Installation:")
    print("   pip install sifaka-redis    # Redis support")
    print("   pip install sifaka-mem0     # Semantic memory")
    print("   pip install sifaka-mcp      # MCP protocol")


if __name__ == "__main__":
    asyncio.run(main())
