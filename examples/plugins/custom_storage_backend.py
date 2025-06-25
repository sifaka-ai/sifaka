"""Example custom storage backend plugin for Sifaka.

This example shows how to create a Redis-based storage backend
that can be used with Sifaka for persistent storage of results.
"""

import json
import os
from typing import Optional, List
from datetime import datetime

# For this example, we'll use a simple file-based storage
# In a real implementation, you would use redis-py
# import redis

from sifaka.storage.base import StorageBackend
from sifaka.core.models import SifakaResult


class RedisStorageBackend(StorageBackend):
    """Redis-based storage backend for Sifaka results.

    This example uses file-based storage to simulate Redis.
    In production, replace with actual Redis client.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "sifaka:",
        ttl: Optional[int] = None,
    ):
        """Initialize Redis storage backend.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            prefix: Key prefix for all Sifaka entries
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self.host = host
        self.port = port
        self.db = db
        self.prefix = prefix
        self.ttl = ttl

        # In production:
        # self.client = redis.Redis(host=host, port=port, db=db)

        # For example, use file storage
        self.storage_dir = f".redis_mock_{host}_{port}_{db}"
        os.makedirs(self.storage_dir, exist_ok=True)

    async def save(self, result: SifakaResult) -> str:
        """Save a SifakaResult to Redis."""
        key = f"{self.prefix}{result.id}"

        # Serialize result to JSON
        data = {
            "id": result.id,
            "original_text": result.original_text,
            "final_text": result.final_text,
            "iteration": result.iteration,
            "generations": [g.dict() for g in result.generations],
            "critiques": [c.dict() for c in result.critiques],
            "validations": [v.dict() for v in result.validations],
            "total_cost": result.total_cost,
            "processing_time": result.processing_time,
            "created_at": result.created_at.isoformat(),
            "updated_at": result.updated_at.isoformat(),
        }

        # In production:
        # self.client.set(key, json.dumps(data), ex=self.ttl)

        # For example:
        with open(os.path.join(self.storage_dir, f"{result.id}.json"), "w") as f:
            json.dump(data, f)

        return result.id

    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load a SifakaResult from Redis."""
        key = f"{self.prefix}{result_id}"

        # In production:
        # data = self.client.get(key)
        # if not data:
        #     return None
        # data = json.loads(data)

        # For example:
        file_path = os.path.join(self.storage_dir, f"{result_id}.json")
        if not os.path.exists(file_path):
            return None

        with open(file_path, "r") as f:
            data = json.load(f)

        # Reconstruct SifakaResult
        return SifakaResult(
            id=data["id"],
            original_text=data["original_text"],
            final_text=data["final_text"],
            iteration=data["iteration"],
            generations=[Generation(**g) for g in data["generations"]],
            critiques=[CritiqueResult(**c) for c in data["critiques"]],
            validations=[ValidationResult(**v) for v in data["validations"]],
            total_cost=data["total_cost"],
            processing_time=data["processing_time"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List stored result IDs."""
        # In production:
        # pattern = f"{self.prefix}*"
        # keys = self.client.keys(pattern)
        # return [k.decode().replace(self.prefix, "") for k in keys[offset:offset+limit]]

        # For example:
        files = os.listdir(self.storage_dir)
        result_ids = [f.replace(".json", "") for f in files if f.endswith(".json")]
        return result_ids[offset : offset + limit]

    async def delete(self, result_id: str) -> bool:
        """Delete a stored result."""
        key = f"{self.prefix}{result_id}"

        # In production:
        # return bool(self.client.delete(key))

        # For example:
        file_path = os.path.join(self.storage_dir, f"{result_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    async def search(self, query: str, limit: int = 10) -> List[str]:
        """Search stored results by text content.

        Note: This is a simple implementation. In production,
        you might want to use Redis Search or maintain a
        separate search index.
        """
        results = []

        # For example: search through all files
        for file_name in os.listdir(self.storage_dir):
            if not file_name.endswith(".json"):
                continue

            file_path = os.path.join(self.storage_dir, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)

            # Simple text search
            if (
                query.lower() in data.get("original_text", "").lower()
                or query.lower() in data.get("final_text", "").lower()
            ):
                results.append(data["id"])

                if len(results) >= limit:
                    break

        return results


# To use this plugin, either:
# 1. Register it manually in your code:
#    from sifaka import register_storage_backend
#    register_storage_backend("redis", RedisStorageBackend)
#
# 2. Or install it as a package with entry points in setup.py:
#    entry_points={
#        "sifaka.storage": [
#            "redis = my_plugin:RedisStorageBackend",
#        ],
#    }


if __name__ == "__main__":
    # Example usage
    from sifaka import register_storage_backend, create_storage_backend

    # Register the backend
    register_storage_backend("redis", RedisStorageBackend)

    # Create an instance
    storage = create_storage_backend(
        "redis",
        host="localhost",
        port=6379,
        prefix="sifaka:example:",
        ttl=3600,  # 1 hour expiration
    )

    print(f"Created Redis storage backend: {storage}")
    print(f"Storage directory: {storage.storage_dir}")
