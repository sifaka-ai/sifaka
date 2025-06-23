"""Caching layer for Sifaka to improve performance."""

import hashlib
import json
import time
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: float
    hits: int = 0
    metadata: Optional[Dict[str, Any]] = None


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache with TTL in seconds."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class InMemoryCache(CacheBackend):
    """Simple in-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.misses += 1
                return None
            
            # Check if expired
            if time.time() > entry.expires_at:
                del self.cache[key]
                self.misses += 1
                return None
            
            # Update stats
            entry.hits += 1
            self.hits += 1
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache with TTL."""
        async with self._lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                await self._evict_lru()
            
            self.cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=time.time() + ttl,
                hits=0
            )
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        async with self._lock:
            self.cache.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Find LRU entry (least hits, oldest)
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (self.cache[k].hits, self.cache[k].created_at)
        )
        del self.cache[lru_key]


class CriticCache:
    """Cache specifically for critic responses."""
    
    def __init__(self, backend: Optional[CacheBackend] = None, ttl: int = 3600):
        self.backend = backend or InMemoryCache()
        self.ttl = ttl
        self.enabled = True
    
    def _generate_key(
        self,
        text: str,
        critic_name: str,
        model: str,
        temperature: float,
        iteration: int = 0
    ) -> str:
        """Generate cache key for critic response."""
        # Create deterministic key
        key_data = {
            "text": text[:1000],  # Limit text length for key
            "critic": critic_name,
            "model": model,
            "temperature": temperature,
            "iteration": iteration
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get_critique(
        self,
        text: str,
        critic_name: str,
        model: str,
        temperature: float,
        iteration: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Get cached critique if available."""
        if not self.enabled:
            return None
        
        key = self._generate_key(text, critic_name, model, temperature, iteration)
        result = await self.backend.get(key)
        
        if result:
            logger.debug(f"Cache hit for critic {critic_name}")
        
        return result
    
    async def set_critique(
        self,
        text: str,
        critic_name: str,
        model: str,
        temperature: float,
        critique_data: Dict[str, Any],
        iteration: int = 0
    ) -> None:
        """Cache critique response."""
        if not self.enabled:
            return
        
        key = self._generate_key(text, critic_name, model, temperature, iteration)
        await self.backend.set(key, critique_data, self.ttl)
        logger.debug(f"Cached critique for {critic_name}")
    
    async def clear(self) -> None:
        """Clear all cached critiques."""
        await self.backend.clear()
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.backend.stats()


# Global cache instance (optional)
_global_cache: Optional[CriticCache] = None


def get_global_cache() -> Optional[CriticCache]:
    """Get global cache instance."""
    return _global_cache


def set_global_cache(cache: Optional[CriticCache]) -> None:
    """Set global cache instance."""
    global _global_cache
    _global_cache = cache


def enable_caching(ttl: int = 3600, max_size: int = 1000) -> CriticCache:
    """Enable global caching with in-memory backend."""
    cache = CriticCache(
        backend=InMemoryCache(max_size=max_size),
        ttl=ttl
    )
    set_global_cache(cache)
    return cache


def disable_caching() -> None:
    """Disable global caching."""
    set_global_cache(None)