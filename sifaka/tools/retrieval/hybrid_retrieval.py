"""Hybrid retrieval tools that combine multiple sources and backends.

This module provides tools that can search across multiple storage backends
and combine results from different sources for comprehensive retrieval.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING

from pydantic_ai.tools import Tool
from sifaka.tools.base import BaseSifakaTool, register_tool
from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from sifaka.storage.base import SifakaBasePersistence

logger = get_logger(__name__)


@register_tool("hybrid_retrieval")
class HybridRetrievalTool(BaseSifakaTool):
    """Hybrid retrieval tool that combines multiple storage backends and sources."""
    
    def __init__(
        self, 
        primary_persistence: "SifakaBasePersistence",
        secondary_persistence: Optional["SifakaBasePersistence"] = None,
        enable_web_search: bool = False,
        web_search_providers: List[str] = None,
        **kwargs
    ):
        super().__init__(
            name="hybrid_retrieval",
            description="Hybrid retrieval across multiple storage backends and web sources",
            category="retrieval",
            provider="hybrid",
            **kwargs
        )
        self.primary_persistence = primary_persistence
        self.secondary_persistence = secondary_persistence
        self.enable_web_search = enable_web_search
        self.web_search_providers = web_search_providers or ["duckduckgo"]
        
        # Initialize web search tools if enabled
        self.web_tools = []
        if enable_web_search:
            try:
                from sifaka.tools.retrieval.web_search import create_web_search_tools
                self.web_tools = create_web_search_tools(providers=self.web_search_providers)
            except Exception as e:
                logger.warning(f"Failed to initialize web search tools: {e}")
    
    async def search_hybrid(
        self, 
        query: str, 
        limit: int = 50,
        include_web: bool = False,
        sources: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search across multiple sources and combine results.
        
        Args:
            query: Search query
            limit: Maximum results per source
            include_web: Whether to include web search results
            sources: Specific sources to search (default: all available)
            
        Returns:
            Dictionary with results from each source
        """
        results = {
            "primary_storage": [],
            "secondary_storage": [],
            "web_search": [],
            "combined_score": []
        }
        
        # Search primary storage
        try:
            thoughts = await self.primary_persistence.list_thoughts(limit=limit)
            for thought in thoughts:
                if query.lower() in thought.final_text.lower():
                    result = thought.model_dump()
                    result["source"] = "primary_storage"
                    result["relevance_score"] = self._calculate_relevance(query, thought.final_text)
                    results["primary_storage"].append(result)
        except Exception as e:
            logger.error(f"Primary storage search failed: {e}")
        
        # Search secondary storage if available
        if self.secondary_persistence:
            try:
                thoughts = await self.secondary_persistence.list_thoughts(limit=limit)
                for thought in thoughts:
                    if query.lower() in thought.final_text.lower():
                        result = thought.model_dump()
                        result["source"] = "secondary_storage"
                        result["relevance_score"] = self._calculate_relevance(query, thought.final_text)
                        results["secondary_storage"].append(result)
            except Exception as e:
                logger.error(f"Secondary storage search failed: {e}")
        
        # Web search if enabled and requested
        if include_web and self.enable_web_search and self.web_tools:
            try:
                # Note: This is a simplified example - in practice you'd need to
                # properly execute the web search tools through PydanticAI
                web_results = []
                results["web_search"] = web_results
            except Exception as e:
                logger.error(f"Web search failed: {e}")
        
        # Combine and rank results
        all_results = []
        for source_results in [results["primary_storage"], results["secondary_storage"]]:
            all_results.extend(source_results)
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        results["combined_score"] = all_results[:limit]
        
        return results
    
    async def search_with_fallback(
        self, 
        query: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search with automatic fallback between storage backends.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results from the first successful source
        """
        # Try primary storage first
        try:
            thoughts = await self.primary_persistence.list_thoughts(limit=limit)
            results = []
            for thought in thoughts:
                if query.lower() in thought.final_text.lower():
                    result = thought.model_dump()
                    result["source"] = "primary_storage"
                    results.append(result)
            
            if results:
                return results
        except Exception as e:
            logger.warning(f"Primary storage search failed, trying fallback: {e}")
        
        # Fallback to secondary storage
        if self.secondary_persistence:
            try:
                thoughts = await self.secondary_persistence.list_thoughts(limit=limit)
                results = []
                for thought in thoughts:
                    if query.lower() in thought.final_text.lower():
                        result = thought.model_dump()
                        result["source"] = "secondary_storage"
                        results.append(result)
                
                return results
            except Exception as e:
                logger.error(f"Secondary storage search also failed: {e}")
        
        return []
    
    async def get_cross_backend_analytics(self) -> Dict[str, Any]:
        """Get analytics across all configured storage backends.
        
        Returns:
            Combined analytics from all backends
        """
        analytics = {
            "primary_storage": {},
            "secondary_storage": {},
            "total_across_backends": {}
        }
        
        # Primary storage analytics
        try:
            primary_thoughts = await self.primary_persistence.list_thoughts(limit=1000)
            analytics["primary_storage"] = {
                "total_thoughts": len(primary_thoughts),
                "validation_passed": sum(1 for t in primary_thoughts if t.validation_passed()),
                "avg_iterations": sum(t.iteration for t in primary_thoughts) / len(primary_thoughts) if primary_thoughts else 0,
            }
        except Exception as e:
            logger.error(f"Primary storage analytics failed: {e}")
            analytics["primary_storage"] = {"error": str(e)}
        
        # Secondary storage analytics
        if self.secondary_persistence:
            try:
                secondary_thoughts = await self.secondary_persistence.list_thoughts(limit=1000)
                analytics["secondary_storage"] = {
                    "total_thoughts": len(secondary_thoughts),
                    "validation_passed": sum(1 for t in secondary_thoughts if t.validation_passed()),
                    "avg_iterations": sum(t.iteration for t in secondary_thoughts) / len(secondary_thoughts) if secondary_thoughts else 0,
                }
            except Exception as e:
                logger.error(f"Secondary storage analytics failed: {e}")
                analytics["secondary_storage"] = {"error": str(e)}
        
        # Combined analytics
        primary_total = analytics["primary_storage"].get("total_thoughts", 0)
        secondary_total = analytics["secondary_storage"].get("total_thoughts", 0)
        
        analytics["total_across_backends"] = {
            "total_thoughts": primary_total + secondary_total,
            "backends_configured": 2 if self.secondary_persistence else 1,
            "web_search_enabled": self.enable_web_search,
        }
        
        return analytics
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text.
        
        Args:
            query: Search query
            text: Text to score
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        query_lower = query.lower()
        text_lower = text.lower()
        
        score = 0.0
        
        # Exact phrase match
        if query_lower in text_lower:
            score += 0.8
        
        # Word matches
        query_words = query_lower.split()
        text_words = text_lower.split()
        word_matches = sum(1 for word in query_words if word in text_words)
        
        if query_words:
            score += (word_matches / len(query_words)) * 0.5
        
        return min(score, 1.0)
    
    def create_pydantic_tools(self) -> List[Tool]:
        """Create hybrid retrieval tools."""
        tools = [
            Tool(self.search_hybrid, takes_ctx=False),
            Tool(self.search_with_fallback, takes_ctx=False),
            Tool(self.get_cross_backend_analytics, takes_ctx=False),
        ]
        
        # Add web search tools if available
        tools.extend(self.web_tools)
        
        return tools


def create_hybrid_retrieval_tools(
    primary_persistence: "SifakaBasePersistence",
    secondary_persistence: Optional["SifakaBasePersistence"] = None,
    enable_web_search: bool = False,
    web_search_providers: List[str] = None,
    **kwargs
) -> List[Tool]:
    """Create hybrid retrieval tools.
    
    Args:
        primary_persistence: Primary storage backend
        secondary_persistence: Optional secondary storage backend
        enable_web_search: Whether to enable web search capabilities
        web_search_providers: List of web search providers to use
        **kwargs: Additional configuration options
        
    Returns:
        List of PydanticAI Tool instances
        
    Example:
        ```python
        from sifaka.storage import RedisPersistence, PostgreSQLPersistence
        
        redis_storage = RedisPersistence(...)
        postgres_storage = PostgreSQLPersistence(...)
        
        tools = create_hybrid_retrieval_tools(
            primary_persistence=redis_storage,
            secondary_persistence=postgres_storage,
            enable_web_search=True,
            web_search_providers=["duckduckgo", "tavily"]
        )
        ```
    """
    tool = HybridRetrievalTool(
        primary_persistence=primary_persistence,
        secondary_persistence=secondary_persistence,
        enable_web_search=enable_web_search,
        web_search_providers=web_search_providers,
        **kwargs
    )
    return tool.create_pydantic_tools()
