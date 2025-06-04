"""Thought-specific retrieval tools for Sifaka.

This module provides tools for searching and retrieving Sifaka thoughts
with advanced filtering and analysis capabilities.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING

from pydantic_ai.tools import Tool
from sifaka.tools.base import BaseSifakaTool, register_tool
from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from sifaka.storage.base import SifakaBasePersistence

logger = get_logger(__name__)


@register_tool("thought_retrieval")
class ThoughtRetrievalTool(BaseSifakaTool):
    """Advanced thought retrieval and analysis tool."""
    
    def __init__(self, persistence: "SifakaBasePersistence", **kwargs):
        super().__init__(
            name="thought_retrieval",
            description="Advanced retrieval and analysis of Sifaka thoughts",
            category="retrieval",
            provider="sifaka",
            **kwargs
        )
        self.persistence = persistence
    
    async def search_thoughts_by_text(
        self, 
        query: str, 
        limit: int = 50,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search thoughts by text content with relevance scoring.
        
        Args:
            query: Text to search for
            limit: Maximum number of results
            min_score: Minimum relevance score (0.0-1.0)
            
        Returns:
            List of thoughts with relevance scores
        """
        try:
            thoughts = await self.persistence.list_thoughts(limit=limit * 2)  # Get more for filtering
            
            results = []
            query_lower = query.lower()
            
            for thought in thoughts:
                # Simple relevance scoring
                text_lower = thought.final_text.lower()
                score = 0.0
                
                # Exact phrase match
                if query_lower in text_lower:
                    score += 0.8
                
                # Word matches
                query_words = query_lower.split()
                text_words = text_lower.split()
                word_matches = sum(1 for word in query_words if word in text_words)
                score += (word_matches / len(query_words)) * 0.5
                
                if score >= min_score:
                    result = thought.model_dump()
                    result["relevance_score"] = score
                    results.append(result)
            
            # Sort by relevance and limit
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    async def search_thoughts_by_conversation(
        self, 
        conversation_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search thoughts by conversation ID.
        
        Args:
            conversation_id: The conversation ID to search for
            limit: Maximum number of results
            
        Returns:
            List of thoughts in the conversation
        """
        try:
            thoughts = await self.persistence.list_thoughts(
                conversation_id=conversation_id,
                limit=limit
            )
            return [thought.model_dump() for thought in thoughts]
        except Exception as e:
            logger.error(f"Conversation search failed: {e}")
            return []
    
    async def search_thoughts_by_validation_status(
        self, 
        passed: bool, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search thoughts by validation status.
        
        Args:
            passed: Whether to find thoughts that passed (True) or failed (False) validation
            limit: Maximum number of results
            
        Returns:
            List of thoughts with the specified validation status
        """
        try:
            thoughts = await self.persistence.list_thoughts(limit=limit * 2)
            
            results = []
            for thought in thoughts:
                if thought.validation_passed() == passed:
                    results.append(thought.model_dump())
                    if len(results) >= limit:
                        break
            
            return results
        except Exception as e:
            logger.error(f"Validation status search failed: {e}")
            return []
    
    async def search_thoughts_by_technique(
        self, 
        technique: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search thoughts by critic technique used.
        
        Args:
            technique: Name of the critic technique (e.g., 'reflexion', 'constitutional')
            limit: Maximum number of results
            
        Returns:
            List of thoughts that used the specified technique
        """
        try:
            thoughts = await self.persistence.list_thoughts(limit=limit * 2)
            
            results = []
            for thought in thoughts:
                # Check if any critique result mentions the technique
                for critique in thought.critique_results:
                    if technique.lower() in critique.get("critic_name", "").lower():
                        results.append(thought.model_dump())
                        break
                
                if len(results) >= limit:
                    break
            
            return results
        except Exception as e:
            logger.error(f"Technique search failed: {e}")
            return []
    
    async def get_thought_details(self, thought_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific thought.
        
        Args:
            thought_id: The thought ID to retrieve
            
        Returns:
            Complete thought data with analysis
        """
        try:
            thought = await self.persistence.retrieve_thought(thought_id)
            if not thought:
                return None
            
            # Add analysis data
            result = thought.model_dump()
            result["analysis"] = {
                "total_iterations": thought.iteration,
                "validation_passed": thought.validation_passed(),
                "critic_count": len(thought.critique_results),
                "validator_count": len(thought.validation_results),
                "tool_calls_count": len(thought.tool_calls),
                "has_parent": thought.parent_id is not None,
                "has_children": len(thought.child_ids) > 0,
                "conversation_length": len(thought.child_ids) + 1,
            }
            
            return result
        except Exception as e:
            logger.error(f"Thought details retrieval failed: {e}")
            return None
    
    async def get_thought_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about stored thoughts.
        
        Returns:
            Dictionary with various statistics
        """
        try:
            # Get a sample of thoughts for analysis
            thoughts = await self.persistence.list_thoughts(limit=1000)
            
            if not thoughts:
                return {"total_thoughts": 0}
            
            stats = {
                "total_thoughts": len(thoughts),
                "validation_passed": sum(1 for t in thoughts if t.validation_passed()),
                "validation_failed": sum(1 for t in thoughts if not t.validation_passed()),
                "avg_iterations": sum(t.iteration for t in thoughts) / len(thoughts),
                "max_iterations": max(t.iteration for t in thoughts),
                "thoughts_with_critics": sum(1 for t in thoughts if t.critique_results),
                "thoughts_with_tools": sum(1 for t in thoughts if t.tool_calls),
                "unique_conversations": len(set(t.conversation_id for t in thoughts if t.conversation_id)),
            }
            
            # Calculate success rate
            if stats["total_thoughts"] > 0:
                stats["validation_success_rate"] = stats["validation_passed"] / stats["total_thoughts"]
            
            return stats
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {"error": str(e)}
    
    def create_pydantic_tools(self) -> List[Tool]:
        """Create thought retrieval tools."""
        return [
            Tool(self.search_thoughts_by_text, takes_ctx=False),
            Tool(self.search_thoughts_by_conversation, takes_ctx=False),
            Tool(self.search_thoughts_by_validation_status, takes_ctx=False),
            Tool(self.search_thoughts_by_technique, takes_ctx=False),
            Tool(self.get_thought_details, takes_ctx=False),
            Tool(self.get_thought_statistics, takes_ctx=False),
        ]


def create_thought_retrieval_tools(
    persistence: "SifakaBasePersistence",
    **kwargs
) -> List[Tool]:
    """Create thought retrieval tools.
    
    Args:
        persistence: Storage persistence backend
        **kwargs: Additional configuration options
        
    Returns:
        List of PydanticAI Tool instances
        
    Example:
        ```python
        from sifaka.storage import MemoryPersistence
        
        storage = MemoryPersistence()
        tools = create_thought_retrieval_tools(storage)
        ```
    """
    tool = ThoughtRetrievalTool(persistence, **kwargs)
    return tool.create_pydantic_tools()
