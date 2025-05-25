"""Thought history management for Sifaka.

This module contains the ThoughtHistory class which manages thought history
and references, optimizing memory usage and providing efficient access patterns.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sifaka.core.thought.thought import Thought, ThoughtReference
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ThoughtHistory:
    """Manages thought history and references.
    
    This class provides efficient management of thought history,
    using lightweight references to avoid memory issues while
    maintaining access to the complete chain of thoughts.
    """
    
    def __init__(self, max_in_memory: int = 10):
        """Initialize the thought history manager.
        
        Args:
            max_in_memory: Maximum number of thoughts to keep in memory.
        """
        self.max_in_memory = max_in_memory
        self._thoughts: Dict[str, Thought] = {}  # thought_id -> Thought
        self._references: Dict[str, List[ThoughtReference]] = {}  # chain_id -> references
        self._chain_order: Dict[str, List[str]] = {}  # chain_id -> ordered thought_ids
    
    def add_thought(self, thought: Thought) -> None:
        """Add a thought to the history.
        
        Args:
            thought: The thought to add.
        """
        chain_id = thought.chain_id or "default"
        
        # Add to thoughts cache
        self._thoughts[thought.id] = thought
        
        # Initialize chain tracking if needed
        if chain_id not in self._references:
            self._references[chain_id] = []
            self._chain_order[chain_id] = []
        
        # Add to chain order
        self._chain_order[chain_id].append(thought.id)
        
        # Create reference
        reference = ThoughtReference(
            thought_id=thought.id,
            iteration=thought.iteration,
            timestamp=thought.timestamp,
            summary=self._create_summary(thought)
        )
        self._references[chain_id].append(reference)
        
        # Cleanup old thoughts if we exceed memory limit
        self._cleanup_memory(chain_id)
        
        logger.debug(f"Added thought {thought.id} to history for chain {chain_id}")
    
    def get_thought(self, thought_id: str) -> Optional[Thought]:
        """Get a thought by ID.
        
        Args:
            thought_id: The thought ID.
            
        Returns:
            The thought if found, None otherwise.
        """
        return self._thoughts.get(thought_id)
    
    def get_latest_thought(self, chain_id: str) -> Optional[Thought]:
        """Get the latest thought for a chain.
        
        Args:
            chain_id: The chain ID.
            
        Returns:
            The latest thought if found, None otherwise.
        """
        if chain_id not in self._chain_order or not self._chain_order[chain_id]:
            return None
        
        latest_id = self._chain_order[chain_id][-1]
        return self._thoughts.get(latest_id)
    
    def get_thought_by_iteration(self, chain_id: str, iteration: int) -> Optional[Thought]:
        """Get a thought by chain ID and iteration.
        
        Args:
            chain_id: The chain ID.
            iteration: The iteration number.
            
        Returns:
            The thought if found, None otherwise.
        """
        references = self._references.get(chain_id, [])
        for ref in references:
            if ref.iteration == iteration:
                return self._thoughts.get(ref.thought_id)
        return None
    
    def get_chain_references(self, chain_id: str) -> List[ThoughtReference]:
        """Get all thought references for a chain.
        
        Args:
            chain_id: The chain ID.
            
        Returns:
            List of thought references in chronological order.
        """
        return self._references.get(chain_id, []).copy()
    
    def get_chain_thoughts(self, chain_id: str, include_cached_only: bool = False) -> List[Thought]:
        """Get all thoughts for a chain.
        
        Args:
            chain_id: The chain ID.
            include_cached_only: If True, only return thoughts currently in memory.
            
        Returns:
            List of thoughts in chronological order.
        """
        if chain_id not in self._chain_order:
            return []
        
        thoughts = []
        for thought_id in self._chain_order[chain_id]:
            thought = self._thoughts.get(thought_id)
            if thought:
                thoughts.append(thought)
            elif not include_cached_only:
                # Could implement loading from storage here
                logger.debug(f"Thought {thought_id} not in memory cache")
        
        return thoughts
    
    def get_iteration_count(self, chain_id: str) -> int:
        """Get the number of iterations for a chain.
        
        Args:
            chain_id: The chain ID.
            
        Returns:
            The number of iterations.
        """
        return len(self._references.get(chain_id, []))
    
    def remove_thought(self, thought_id: str) -> bool:
        """Remove a thought from history.
        
        Args:
            thought_id: The thought ID to remove.
            
        Returns:
            True if removed, False if not found.
        """
        thought = self._thoughts.get(thought_id)
        if not thought:
            return False
        
        chain_id = thought.chain_id or "default"
        
        # Remove from thoughts cache
        del self._thoughts[thought_id]
        
        # Remove from chain order
        if chain_id in self._chain_order:
            try:
                self._chain_order[chain_id].remove(thought_id)
            except ValueError:
                pass
        
        # Remove from references
        if chain_id in self._references:
            self._references[chain_id] = [
                ref for ref in self._references[chain_id]
                if ref.thought_id != thought_id
            ]
        
        logger.debug(f"Removed thought {thought_id} from history")
        return True
    
    def clear_chain(self, chain_id: str) -> int:
        """Clear all thoughts for a chain.
        
        Args:
            chain_id: The chain ID.
            
        Returns:
            Number of thoughts removed.
        """
        if chain_id not in self._chain_order:
            return 0
        
        thought_ids = self._chain_order[chain_id].copy()
        count = 0
        
        for thought_id in thought_ids:
            if self.remove_thought(thought_id):
                count += 1
        
        # Clean up empty chain entries
        if chain_id in self._references and not self._references[chain_id]:
            del self._references[chain_id]
        if chain_id in self._chain_order and not self._chain_order[chain_id]:
            del self._chain_order[chain_id]
        
        logger.debug(f"Cleared {count} thoughts for chain {chain_id}")
        return count
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage stats.
        """
        total_thoughts = len(self._thoughts)
        total_chains = len(self._references)
        total_references = sum(len(refs) for refs in self._references.values())
        
        return {
            "total_thoughts_in_memory": total_thoughts,
            "total_chains": total_chains,
            "total_references": total_references,
            "average_thoughts_per_chain": total_thoughts / total_chains if total_chains > 0 else 0
        }
    
    def _create_summary(self, thought: Thought) -> str:
        """Create a summary for a thought reference.
        
        Args:
            thought: The thought to summarize.
            
        Returns:
            A brief summary string.
        """
        text_length = len(thought.text or "")
        validation_count = len(thought.validation_results or {})
        feedback_count = len(thought.critic_feedback or [])
        
        return (
            f"Iteration {thought.iteration}: {text_length} chars, "
            f"{validation_count} validations, {feedback_count} feedback"
        )
    
    def _cleanup_memory(self, chain_id: str) -> None:
        """Clean up old thoughts from memory if we exceed the limit.
        
        Args:
            chain_id: The chain ID to clean up.
        """
        if chain_id not in self._chain_order:
            return
        
        chain_thoughts = self._chain_order[chain_id]
        if len(chain_thoughts) <= self.max_in_memory:
            return
        
        # Remove oldest thoughts from memory (but keep references)
        thoughts_to_remove = chain_thoughts[:-self.max_in_memory]
        for thought_id in thoughts_to_remove:
            if thought_id in self._thoughts:
                del self._thoughts[thought_id]
                logger.debug(f"Removed thought {thought_id} from memory cache")
    
    def optimize_memory(self) -> int:
        """Optimize memory usage by removing old thoughts.
        
        Returns:
            Number of thoughts removed from memory.
        """
        removed_count = 0
        for chain_id in self._chain_order:
            old_count = len([tid for tid in self._chain_order[chain_id] if tid in self._thoughts])
            self._cleanup_memory(chain_id)
            new_count = len([tid for tid in self._chain_order[chain_id] if tid in self._thoughts])
            removed_count += old_count - new_count
        
        logger.debug(f"Optimized memory: removed {removed_count} thoughts from cache")
        return removed_count
