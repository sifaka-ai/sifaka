"""Tools for thought retrieval and search operations.

This module provides tools that can be used by PydanticAI agents
to search and retrieve thoughts from storage backends.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic_ai.tools import Tool

if TYPE_CHECKING:
    from sifaka.storage.base import SifakaBasePersistence

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ThoughtRetrievalTools:
    """Collection of tools for thought retrieval and search.

    These tools can be used by PydanticAI agents to access stored thoughts
    and perform various search operations.
    """

    def __init__(self, persistence: "SifakaBasePersistence"):
        """Initialize retrieval tools with a persistence backend.

        Args:
            persistence: The persistence backend to use for retrieval
        """
        self.persistence = persistence
        logger.debug(f"Initialized ThoughtRetrievalTools with {type(persistence).__name__}")

    async def search_thoughts_by_text(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search thoughts by text content.

        Performs a simple text search across thought content, looking for
        the query string in the prompt, current_text, and final_text fields.

        Args:
            query: Text to search for
            limit: Maximum number of results to return

        Returns:
            List of thought summaries matching the query
        """
        try:
            # Get all thoughts
            all_thoughts = await self.persistence.list_thoughts(limit=None)

            # Filter by text content
            matching_thoughts = []
            query_lower = query.lower()

            for thought in all_thoughts:
                # Search in prompt
                if query_lower in thought.prompt.lower():
                    matching_thoughts.append(thought)
                    continue

                # Search in current text
                if thought.current_text and query_lower in thought.current_text.lower():
                    matching_thoughts.append(thought)
                    continue

                # Search in final text
                if thought.final_text and query_lower in thought.final_text.lower():
                    matching_thoughts.append(thought)
                    continue

            # Apply limit and return summaries
            limited_thoughts = matching_thoughts[:limit]
            results = [thought.get_summary() for thought in limited_thoughts]

            logger.debug(f"Text search for '{query}' found {len(results)} thoughts")
            return results

        except Exception as e:
            logger.error(f"Failed to search thoughts by text: {e}")
            return []

    async def search_thoughts_by_conversation(
        self, conversation_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search thoughts by conversation ID.

        Retrieves all thoughts that belong to a specific conversation,
        ordered by creation time.

        Args:
            conversation_id: The conversation ID to search for
            limit: Maximum number of results to return

        Returns:
            List of thought summaries in the conversation
        """
        try:
            thoughts = await self.persistence.list_thoughts(
                conversation_id=conversation_id, limit=limit
            )

            results = [thought.get_summary() for thought in thoughts]

            logger.debug(
                f"Conversation search for '{conversation_id}' found {len(results)} thoughts"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to search thoughts by conversation: {e}")
            return []

    async def get_thought_details(self, thought_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific thought.

        Retrieves the complete thought object with all audit trail information.

        Args:
            thought_id: The thought ID to retrieve

        Returns:
            Complete thought data if found, None otherwise
        """
        try:
            thought = await self.persistence.retrieve_thought(thought_id)

            if thought is None:
                logger.debug(f"Thought not found: {thought_id}")
                return None

            # Return full thought data as dictionary
            result = thought.model_dump()

            logger.debug(f"Retrieved thought details: {thought_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to get thought details: {e}")
            return None

    async def search_thoughts_by_technique(
        self, technique: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search thoughts by applied research technique.

        Finds thoughts that have used a specific research technique
        (e.g., "reflexion", "constitutional", "self_refine").

        Args:
            technique: The research technique to search for
            limit: Maximum number of results to return

        Returns:
            List of thought summaries that used the technique
        """
        try:
            # Get all thoughts
            all_thoughts = await self.persistence.list_thoughts(limit=None)

            # Filter by technique
            matching_thoughts = []

            for thought in all_thoughts:
                if technique in thought.techniques_applied:
                    matching_thoughts.append(thought)

            # Apply limit and return summaries
            limited_thoughts = matching_thoughts[:limit]
            results = [thought.get_summary() for thought in limited_thoughts]

            logger.debug(f"Technique search for '{technique}' found {len(results)} thoughts")
            return results

        except Exception as e:
            logger.error(f"Failed to search thoughts by technique: {e}")
            return []

    async def search_thoughts_by_validation_status(
        self, passed: bool, validator_name: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search thoughts by validation status.

        Finds thoughts based on whether they passed or failed validation,
        optionally filtered by a specific validator.

        Args:
            passed: Whether to search for passed (True) or failed (False) validations
            validator_name: Optional specific validator to filter by
            limit: Maximum number of results to return

        Returns:
            List of thought summaries matching the validation criteria
        """
        try:
            # Get all thoughts
            all_thoughts = await self.persistence.list_thoughts(limit=None)

            # Filter by validation status
            matching_thoughts = []

            for thought in all_thoughts:
                for validation in thought.validations:
                    # Check validation status
                    if validation.passed != passed:
                        continue

                    # Check specific validator if specified
                    if validator_name and validation.validator != validator_name:
                        continue

                    # This thought matches
                    matching_thoughts.append(thought)
                    break  # Don't add the same thought multiple times

            # Apply limit and return summaries
            limited_thoughts = matching_thoughts[:limit]
            results = [thought.get_summary() for thought in limited_thoughts]

            status_str = "passed" if passed else "failed"
            validator_str = f" by {validator_name}" if validator_name else ""
            logger.debug(
                f"Validation search for {status_str}{validator_str} found {len(results)} thoughts"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to search thoughts by validation status: {e}")
            return []

    async def get_thought_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored thoughts.

        Provides overview statistics about the thought storage,
        including counts, techniques used, and validation results.

        Returns:
            Dictionary with storage statistics
        """
        try:
            # Get all thoughts
            all_thoughts = await self.persistence.list_thoughts(limit=None)

            # Calculate statistics
            total_thoughts = len(all_thoughts)
            finalized_thoughts = len([t for t in all_thoughts if t.final_text is not None])

            # Technique statistics
            technique_counts = {}
            for thought in all_thoughts:
                for technique in thought.techniques_applied:
                    technique_counts[technique] = technique_counts.get(technique, 0) + 1

            # Validation statistics
            validation_counts = {"passed": 0, "failed": 0}
            validator_counts = {}

            for thought in all_thoughts:
                for validation in thought.validations:
                    if validation.passed:
                        validation_counts["passed"] += 1
                    else:
                        validation_counts["failed"] += 1

                    validator_counts[validation.validator] = (
                        validator_counts.get(validation.validator, 0) + 1
                    )

            # Critique statistics
            critique_counts = {}
            for thought in all_thoughts:
                for critique in thought.critiques:
                    critique_counts[critique.critic] = critique_counts.get(critique.critic, 0) + 1

            # Conversation statistics
            conversation_ids = set()
            for thought in all_thoughts:
                if thought.conversation_id:
                    conversation_ids.add(thought.conversation_id)

            stats = {
                "total_thoughts": total_thoughts,
                "finalized_thoughts": finalized_thoughts,
                "active_conversations": len(conversation_ids),
                "techniques_used": technique_counts,
                "validation_results": validation_counts,
                "validator_usage": validator_counts,
                "critic_usage": critique_counts,
            }

            # Add storage backend statistics if available
            if hasattr(self.persistence, "get_stats"):
                try:
                    backend_stats = await self.persistence.get_stats()
                    stats["backend_stats"] = backend_stats
                except Exception as e:
                    logger.warning(f"Failed to get backend stats: {e}")

            logger.debug(f"Generated thought statistics: {total_thoughts} total thoughts")
            return stats

        except Exception as e:
            logger.error(f"Failed to get thought statistics: {e}")
            return {"error": str(e)}


def create_retrieval_tools(persistence: "SifakaBasePersistence") -> List[Tool]:
    """Create a list of tools for thought retrieval.

    Args:
        persistence: The persistence backend to use

    Returns:
        List of Tool instances that can be used by PydanticAI agents
    """
    tools_instance = ThoughtRetrievalTools(persistence)

    return [
        Tool(tools_instance.search_thoughts_by_text, takes_ctx=False),
        Tool(tools_instance.search_thoughts_by_conversation, takes_ctx=False),
        Tool(tools_instance.get_thought_details, takes_ctx=False),
        Tool(tools_instance.search_thoughts_by_technique, takes_ctx=False),
        Tool(tools_instance.search_thoughts_by_validation_status, takes_ctx=False),
        Tool(tools_instance.get_thought_statistics, takes_ctx=False),
    ]
