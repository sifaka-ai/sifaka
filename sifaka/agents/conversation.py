"""PydanticAI conversation history integration for Sifaka.

This module provides adapters and utilities for integrating Sifaka's thought-based
workflow with PydanticAI's native conversation history system.

The key insight is to leverage PydanticAI's conversation memory instead of building
custom storage, while providing seamless conversion between Sifaka thoughts and
PydanticAI messages.

Example:
    ```python
    from pydantic_ai import Agent
    from sifaka.agents.conversation import ConversationHistoryAdapter
    from sifaka.agents import create_pydantic_chain

    # Create agent with conversation memory
    agent = Agent("openai:gpt-4", system_prompt="You are a helpful assistant.")

    # Create chain with conversation history integration
    chain = create_pydantic_chain(agent=agent, validators=[validator])

    # Run multiple iterations - agent remembers previous context
    result1 = chain.run("Write a story about AI")
    result2 = chain.run("Make it more dramatic")  # Remembers the previous story

    # Access conversation history
    adapter = ConversationHistoryAdapter(agent)
    history = adapter.get_conversation_summary()
    ```
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# PydanticAI is a required dependency
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage


class ConversationHistoryAdapter:
    """Simplified adapter for read-only access to PydanticAI conversation history.

    This class provides read-only access to PydanticAI conversation data and
    optional export functionality for analytics. It eliminates complex bidirectional
    conversion to reduce data duplication and synchronization issues.

    Key features:
    - Read-only access to PydanticAI conversation history
    - Conversation summary generation
    - Optional export for analytics/debugging
    """

    def __init__(self, agent: "Agent"):
        """Initialize the conversation history adapter.

        Args:
            agent: The PydanticAI agent to integrate with.

        """

        self.agent = agent
        logger.debug("Initialized ConversationHistoryAdapter")

    # NOTE: thought_to_messages() removed - complex bidirectional conversion eliminated
    # PydanticAI conversation history is the primary memory source

    # NOTE: messages_to_thought() removed - complex bidirectional conversion eliminated

    def get_conversation_history(self) -> List["ModelMessage"]:
        """Get the current conversation history from the agent.

        Returns:
            List of messages in the agent's conversation history.
        """
        # Note: This assumes the agent has a way to access conversation history
        # The exact implementation depends on PydanticAI's API
        if hasattr(self.agent, "_conversation_history"):
            return self.agent._conversation_history  # type: ignore
        elif hasattr(self.agent, "conversation"):
            return self.agent.conversation  # type: ignore
        else:
            logger.warning("Agent does not expose conversation history")
            return []

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation.

        Returns:
            Dictionary containing conversation statistics and summary.
        """
        history = self.get_conversation_history()

        summary = {
            "total_messages": len(history),
            "request_count": sum(1 for msg in history if msg.kind == "request"),
            "response_count": sum(1 for msg in history if msg.kind == "response"),
            "conversation_length": sum(len(self._extract_message_content(msg)) for msg in history),
        }

        logger.debug(f"Generated conversation summary: {summary}")
        return summary

    def export_for_analytics(self, storage: Optional[Any] = None) -> Dict[str, Any]:
        """Export conversation data for analytics/debugging purposes.

        This method provides optional export functionality for analytics,
        maintaining clear separation between PydanticAI memory and Sifaka storage.

        Args:
            storage: Optional storage backend to save analytics data.

        Returns:
            Dictionary containing conversation data for analytics.
        """
        history = self.get_conversation_history()

        # Extract conversation data for analytics
        analytics_data = {
            "conversation_summary": self.get_conversation_summary(),
            "messages": [],
            "export_timestamp": str(datetime.now()),
        }

        # Extract message data for analytics
        for msg in history:
            msg_data = {
                "kind": getattr(msg, "kind", None),
                "content": self._extract_message_content(msg),
                "timestamp": getattr(msg, "timestamp", None),
            }
            analytics_data["messages"].append(msg_data)

        # Optionally save to storage
        if storage and hasattr(storage, "store"):
            try:
                storage.store(
                    f"conversation_analytics_{datetime.now().isoformat()}", analytics_data
                )
                logger.debug("Exported conversation data to analytics storage")
            except Exception as e:
                logger.warning(f"Failed to export to analytics storage: {e}")

        logger.debug(
            f"Exported conversation analytics data with {len(analytics_data['messages'])} messages"
        )
        return analytics_data

    def clear_conversation_history(self):
        """Clear the agent's conversation history.

        Note: This may not be supported by all PydanticAI agent implementations.
        """
        if hasattr(self.agent, "clear_conversation"):
            self.agent.clear_conversation()  # type: ignore
            logger.debug("Cleared conversation history")
        else:
            logger.warning("Agent does not support clearing conversation history")

    def _extract_message_content(self, message: "ModelMessage") -> str:
        """Extract text content from a PydanticAI message.

        Args:
            message: The PydanticAI message to extract content from.

        Returns:
            Extracted text content.
        """
        content_parts = []
        for part in message.parts:
            if part.part_kind == "text":
                content_parts.append(part.content)

        return " ".join(content_parts)

    # NOTE: Validation and critic formatting methods removed
    # These were used for complex bidirectional conversion which is now eliminated
