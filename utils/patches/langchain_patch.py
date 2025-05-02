"""
Monkey patch for LangChain to fix Pydantic v2 compatibility issues.

This patch addresses the "Model 'AIMessage' needs a discriminator field" error
by patching the ChatPromptValueConcrete class in langchain_core.
"""

import sys
import types
import logging
from typing import List, Union, Optional, get_origin, get_args
import inspect

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply all monkey patches to fix LangChain compatibility with Pydantic v2."""
    success = patch_langchain_core_prompt_values()
    if success:
        logger.info("Applied LangChain patches for Pydantic v2 compatibility")
    return success

def patch_langchain_core_prompt_values():
    """
    Patch langchain_core.prompt_values to fix discriminator issues with Pydantic v2.

    This addresses the error:
    "Model 'AIMessage' needs a discriminator field for key Discriminator()"
    """
    try:
        import langchain_core.prompt_values
        from pydantic import Field, Discriminator

        # Get the original class
        ChatPromptValueConcrete = langchain_core.prompt_values.ChatPromptValueConcrete

        # Create a patched class that's compatible with Pydantic v2
        class PatchedChatPromptValueConcrete(langchain_core.prompt_values.ChatPromptValue):
            """Patched version of ChatPromptValueConcrete with proper discriminator support."""

            # Use Field with discriminator for message types
            messages: List[langchain_core.prompt_values.BaseMessage] = Field(
                ...,  # Required field
                discriminator="type"  # Use the 'type' field as discriminator
            )

            def __init__(self, *args, **kwargs):
                """Initialize with the same signature as the original."""
                super().__init__(*args, **kwargs)

            def to_string(self) -> str:
                """Convert to string using the same method as the original."""
                return ChatPromptValueConcrete.to_string(self)

            def to_messages(self) -> List[langchain_core.prompt_values.BaseMessage]:
                """Convert to messages using the same method as the original."""
                return self.messages

        # Replace the original class with our patched one
        langchain_core.prompt_values.ChatPromptValueConcrete = PatchedChatPromptValueConcrete

        return True
    except ImportError:
        logger.warning("LangChain core not found, skipping patch")
        return False
    except Exception as e:
        logger.error(f"Failed to patch langchain_core.prompt_values: {e}")
        return False