"""
Compatibility patches for external libraries.

This module contains patches for external libraries that Sifaka depends on,
addressing compatibility issues between different versions.
"""

import logging

logger = logging.getLogger(__name__)

def apply_all_patches():
    """Apply all compatibility patches."""
    success = True

    # Apply LangChain patches
    try:
        from .langchain_patch import apply_patches as apply_langchain_patches
        langchain_result = apply_langchain_patches()
        if not langchain_result:
            logger.warning("LangChain patches were not fully applied")
            success = False
    except ImportError:
        logger.warning("LangChain not found, skipping patches")
    except Exception as e:
        logger.warning(f"Error applying LangChain patches: {e}")
        success = False

    # Apply LangGraph patches
    try:
        from .langgraph_patch import apply_patches as apply_langgraph_patches
        langgraph_result = apply_langgraph_patches()
        if not langgraph_result:
            logger.warning("LangGraph patches were not fully applied")
            success = False
    except ImportError:
        logger.warning("LangGraph not found, skipping patches")
    except Exception as e:
        logger.warning(f"Error applying LangGraph patches: {e}")
        success = False

    return success