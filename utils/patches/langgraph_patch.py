"""
Patch for LangGraph compatibility issues.

This patch addresses import and compatibility issues with LangGraph and LangChain.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply all patches for LangGraph."""
    success = patch_langgraph_module()
    if success:
        logger.info("Applied LangGraph patches successfully")
    return success

def patch_langgraph_module():
    """Patch the langgraph.py module to fix the StateGraph import."""
    try:
        # Path to the langgraph.py file
        langgraph_path = os.path.join("sifaka", "integrations", "langgraph.py")

        # Check if the file exists
        if not os.path.exists(langgraph_path):
            logger.warning(f"LangGraph integration file not found at {langgraph_path}")
            return False

        # Read the file content
        with open(langgraph_path, "r") as f:
            content = f.read()

        # Check if patch is needed
        if "from langchain.graphs import StateGraph" not in content:
            logger.info("LangGraph patch not needed, skipping")
            return True

        # Replace the import statement
        new_content = re.sub(
            r"from langchain\.graphs import StateGraph",
            "from langgraph.graph import StateGraph",
            content,
        )

        # Write the updated content back to the file
        with open(langgraph_path, "w") as f:
            f.write(new_content)

        logger.info(f"Patched {langgraph_path} to use correct StateGraph import")
        return True
    except Exception as e:
        logger.error(f"Failed to patch LangGraph module: {e}")
        return False