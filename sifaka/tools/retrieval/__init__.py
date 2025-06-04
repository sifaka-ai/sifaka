"""External retrieval tools for Sifaka.

This module provides external retrieval tools that can be used with PydanticAI agents
to search and retrieve information from external sources like the web.

Note: For internal Sifaka state/thought retrieval, use the existing storage.tools module.
"""

from sifaka.tools.retrieval.web_search import create_web_search_tools

__all__ = [
    "create_web_search_tools",
]
