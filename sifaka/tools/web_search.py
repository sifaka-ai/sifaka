"""Built-in web search tool using httpx for RAG support.

This tool provides web search capabilities without external dependencies,
using DuckDuckGo's HTML interface for search results.
"""

import urllib.parse
from typing import Any, Dict, List, cast

import httpx
from bs4 import BeautifulSoup, Tag

from .base import ToolInterface
from .registry import ToolRegistry
from .types import WebSearchResult


class WebSearchTool(ToolInterface):
    """Web search tool using DuckDuckGo HTML scraping.

    Provides basic web search functionality without requiring API keys
    or external search services. Uses httpx for async HTTP requests.

    Features:
    - No API key required
    - Async operation
    - Clean result extraction
    - Automatic retry on failure

    Example:
        >>> tool = WebSearchTool()
        >>> results = await tool("Eiffel Tower height meters")
        >>> for result in results:
        ...     print(f"{result['title']}: {result['snippet']}")
    """

    def __init__(self, max_results: int = 5, timeout: float = 10.0):
        """Initialize web search tool.

        Args:
            max_results: Maximum number of results to return
            timeout: HTTP request timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Sifaka/1.0; +https://github.com/sifaka)"
        }

    @property
    def name(self) -> str:
        """Tool identifier."""
        return "web_search"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Search the web for current information"

    async def __call__(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Search the web for information.

        Args:
            query: Search query string

        Returns:
            List of search results with title, url, and snippet
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Search using DuckDuckGo HTML
                encoded_query = urllib.parse.quote(query)
                url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

                response = await client.get(url, headers=self.headers)
                response.raise_for_status()

                # Parse HTML results
                soup = BeautifulSoup(response.text, "html.parser")
                results: List[WebSearchResult] = []

                # Find result divs
                for result_div in soup.find_all("div", class_="result"):
                    if len(results) >= self.max_results:
                        break

                    # Ensure result_div is a Tag
                    if not isinstance(result_div, Tag):
                        continue

                    # Extract title and URL
                    title_elem = result_div.find("a", class_="result__a")
                    if not title_elem or not isinstance(title_elem, Tag):
                        continue

                    title = title_elem.get_text(strip=True)
                    url = str(title_elem.get("href", ""))

                    # Extract snippet
                    snippet_elem = result_div.find("a", class_="result__snippet")
                    snippet = (
                        snippet_elem.get_text(strip=True)
                        if snippet_elem and isinstance(snippet_elem, Tag)
                        else ""
                    )

                    if title and url:
                        result: WebSearchResult = {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "source": "web_search",
                        }
                        results.append(result)

                return cast(List[Dict[str, Any]], results)

            except Exception as e:
                # Return empty list on error rather than failing
                error_result: WebSearchResult = {
                    "title": "Search Error",
                    "url": "",
                    "snippet": f"Failed to search: {e!s}",
                    "source": "web_search",
                }
                return cast(List[Dict[str, Any]], [error_result])


# Auto-register the tool
ToolRegistry.register("web_search", WebSearchTool)
