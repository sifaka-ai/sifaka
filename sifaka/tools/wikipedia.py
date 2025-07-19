"""Wikipedia API tool for fact-checking and knowledge retrieval.

Provides direct access to Wikipedia content without external dependencies.
"""

import urllib.parse
from typing import Any, Dict, List, Union, cast

import httpx

from .base import ToolInterface
from .registry import ToolRegistry
from .types import WikipediaResult


class WikipediaTool(ToolInterface):
    """Wikipedia search and content retrieval tool.

    Uses Wikipedia's public API for searching and retrieving article content.
    No authentication required.

    Features:
    - Search for articles by query
    - Retrieve article summaries
    - Get full article content
    - Extract specific sections

    Example:
        >>> tool = WikipediaTool()
        >>> results = await tool("Eiffel Tower")
        >>> print(results[0]['summary'])
    """

    def __init__(self, max_results: int = 3, timeout: float = 10.0):
        """Initialize Wikipedia tool.

        Args:
            max_results: Maximum number of articles to return
            timeout: HTTP request timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://en.wikipedia.org/api/rest_v1"
        self.search_url = "https://en.wikipedia.org/w/api.php"

    @property
    def name(self) -> str:
        """Tool identifier."""
        return "wikipedia"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Search and retrieve information from Wikipedia"

    async def __call__(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Search Wikipedia and return article information.

        Args:
            query: Search query or article title

        Returns:
            List of article information with title, url, and summary
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # First, search for relevant articles
                search_params: Dict[str, Union[str, int]] = {
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": self.max_results,
                }

                search_response = await client.get(
                    self.search_url, params=search_params
                )
                search_response.raise_for_status()
                search_data = search_response.json()

                results: List[WikipediaResult] = []

                # Get summaries for each search result
                for item in search_data.get("query", {}).get("search", []):
                    title = item["title"]

                    # Get article summary
                    summary_url = (
                        f"{self.base_url}/page/summary/{urllib.parse.quote(title)}"
                    )

                    try:
                        summary_response = await client.get(summary_url)
                        summary_response.raise_for_status()
                        summary_data = summary_response.json()

                        result: WikipediaResult = {
                            "title": summary_data.get("title", title),
                            "url": summary_data.get("content_urls", {})
                            .get("desktop", {})
                            .get("page", ""),
                            "summary": summary_data.get("extract", ""),
                            "snippet": item.get("snippet", "")
                            .replace('<span class="searchmatch">', "")
                            .replace("</span>", ""),
                            "source": "wikipedia",
                        }
                        results.append(result)
                    except Exception:
                        # If summary fails, still include basic search result
                        fallback_result: WikipediaResult = {
                            "title": title,
                            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                            "summary": "",
                            "snippet": item.get("snippet", "")
                            .replace('<span class="searchmatch">', "")
                            .replace("</span>", ""),
                            "source": "wikipedia",
                        }
                        results.append(fallback_result)

                return cast(List[Dict[str, Any]], results)

            except Exception as e:
                # Return error information
                error_result: WikipediaResult = {
                    "title": "Wikipedia Error",
                    "url": "",
                    "summary": f"Failed to search Wikipedia: {e!s}",
                    "snippet": "",
                    "source": "wikipedia",
                }
                return cast(List[Dict[str, Any]], [error_result])


# Auto-register the tool
ToolRegistry.register("wikipedia", WikipediaTool)
