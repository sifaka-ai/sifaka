"""arXiv API tool for retrieving research papers and scientific literature.

Provides access to arXiv's repository of scientific papers across multiple disciplines.
"""

from typing import Any, Dict, List, Union, cast

import defusedxml.ElementTree as ET
import httpx

from .base import ToolInterface
from .registry import ToolRegistry
from .types import ArxivResult


class ArxivTool(ToolInterface):
    """arXiv search and paper retrieval tool.

    Uses arXiv's public API to search and retrieve research papers.
    No authentication required.

    Features:
    - Search papers by query, author, or category
    - Retrieve paper abstracts and metadata
    - Get paper details including authors, categories, and publication date
    - Direct links to PDFs

    Example:
        >>> tool = ArxivTool()
        >>> papers = await tool("transformer attention mechanism")
        >>> for paper in papers:
        ...     print(f"{paper['title']} by {paper['authors'][0]}")
    """

    def __init__(self, max_results: int = 5, timeout: float = 10.0):
        """Initialize arXiv tool.

        Args:
            max_results: Maximum number of papers to return
            timeout: HTTP request timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "http://export.arxiv.org/api/query"

    @property
    def name(self) -> str:
        """Tool identifier."""
        return "arxiv"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Search and retrieve research papers from arXiv"

    async def __call__(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Search arXiv and return paper information.

        Args:
            query: Search query (can include author:, title:, cat: prefixes)

        Returns:
            List of paper information with title, abstract, authors, etc.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Build search query
                params: Dict[str, Union[str, int]] = {
                    "search_query": query,
                    "start": 0,
                    "max_results": self.max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                }

                response = await client.get(self.base_url, params=params)
                response.raise_for_status()

                # Parse XML response
                root = ET.fromstring(response.text)

                # Define XML namespaces
                namespaces = {
                    "atom": "http://www.w3.org/2005/Atom",
                    "arxiv": "http://arxiv.org/schemas/atom",
                }

                results: List[ArxivResult] = []

                # Extract entries
                for entry in root.findall("atom:entry", namespaces):
                    # Extract basic information
                    title = entry.find("atom:title", namespaces)
                    title_text = (
                        title.text.strip().replace("\n", " ")
                        if title is not None and title.text is not None
                        else ""
                    )

                    summary = entry.find("atom:summary", namespaces)
                    abstract = (
                        summary.text.strip()
                        if summary is not None and summary.text is not None
                        else ""
                    )

                    # Extract authors
                    authors = []
                    for author in entry.findall("atom:author", namespaces):
                        name = author.find("atom:name", namespaces)
                        if name is not None and name.text is not None:
                            authors.append(name.text)

                    # Extract links
                    pdf_url = ""
                    abs_url = ""
                    for link in entry.findall("atom:link", namespaces):
                        if link.get("title") == "pdf":
                            pdf_url = link.get("href", "")
                        elif link.get("rel") == "alternate":
                            abs_url = link.get("href", "")

                    # Extract ID and dates
                    id_elem = entry.find("atom:id", namespaces)
                    arxiv_id = (
                        id_elem.text.split("/")[-1]
                        if id_elem is not None and id_elem.text is not None
                        else ""
                    )

                    published = entry.find("atom:published", namespaces)
                    pub_date = (
                        published.text
                        if published is not None and published.text is not None
                        else ""
                    )

                    # Extract categories
                    categories = []
                    for category in entry.findall("arxiv:primary_category", namespaces):
                        cat_term = category.get("term", "")
                        if cat_term:
                            categories.append(cat_term)

                    result: ArxivResult = {
                        "title": title_text,
                        "abstract": (
                            abstract[:500] + "..." if len(abstract) > 500 else abstract
                        ),
                        "authors": authors,
                        "arxiv_id": arxiv_id,
                        "url": abs_url,
                        "pdf_url": pdf_url,
                        "published": pub_date,
                        "categories": categories,
                        "source": "arxiv",
                    }
                    results.append(result)

                return cast(List[Dict[str, Any]], results)

            except Exception as e:
                # Return error information
                error_result: ArxivResult = {
                    "title": "arXiv Error",
                    "abstract": f"Failed to search arXiv: {e!s}",
                    "authors": [],
                    "arxiv_id": "",
                    "url": "",
                    "pdf_url": "",
                    "published": "",
                    "categories": [],
                    "source": "arxiv",
                }
                return cast(List[Dict[str, Any]], [error_result])


# Auto-register the tool
ToolRegistry.register("arxiv", ArxivTool)
