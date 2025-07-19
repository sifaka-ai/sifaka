"""Type definitions for tool results."""

import sys
from typing import List

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class ArxivResult(TypedDict):
    """Result from ArXiv search."""

    title: str
    abstract: str
    authors: List[str]
    arxiv_id: str
    url: str
    pdf_url: str
    published: str
    categories: List[str]
    source: str


class WikipediaResult(TypedDict):
    """Result from Wikipedia search."""

    title: str
    summary: str
    url: str
    snippet: str
    source: str


class WebSearchResult(TypedDict):
    """Result from web search."""

    title: str
    snippet: str
    url: str
    source: str
