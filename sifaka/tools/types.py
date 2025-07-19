"""Type definitions for tool results."""

from typing import List, TypedDict


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
