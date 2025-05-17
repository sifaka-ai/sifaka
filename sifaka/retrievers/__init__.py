"""
Retrievers for Sifaka.

This package provides retrievers for retrieving relevant documents for a query.
Retrievers are used by components like Self-RAG to augment generation with retrieved information.
"""

from sifaka.retrievers.base import Retriever

# Import retrievers conditionally to avoid hard dependencies
try:
    from sifaka.retrievers.elasticsearch_retriever import ElasticsearchRetriever
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

try:
    from sifaka.retrievers.milvus_retriever import MilvusRetriever
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

__all__ = ["Retriever"]

if ELASTICSEARCH_AVAILABLE:
    __all__.append("ElasticsearchRetriever")

if MILVUS_AVAILABLE:
    __all__.append("MilvusRetriever")
