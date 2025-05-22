"""
Retrievers for Sifaka.

This package provides retrievers for retrieving relevant documents for a query.
Retrievers are used by components like Self-RAG to augment generation with retrieved information.
The package also provides a generic retrieval augmenter that can be used by multiple critics.
"""

from sifaka.retrievers.base import Retriever
from sifaka.retrievers.factory import (
    create_elasticsearch_retriever,
    create_milvus_retriever,
    create_retrieval_augmenter,
)
from sifaka.retrievers.simple import SimpleRetriever

# Import retrievers conditionally to avoid hard dependencies
try:
    pass

    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

try:
    pass

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    pass

    RETRIEVAL_AUGMENTER_AVAILABLE = True
except ImportError:
    RETRIEVAL_AUGMENTER_AVAILABLE = False

__all__ = [
    "Retriever",
    "SimpleRetriever",
    "create_elasticsearch_retriever",
    "create_milvus_retriever",
    "create_retrieval_augmenter",
]

if ELASTICSEARCH_AVAILABLE:
    __all__.append("ElasticsearchRetriever")

if MILVUS_AVAILABLE:
    __all__.append("MilvusRetriever")

if RETRIEVAL_AUGMENTER_AVAILABLE:
    __all__.append("RetrievalAugmenter")
