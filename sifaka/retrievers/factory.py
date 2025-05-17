"""
Factory functions for creating retrievers.

This module provides factory functions for creating retrievers.
"""

from typing import List, Dict, Any, Optional, Callable, Union
import logging

from sifaka.retrievers.base import Retriever
from sifaka.errors import RetrieverError

logger = logging.getLogger(__name__)


def create_elasticsearch_retriever(
    es_host: str,
    es_index: str,
    embedding_model: Callable[[str], List[float]],
    hybrid_search: bool = True,
    top_k: int = 5,
    text_field: str = "text",
    embedding_field: str = "embedding",
    **kwargs: Any,
) -> Retriever:
    """Create an Elasticsearch retriever.

    Args:
        es_host: The Elasticsearch host URL.
        es_index: The Elasticsearch index to search.
        embedding_model: Model or function to generate embeddings for queries.
        hybrid_search: Whether to use hybrid search (combining keyword and semantic).
        top_k: Number of documents to retrieve.
        text_field: The field containing document text.
        embedding_field: The field containing document embeddings.
        **kwargs: Additional arguments to pass to the Elasticsearch client.

    Returns:
        An Elasticsearch retriever.

    Raises:
        RetrieverError: If Elasticsearch is not available.
    """
    try:
        from sifaka.retrievers.elasticsearch_retriever import ElasticsearchRetriever
        return ElasticsearchRetriever(
            es_host=es_host,
            es_index=es_index,
            embedding_model=embedding_model,
            hybrid_search=hybrid_search,
            top_k=top_k,
            text_field=text_field,
            embedding_field=embedding_field,
            **kwargs,
        )
    except ImportError:
        raise RetrieverError(
            "Elasticsearch is not available. Please install it with: pip install elasticsearch"
        )


def create_milvus_retriever(
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    embedding_model: Callable[[str], List[float]],
    text_field: str = "text",
    embedding_field: str = "embedding",
    top_k: int = 5,
    metric_type: str = "COSINE",
    **kwargs: Any,
) -> Retriever:
    """Create a Milvus retriever.

    Args:
        milvus_host: The Milvus host.
        milvus_port: The Milvus port.
        collection_name: The Milvus collection to search.
        embedding_model: Model or function to generate embeddings for queries.
        text_field: The field containing document text.
        embedding_field: The field containing document embeddings.
        top_k: Number of documents to retrieve.
        metric_type: The metric type to use for similarity search.
        **kwargs: Additional arguments to pass to the Milvus connection.

    Returns:
        A Milvus retriever.

    Raises:
        RetrieverError: If Milvus is not available.
    """
    try:
        from sifaka.retrievers.milvus_retriever import MilvusRetriever
        return MilvusRetriever(
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name=collection_name,
            embedding_model=embedding_model,
            text_field=text_field,
            embedding_field=embedding_field,
            top_k=top_k,
            metric_type=metric_type,
            **kwargs,
        )
    except ImportError:
        raise RetrieverError(
            "Milvus is not available. Please install it with: pip install pymilvus"
        )
