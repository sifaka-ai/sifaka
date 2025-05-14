"""
Retrieval Configuration Module

This module provides configuration classes and standardization functions for retrieval components.

## Overview
The retrieval configuration module defines configuration classes for retrieval components
in the Sifaka framework. It provides a consistent approach to configuring retrievers,
ranking strategies, index management, and query processing with standardized parameter
handling, validation, and serialization.

## Components
- **RetrieverConfig**: Configuration for retrievers
- **RankingConfig**: Configuration for ranking strategies
- **IndexConfig**: Configuration for index management
- **QueryProcessingConfig**: Configuration for query processing
- **standardize_retriever_config**: Standardization function for retriever configurations

## Usage Examples
```python
from sifaka.utils.config.retrieval import (
    RetrieverConfig, RankingConfig, standardize_retriever_config
)

# Create a retriever configuration
config = RetrieverConfig(
    name="my_retriever",
    description="A custom retriever",
    top_k=10,
    min_score=0.7
)

# Create a ranking configuration
ranking_config = RankingConfig(
    name="my_ranking",
    description="A custom ranking strategy",
    algorithm="bm25",
    params={
        "k1": 1.2,
        "b": 0.75
    }
)

# Use standardization function
config = standardize_retriever_config(
    top_k=10,
    min_score=0.7,
    params={
        "algorithm": "bm25",
        "k1": 1.2,
        "b": 0.75
    }
)
```

## Error Handling
The configuration utilities use Pydantic for validation, which ensures that
configuration values are valid and properly typed. If invalid configuration
is provided, Pydantic will raise validation errors with detailed information
about the validation failure.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast
from pydantic import Field
from .base import BaseConfig

T = TypeVar("T", bound="RetrieverConfig")


class RetrieverConfig(BaseConfig):
    """
    Configuration for retrievers.

    This class provides a consistent way to configure retrievers across the Sifaka framework.
    It handles common configuration options like top_k and min_score, while
    allowing retriever-specific options through the params dictionary.

    ## Architecture
    RetrieverConfig extends BaseConfig with retriever-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during retriever initialization and
    remain immutable throughout the retriever's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.retrieval import RetrieverConfig

    # Create a retriever configuration
    config = RetrieverConfig(
        name="my_retriever",
        description="A custom retriever",
        top_k=10,
        max_results=10,
        min_score=0.7,
        cache_size=100,
        trace_enabled=True,
        params={
            "algorithm": "bm25",
            "k1": 1.2,
            "b": 0.75
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Top K: {config.top_k}")
    print(f"Max Results: {config.max_results}")
    print(f"Algorithm: {config.params.get('algorithm')}")

    # Create a new configuration with updated options
    updated_config = config.with_options(top_k=20)

    # Create a new configuration with updated params
    updated_config = config.with_params(algorithm="tfidf")
    ```

    Attributes:
        top_k: Number of results to retrieve
        max_results: Maximum number of results to return
        min_score: Minimum score threshold
        score_threshold: Threshold for score-based filtering
        cache_size: Size of the result cache
        trace_enabled: Whether to enable tracing
        ranking: Configuration for ranking strategy
    """

    top_k: int = Field(default=10, ge=1, description="Number of results to retrieve")
    max_results: int = Field(default=5, ge=1, description="Maximum number of results to return")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score threshold")
    score_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Threshold for score-based filtering"
    )
    cache_size: int = Field(default=100, ge=0, description="Size of the result cache")
    trace_enabled: bool = Field(default=False, description="Whether to enable tracing")
    ranking: Optional[Dict[str, Any]] = Field(
        default=None, description="Configuration for ranking strategy"
    )


class RankingConfig(BaseConfig):
    """
    Configuration for ranking strategies.

    This class provides a consistent way to configure ranking strategies across the Sifaka framework.
    It handles common configuration options like algorithm and weights, while
    allowing ranking-specific options through the params dictionary.

    ## Architecture
    RankingConfig extends BaseConfig with ranking-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during ranking strategy initialization and
    remain immutable throughout the strategy's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.retrieval import RankingConfig

    # Create a ranking configuration
    config = RankingConfig(
        name="my_ranking",
        description="A custom ranking strategy",
        algorithm="bm25",
        weights={
            "title": 2.0,
            "content": 1.0
        },
        params={
            "k1": 1.2,
            "b": 0.75
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Title weight: {config.weights.get('title')}")

    # Create a new configuration with updated options
    updated_config = config.with_options(algorithm="tfidf")

    # Create a new configuration with updated params
    updated_config = config.with_params(k1=1.5, b=0.8)
    ```

    Attributes:
        algorithm: Ranking algorithm to use
        weights: Dictionary of field weights
    """

    algorithm: str = Field(default="", description="Ranking algorithm to use")
    weights: Dict[str, float] = Field(
        default_factory=dict, description="Dictionary of field weights"
    )


class IndexConfig(BaseConfig):
    """
    Configuration for index management.

    This class provides a consistent way to configure index management across the Sifaka framework.
    It handles common configuration options like index_type and chunk_size, while
    allowing index-specific options through the params dictionary.

    ## Architecture
    IndexConfig extends BaseConfig with index-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during index initialization and
    remain immutable throughout the index's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.retrieval import IndexConfig

    # Create an index configuration
    config = IndexConfig(
        name="my_index",
        description="A custom index",
        index_type="vector",
        chunk_size=1000,
        overlap=200,
        params={
            "dimensions": 1536,
            "metric": "cosine"
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Index type: {config.index_type}")
    print(f"Chunk size: {config.chunk_size}")

    # Create a new configuration with updated options
    updated_config = config.with_options(chunk_size=500)

    # Create a new configuration with updated params
    updated_config = config.with_params(dimensions=768)
    ```

    Attributes:
        index_type: Type of index to use
        chunk_size: Size of chunks for indexing
        overlap: Overlap between chunks
    """

    index_type: str = Field(default="", description="Type of index to use")
    chunk_size: int = Field(default=1000, ge=1, description="Size of chunks for indexing")
    overlap: int = Field(default=0, ge=0, description="Overlap between chunks")


class QueryProcessingConfig(BaseConfig):
    """
    Configuration for query processing.

    This class provides a consistent way to configure query processing across the Sifaka framework.
    It handles common configuration options like expansion_enabled and rewriting_enabled, while
    allowing query-specific options through the params dictionary.

    ## Architecture
    QueryProcessingConfig extends BaseConfig with query-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during query processor initialization and
    remain immutable throughout the processor's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.retrieval import QueryProcessingConfig

    # Create a query processing configuration
    config = QueryProcessingConfig(
        name="my_query_processor",
        description="A custom query processor",
        expansion_enabled=True,
        rewriting_enabled=True,
        preprocessing_steps=["lowercase", "remove_stopwords"],
        expansion_method="synonym",
        params={
            "expansion_model": "gpt-4",
            "expansion_temperature": 0.7
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Expansion enabled: {config.expansion_enabled}")
    print(f"Rewriting enabled: {config.rewriting_enabled}")
    print(f"Preprocessing steps: {config.preprocessing_steps}")
    print(f"Expansion method: {config.expansion_method}")

    # Create a new configuration with updated options
    updated_config = config.with_options(expansion_enabled=False)

    # Create a new configuration with updated params
    updated_config = config.with_params(expansion_model="gpt-3.5-turbo")
    ```

    Attributes:
        expansion_enabled: Whether to enable query expansion
        rewriting_enabled: Whether to enable query rewriting
        preprocessing_steps: List of preprocessing steps to apply to queries
        expansion_method: Method to use for query expansion
    """

    expansion_enabled: bool = Field(default=False, description="Whether to enable query expansion")
    rewriting_enabled: bool = Field(default=False, description="Whether to enable query rewriting")
    preprocessing_steps: List[str] = Field(
        default_factory=list, description="List of preprocessing steps to apply to queries"
    )
    expansion_method: Optional[str] = Field(
        default=None, description="Method to use for query expansion"
    )


def standardize_retriever_config(
    config: Optional[Union[Dict[str, Any], RetrieverConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = None,  # type: ignore
    **kwargs: Any,
) -> T:
    """
    Standardize retriever configuration.

    This utility function ensures that retriever configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized RetrieverConfig object or a subclass.

    Args:
        config: Optional configuration (either a dict or RetrieverConfig)
        params: Optional params dictionary to merge with config
        config_class: The config class to use (default: RetrieverConfig)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized RetrieverConfig object or subclass

    Examples:
        from sifaka.utils.config.retrieval import standardize_retriever_config, RankingConfig

        # Create from parameters
        config = standardize_retriever_config(
            top_k=10,
            min_score=0.7,
            params={
                "algorithm": "bm25",
                "k1": 1.2,
                "b": 0.75
            }
        )

        # Create from existing config
        from sifaka.utils.config.retrieval import RetrieverConfig
        existing = RetrieverConfig(top_k=10)
        updated = standardize_retriever_config(
            config=existing,
            params={
                "algorithm": "bm25",
                "k1": 1.2,
                "b": 0.75
            }
        )

        # Create from dictionary
        dict_config = {
            "top_k": 10,
            "min_score": 0.7,
            "params": {
                "algorithm": "bm25",
                "k1": 1.2,
                "b": 0.75
            }
        }
        config = standardize_retriever_config(config=dict_config)

        # Create specialized config
        ranking_config = standardize_retriever_config(
            config_class=RankingConfig,
            algorithm="bm25",
            weights={
                "title": 2.0,
                "content": 1.0
            }
        )
    """
    if config_class is None:
        config_class = RetrieverConfig  # type: ignore
    final_params: Dict[str, Any] = {}
    if params:
        final_params.update(params)
    if isinstance(config, dict):
        dict_params = config.pop("params", {}) if config else {}
        final_params.update(dict_params)
        return cast(
            T, config_class(**{} if config is None else config, params=final_params, **kwargs)
        )
    elif isinstance(config, RetrieverConfig):
        final_params.update(config.params)
        config_dict = {**config.model_dump(), "params": final_params, **kwargs}
        return cast(T, config_class(**config_dict))
    else:
        return cast(T, config_class(params=final_params, **kwargs))
