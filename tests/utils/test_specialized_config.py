"""
Tests for specialized configuration classes.
"""

import pytest
from pydantic import ValidationError

# Import specialized model configurations
from sifaka.utils.config.models import (
    ModelConfig,
    OpenAIConfig,
    AnthropicConfig,
    GeminiConfig,
)

# Import specialized chain configurations
from sifaka.utils.config.chain import (
    ChainConfig,
    EngineConfig,
    ValidatorConfig,
    ImproverConfig,
)

# Import specialized retrieval configurations
from sifaka.utils.config.retrieval import (
    RetrieverConfig,
    RankingConfig,
    IndexConfig,
    QueryProcessingConfig,
)


# Tests for specialized model configurations
def test_openai_config():
    """Test that OpenAIConfig works correctly."""
    # Test with valid parameters
    config = OpenAIConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=100,
        params={
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        },
    )

    assert config.model == "gpt-4"
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.params["top_p"] == 0.9
    assert config.params["frequency_penalty"] == 0.5
    assert config.params["presence_penalty"] == 0.5

    # Test with_options method
    updated_config = config.with_options(temperature=0.8)
    assert updated_config.temperature == 0.8
    assert updated_config.model == "gpt-4"  # Unchanged
    assert updated_config.params == config.params  # Unchanged

    # Test with_params method
    updated_config = config.with_params(top_p=0.95)
    assert updated_config.params["top_p"] == 0.95
    assert updated_config.params["frequency_penalty"] == 0.5  # Unchanged
    assert updated_config.temperature == 0.7  # Unchanged


def test_anthropic_config():
    """Test that AnthropicConfig works correctly."""
    # Test with valid parameters
    config = AnthropicConfig(
        model="claude-3-opus",
        temperature=0.7,
        max_tokens=100,
        params={
            "top_k": 50,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman:", "\n\nAssistant:"],
        },
    )

    assert config.model == "claude-3-opus"
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.params["top_k"] == 50
    assert config.params["top_p"] == 0.9
    assert config.params["stop_sequences"] == ["\n\nHuman:", "\n\nAssistant:"]

    # Test with_options method
    updated_config = config.with_options(temperature=0.8)
    assert updated_config.temperature == 0.8
    assert updated_config.model == "claude-3-opus"  # Unchanged
    assert updated_config.params == config.params  # Unchanged

    # Test with_params method
    updated_config = config.with_params(top_p=0.95)
    assert updated_config.params["top_p"] == 0.95
    assert updated_config.params["top_k"] == 50  # Unchanged
    assert updated_config.temperature == 0.7  # Unchanged


def test_gemini_config():
    """Test that GeminiConfig works correctly."""
    # Test with valid parameters
    config = GeminiConfig(
        model="gemini-pro",
        temperature=0.7,
        max_tokens=100,
        params={
            "top_k": 40,
            "top_p": 0.95,
            "candidate_count": 1,
        },
    )

    assert config.model == "gemini-pro"
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.params["top_k"] == 40
    assert config.params["top_p"] == 0.95
    assert config.params["candidate_count"] == 1

    # Test with_options method
    updated_config = config.with_options(temperature=0.8)
    assert updated_config.temperature == 0.8
    assert updated_config.model == "gemini-pro"  # Unchanged
    assert updated_config.params == config.params  # Unchanged

    # Test with_params method
    updated_config = config.with_params(candidate_count=3)
    assert updated_config.params["candidate_count"] == 3
    assert updated_config.params["top_k"] == 40  # Unchanged
    assert updated_config.temperature == 0.7  # Unchanged


# Tests for specialized chain configurations
def test_engine_config():
    """Test that EngineConfig works correctly."""
    # Test with valid parameters
    config = EngineConfig(
        max_attempts=3,
        timeout_seconds=30,
        fail_fast=True,
        params={
            "parallel_execution": True,
        },
    )

    assert config.max_attempts == 3
    assert config.timeout_seconds == 30
    assert config.fail_fast is True
    assert config.params["parallel_execution"] is True

    # Test with_options method
    updated_config = config.with_options(max_attempts=5)
    assert updated_config.max_attempts == 5
    assert updated_config.timeout_seconds == 30  # Unchanged
    assert updated_config.params == config.params  # Unchanged

    # Test with_params method
    updated_config = config.with_params(parallel_execution=False)
    assert updated_config.params["parallel_execution"] is False
    assert updated_config.max_attempts == 3  # Unchanged


def test_validator_config():
    """Test that ValidatorConfig works correctly."""
    # Test with valid parameters
    config = ValidatorConfig(
        prioritize_by_cost=True,
        parallel_validation=False,
        params={
            "strict_mode": True,
            "validation_threshold": 0.8,
        },
    )

    assert config.prioritize_by_cost is True
    assert config.parallel_validation is False
    assert config.params["strict_mode"] is True
    assert config.params["validation_threshold"] == 0.8

    # Test with_options method
    updated_config = config.with_options(parallel_validation=True)
    assert updated_config.parallel_validation is True
    assert updated_config.prioritize_by_cost is True  # Unchanged
    assert updated_config.params == config.params  # Unchanged

    # Test with_params method
    updated_config = config.with_params(validation_threshold=0.9)
    assert updated_config.params["validation_threshold"] == 0.9
    assert updated_config.params["strict_mode"] is True  # Unchanged
    assert updated_config.prioritize_by_cost is True  # Unchanged


def test_improver_config():
    """Test that ImproverConfig works correctly."""
    # Test with valid parameters
    config = ImproverConfig(
        max_attempts=3,
        timeout_seconds=30,
        params={
            "strategy": "incremental",
            "feedback_mode": "detailed",
        },
    )

    assert config.max_attempts == 3
    assert config.timeout_seconds == 30
    assert config.params["strategy"] == "incremental"
    assert config.params["feedback_mode"] == "detailed"

    # Test with_options method
    updated_config = config.with_options(max_attempts=5)
    assert updated_config.max_attempts == 5
    assert updated_config.timeout_seconds == 30  # Unchanged
    assert updated_config.params == config.params  # Unchanged

    # Test with_params method
    updated_config = config.with_params(strategy="complete")
    assert updated_config.params["strategy"] == "complete"
    assert updated_config.params["feedback_mode"] == "detailed"  # Unchanged
    assert updated_config.max_attempts == 3  # Unchanged


# Tests for specialized retrieval configurations
def test_ranking_config():
    """Test that RankingConfig works correctly."""
    # Test with valid parameters
    config = RankingConfig(
        algorithm="bm25",
        weights={
            "title": 2.0,
            "content": 1.0,
        },
        params={
            "k1": 1.2,
            "b": 0.75,
        },
    )

    assert config.algorithm == "bm25"
    assert config.weights["title"] == 2.0
    assert config.weights["content"] == 1.0
    assert config.params["k1"] == 1.2
    assert config.params["b"] == 0.75

    # Test with_options method
    updated_config = config.with_options(algorithm="tfidf")
    assert updated_config.algorithm == "tfidf"
    assert updated_config.weights == config.weights  # Unchanged
    assert updated_config.params == config.params  # Unchanged

    # Test with_params method
    updated_config = config.with_params(k1=1.5)
    assert updated_config.params["k1"] == 1.5
    assert updated_config.params["b"] == 0.75  # Unchanged
    assert updated_config.algorithm == "bm25"  # Unchanged


def test_index_config():
    """Test that IndexConfig works correctly."""
    # Test with valid parameters
    config = IndexConfig(
        index_type="vector",
        chunk_size=500,
        overlap=50,
        params={
            "dimensions": 768,
            "metric": "cosine",
        },
    )

    assert config.index_type == "vector"
    assert config.chunk_size == 500
    assert config.overlap == 50
    assert config.params["dimensions"] == 768
    assert config.params["metric"] == "cosine"

    # Test with_options method
    updated_config = config.with_options(chunk_size=1000)
    assert updated_config.chunk_size == 1000
    assert updated_config.index_type == "vector"  # Unchanged
    assert updated_config.params == config.params  # Unchanged

    # Test with_params method
    updated_config = config.with_params(dimensions=1024)
    assert updated_config.params["dimensions"] == 1024
    assert updated_config.params["metric"] == "cosine"  # Unchanged
    assert updated_config.chunk_size == 500  # Unchanged


def test_query_processing_config():
    """Test that QueryProcessingConfig works correctly."""
    # Test with valid parameters
    config = QueryProcessingConfig(
        preprocessing_steps=["lowercase", "remove_stopwords"],
        expansion_method="synonyms",
        params={
            "max_synonyms": 3,
            "language": "en",
        },
    )

    assert config.preprocessing_steps == ["lowercase", "remove_stopwords"]
    assert config.expansion_method == "synonyms"
    assert config.params["max_synonyms"] == 3
    assert config.params["language"] == "en"

    # Test with_options method
    updated_config = config.with_options(expansion_method="embeddings")
    assert updated_config.expansion_method == "embeddings"
    assert updated_config.preprocessing_steps == config.preprocessing_steps  # Unchanged
    assert updated_config.params == config.params  # Unchanged

    # Test with_params method
    updated_config = config.with_params(max_synonyms=5)
    assert updated_config.params["max_synonyms"] == 5
    assert updated_config.params["language"] == "en"  # Unchanged
    assert updated_config.expansion_method == "synonyms"  # Unchanged
