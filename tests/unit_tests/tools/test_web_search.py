"""Comprehensive unit tests for web search tools.

This module tests the web search tool implementations:
- DuckDuckGo search integration
- Search result processing and formatting
- Error handling and rate limiting
- Performance characteristics

Tests cover:
- Basic search functionality
- Result filtering and ranking
- Error scenarios and recovery
- Mock-based testing without external API calls
- Integration with PydanticAI tools
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from sifaka.tools.retrieval.web_search import DuckDuckGoSearchTool


class TestDuckDuckGoSearchTool:
    """Test the DuckDuckGo search tool implementation."""

    @pytest.fixture
    def search_tool(self):
        """Create a search tool instance for testing."""
        return DuckDuckGoSearchTool()

    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results for testing."""
        return [
            {
                "title": "Renewable Energy Benefits - EPA",
                "url": "https://www.epa.gov/renewable-energy",
                "snippet": "Renewable energy provides environmental and economic benefits including reduced greenhouse gas emissions.",
                "rank": 1
            },
            {
                "title": "Solar Power Advantages and Disadvantages",
                "url": "https://example.com/solar-power",
                "snippet": "Solar power offers clean energy but has initial cost considerations and weather dependencies.",
                "rank": 2
            },
            {
                "title": "Wind Energy Facts and Statistics",
                "url": "https://example.com/wind-energy",
                "snippet": "Wind energy is one of the fastest-growing renewable energy sources worldwide.",
                "rank": 3
            }
        ]

    def test_tool_initialization_default(self):
        """Test tool initialization with default parameters."""
        tool = DuckDuckGoSearchTool()
        
        assert tool.name == "duckduckgo_search"
        assert tool.max_results == 5
        assert tool.timeout == 10.0
        assert hasattr(tool, 'description')

    def test_tool_initialization_custom(self):
        """Test tool initialization with custom parameters."""
        tool = DuckDuckGoSearchTool(
            max_results=10,
            timeout=15.0,
            name="custom_search"
        )
        
        assert tool.name == "custom_search"
        assert tool.max_results == 10
        assert tool.timeout == 15.0

    def test_tool_initialization_invalid_params(self):
        """Test tool initialization with invalid parameters."""
        # Invalid max_results
        with pytest.raises(ValueError):
            DuckDuckGoSearchTool(max_results=0)
        
        with pytest.raises(ValueError):
            DuckDuckGoSearchTool(max_results=-1)
        
        # Invalid timeout
        with pytest.raises(ValueError):
            DuckDuckGoSearchTool(timeout=0)
        
        with pytest.raises(ValueError):
            DuckDuckGoSearchTool(timeout=-1)

    @pytest.mark.asyncio
    async def test_search_basic_functionality(self, search_tool, mock_search_results):
        """Test basic search functionality."""
        query = "renewable energy benefits"
        
        # Mock the actual search implementation
        with patch.object(search_tool, '_perform_search', return_value=mock_search_results):
            results = await search_tool.search_async(query)
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        # Verify result structure
        for result in results:
            assert "title" in result
            assert "url" in result
            assert "snippet" in result
            assert isinstance(result["title"], str)
            assert isinstance(result["url"], str)
            assert isinstance(result["snippet"], str)

    @pytest.mark.asyncio
    async def test_search_with_limit(self, search_tool, mock_search_results):
        """Test search with result limit."""
        query = "renewable energy"
        limit = 2
        
        # Mock search to return limited results
        limited_results = mock_search_results[:limit]
        with patch.object(search_tool, '_perform_search', return_value=limited_results):
            results = await search_tool.search_async(query, max_results=limit)
        
        assert len(results) == limit
        assert results[0]["title"] == "Renewable Energy Benefits - EPA"
        assert results[1]["title"] == "Solar Power Advantages and Disadvantages"

    @pytest.mark.asyncio
    async def test_search_empty_query(self, search_tool):
        """Test search with empty query."""
        with pytest.raises(ValueError):
            await search_tool.search_async("")

    @pytest.mark.asyncio
    async def test_search_whitespace_query(self, search_tool):
        """Test search with whitespace-only query."""
        with pytest.raises(ValueError):
            await search_tool.search_async("   \n\t  ")

    @pytest.mark.asyncio
    async def test_search_no_results(self, search_tool):
        """Test search when no results are found."""
        query = "very specific query with no results"
        
        # Mock search to return empty results
        with patch.object(search_tool, '_perform_search', return_value=[]):
            results = await search_tool.search_async(query)
        
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_network_error(self, search_tool):
        """Test search when network error occurs."""
        query = "test query"
        
        # Mock search to raise network error
        with patch.object(search_tool, '_perform_search', side_effect=ConnectionError("Network error")):
            with pytest.raises(ConnectionError):
                await search_tool.search_async(query)

    @pytest.mark.asyncio
    async def test_search_timeout_error(self, search_tool):
        """Test search when timeout occurs."""
        query = "test query"
        
        # Mock search to raise timeout error
        with patch.object(search_tool, '_perform_search', side_effect=TimeoutError("Request timeout")):
            with pytest.raises(TimeoutError):
                await search_tool.search_async(query)

    @pytest.mark.asyncio
    async def test_search_result_filtering(self, search_tool):
        """Test search result filtering and processing."""
        query = "renewable energy"
        
        # Mock results with some invalid entries
        mixed_results = [
            {
                "title": "Valid Result 1",
                "url": "https://example.com/valid1",
                "snippet": "Valid snippet content"
            },
            {
                "title": "",  # Invalid - empty title
                "url": "https://example.com/invalid1",
                "snippet": "Some content"
            },
            {
                "title": "Valid Result 2",
                "url": "",  # Invalid - empty URL
                "snippet": "Valid snippet"
            },
            {
                "title": "Valid Result 3",
                "url": "https://example.com/valid3",
                "snippet": "Another valid snippet"
            }
        ]
        
        with patch.object(search_tool, '_perform_search', return_value=mixed_results):
            results = await search_tool.search_async(query)
        
        # Should filter out invalid results
        assert len(results) <= len(mixed_results)
        
        # All returned results should be valid
        for result in results:
            assert len(result["title"]) > 0
            assert len(result["url"]) > 0
            assert result["url"].startswith("http")

    @pytest.mark.asyncio
    async def test_search_result_ranking(self, search_tool, mock_search_results):
        """Test that search results maintain proper ranking."""
        query = "renewable energy"
        
        with patch.object(search_tool, '_perform_search', return_value=mock_search_results):
            results = await search_tool.search_async(query)
        
        # Results should be in rank order
        for i, result in enumerate(results):
            if "rank" in result:
                assert result["rank"] == i + 1

    @pytest.mark.asyncio
    async def test_search_query_sanitization(self, search_tool, mock_search_results):
        """Test that search queries are properly sanitized."""
        # Query with special characters and extra whitespace
        query = "  renewable energy & solar power!  "
        
        with patch.object(search_tool, '_perform_search', return_value=mock_search_results) as mock_search:
            await search_tool.search_async(query)
        
        # Should call with sanitized query
        mock_search.assert_called_once()
        called_query = mock_search.call_args[0][0]
        assert called_query.strip() == called_query  # No leading/trailing whitespace
        assert len(called_query) > 0

    @pytest.mark.asyncio
    async def test_search_performance(self, search_tool, mock_search_results):
        """Test search performance characteristics."""
        query = "renewable energy"
        
        import time
        start_time = time.time()
        
        with patch.object(search_tool, '_perform_search', return_value=mock_search_results):
            results = await search_tool.search_async(query)
        
        end_time = time.time()
        
        # Should complete quickly (mock should be fast)
        assert (end_time - start_time) < 1.0
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_concurrent_requests(self, search_tool, mock_search_results):
        """Test handling of concurrent search requests."""
        import asyncio
        
        queries = [
            "renewable energy",
            "solar power",
            "wind energy",
            "hydroelectric power",
            "geothermal energy"
        ]
        
        with patch.object(search_tool, '_perform_search', return_value=mock_search_results):
            # Execute multiple searches concurrently
            tasks = [search_tool.search_async(query) for query in queries]
            results_list = await asyncio.gather(*tasks)
        
        # All searches should complete successfully
        assert len(results_list) == len(queries)
        for results in results_list:
            assert isinstance(results, list)
            assert len(results) > 0

    def test_search_tool_as_pydantic_tool(self, search_tool):
        """Test that the search tool can be used as a PydanticAI tool."""
        # Test tool metadata for PydanticAI integration
        assert hasattr(search_tool, 'name')
        assert hasattr(search_tool, 'description')
        
        # Test that it can be converted to PydanticAI tool format
        tool_dict = search_tool.to_pydantic_tool()
        
        assert "name" in tool_dict
        assert "description" in tool_dict
        assert callable(tool_dict.get("function"))

    @pytest.mark.asyncio
    async def test_search_with_custom_parameters(self, search_tool, mock_search_results):
        """Test search with custom parameters."""
        query = "renewable energy"
        
        # Test with custom parameters
        custom_params = {
            "region": "us-en",
            "safe_search": "moderate",
            "time_range": "month"
        }
        
        with patch.object(search_tool, '_perform_search', return_value=mock_search_results) as mock_search:
            results = await search_tool.search_async(query, **custom_params)
        
        # Should pass custom parameters to search implementation
        mock_search.assert_called_once()
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_result_deduplication(self, search_tool):
        """Test that duplicate search results are handled properly."""
        query = "renewable energy"
        
        # Mock results with duplicates
        duplicate_results = [
            {
                "title": "Renewable Energy Guide",
                "url": "https://example.com/guide",
                "snippet": "Comprehensive guide to renewable energy"
            },
            {
                "title": "Renewable Energy Guide",  # Duplicate title
                "url": "https://example.com/guide",  # Duplicate URL
                "snippet": "Comprehensive guide to renewable energy"
            },
            {
                "title": "Solar Power Basics",
                "url": "https://example.com/solar",
                "snippet": "Introduction to solar power technology"
            }
        ]
        
        with patch.object(search_tool, '_perform_search', return_value=duplicate_results):
            results = await search_tool.search_async(query)
        
        # Should handle duplicates appropriately (implementation dependent)
        assert isinstance(results, list)
        assert len(results) <= len(duplicate_results)

    def test_search_tool_configuration(self):
        """Test search tool configuration options."""
        # Test with various configuration options
        configs = [
            {"max_results": 3, "timeout": 5.0},
            {"max_results": 10, "timeout": 20.0},
            {"max_results": 1, "timeout": 30.0}
        ]
        
        for config in configs:
            tool = DuckDuckGoSearchTool(**config)
            assert tool.max_results == config["max_results"]
            assert tool.timeout == config["timeout"]

    def test_search_tool_string_representation(self):
        """Test string representation of search tool."""
        tool = DuckDuckGoSearchTool(max_results=5)
        
        str_repr = str(tool)
        assert "DuckDuckGoSearchTool" in str_repr
        assert "max_results=5" in str_repr

    @pytest.mark.asyncio
    async def test_search_with_special_characters(self, search_tool, mock_search_results):
        """Test search with special characters in query."""
        # Query with various special characters
        query = "renewable energy: solar & wind (2024) - benefits?"
        
        with patch.object(search_tool, '_perform_search', return_value=mock_search_results):
            results = await search_tool.search_async(query)
        
        # Should handle special characters gracefully
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_with_unicode(self, search_tool, mock_search_results):
        """Test search with unicode characters."""
        # Query with unicode characters
        query = "renewable energy 可再生能源 énergie renouvelable"
        
        with patch.object(search_tool, '_perform_search', return_value=mock_search_results):
            results = await search_tool.search_async(query)
        
        # Should handle unicode gracefully
        assert isinstance(results, list)
        assert len(results) > 0
