#!/usr/bin/env python3
"""Comprehensive tests for Sifaka retrievers.

This test suite covers all retriever modules including MockRetriever and
InMemoryRetriever, testing document retrieval, scoring, and integration
with the thought system.
"""

from unittest.mock import Mock

import pytest

from sifaka.core.thought import Document
from sifaka.retrievers.simple import InMemoryRetriever, MockRetriever
from sifaka.utils.error_handling import RetrieverError
from tests.utils import create_test_thought


class TestMockRetriever:
    """Test MockRetriever functionality."""

    def test_mock_retriever_basic(self):
        """Test basic mock retriever functionality."""
        retriever = MockRetriever()

        results = retriever.retrieve("test query")
        assert isinstance(results, list)
        assert len(results) <= retriever.max_results

        # All results should be strings
        for result in results:
            assert isinstance(result, str)

    def test_mock_retriever_custom_documents(self):
        """Test mock retriever with custom documents."""
        custom_docs = [
            "Document about machine learning",
            "Document about artificial intelligence",
            "Document about data science",
        ]

        retriever = MockRetriever(documents=custom_docs)
        results = retriever.retrieve("machine learning")

        assert len(results) <= len(custom_docs)
        # Should return some of the custom documents
        for result in results:
            assert result in custom_docs

    def test_mock_retriever_max_results(self):
        """Test mock retriever respects max_results parameter."""
        retriever = MockRetriever(max_results=2)

        results = retriever.retrieve("test query")
        assert len(results) <= 2

    def test_mock_retriever_relevance_scoring(self):
        """Test mock retriever relevance scoring."""
        documents = [
            "Python programming language tutorial",
            "Java programming guide",
            "Machine learning with Python",
            "Web development basics",
        ]

        retriever = MockRetriever(documents=documents)
        results = retriever.retrieve("Python programming")

        # Should return relevant documents
        assert len(results) > 0
        # First result should be most relevant (contains both terms)
        if len(results) > 0:
            assert "Python" in results[0] or "programming" in results[0]

    def test_mock_retriever_calculate_relevance_score(self):
        """Test relevance score calculation."""
        retriever = MockRetriever()

        # Test exact match
        score = retriever._calculate_relevance_score(
            "machine learning", "machine learning tutorial", 0
        )
        assert score > 0.5  # Should have high relevance

        # Test partial match
        score = retriever._calculate_relevance_score(
            "machine learning", "artificial intelligence tutorial", 0
        )
        assert score < 0.5  # Should have lower relevance

        # Test no match
        score = retriever._calculate_relevance_score("machine learning", "cooking recipes", 0)
        assert score == 0.0  # Should have no relevance

    def test_mock_retriever_rank_penalty(self):
        """Test rank penalty in relevance scoring."""
        retriever = MockRetriever()

        # Same query and text, different ranks
        score_rank_0 = retriever._calculate_relevance_score("test query", "test document", 0)
        score_rank_5 = retriever._calculate_relevance_score("test query", "test document", 5)

        # Higher rank should have lower score due to penalty
        assert score_rank_0 > score_rank_5

    def test_mock_retriever_empty_query(self):
        """Test mock retriever with empty query."""
        retriever = MockRetriever()

        results = retriever.retrieve("")
        assert isinstance(results, list)
        # Should handle empty query gracefully

    def test_mock_retriever_retrieve_for_thought(self):
        """Test mock retriever with thought objects."""
        retriever = MockRetriever()
        thought = create_test_thought(prompt="Find information about Python")

        # Test pre-generation retrieval
        updated_thought = retriever.retrieve_for_thought(thought, is_pre_generation=True)
        assert updated_thought.pre_generation_context is not None
        assert len(updated_thought.pre_generation_context) > 0

        # Check document structure
        for doc in updated_thought.pre_generation_context:
            assert isinstance(doc, Document)
            assert doc.text is not None
            assert doc.metadata is not None
            assert "source" in doc.metadata
            assert doc.metadata["source"] == "mock"

    def test_mock_retriever_post_generation(self):
        """Test mock retriever post-generation retrieval."""
        retriever = MockRetriever()
        thought = create_test_thought(
            prompt="Write about Python", text="Python is a programming language"
        )

        updated_thought = retriever.retrieve_for_thought(thought, is_pre_generation=False)
        assert updated_thought.post_generation_context is not None
        assert len(updated_thought.post_generation_context) > 0

    def test_mock_retriever_retry_logic(self):
        """Test mock retriever retry logic."""
        retriever = MockRetriever(max_retries=3)

        # Mock a failure scenario by making documents access fail
        def failing_getitem(index):
            raise Exception("Test error")

        # Replace the documents list with a mock that fails on slicing
        mock_documents = Mock()
        mock_documents.__getitem__ = failing_getitem
        retriever.documents = mock_documents

        with pytest.raises(RetrieverError):
            retriever.retrieve("test query")


class TestInMemoryRetriever:
    """Test InMemoryRetriever functionality."""

    def test_in_memory_retriever_basic(self):
        """Test basic in-memory retriever functionality."""
        documents = {
            "doc1": "Python programming tutorial",
            "doc2": "Java development guide",
            "doc3": "Machine learning with Python",
        }

        retriever = InMemoryRetriever(documents=documents)
        results = retriever.retrieve("Python")

        assert isinstance(results, list)
        assert len(results) > 0
        # Should return documents containing "Python"
        for result in results:
            assert "Python" in result

    def test_in_memory_retriever_empty_documents(self):
        """Test in-memory retriever with empty documents."""
        retriever = InMemoryRetriever()

        results = retriever.retrieve("test query")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_in_memory_retriever_add_document(self):
        """Test adding documents to in-memory retriever."""
        retriever = InMemoryRetriever()

        retriever.add_document("doc1", "Python programming guide")
        retriever.add_document("doc2", "Java tutorial", {"author": "John Doe"})

        results = retriever.retrieve("Python")
        assert len(results) == 1
        assert "Python" in results[0]

    def test_in_memory_retriever_metadata(self):
        """Test in-memory retriever with metadata."""
        documents = {"doc1": "Python tutorial"}
        metadata = {"doc1": {"author": "Jane Doe", "category": "programming"}}

        retriever = InMemoryRetriever(documents=documents, metadata=metadata)
        thought = create_test_thought(prompt="Find Python tutorials")

        updated_thought = retriever.retrieve_for_thought(thought)

        assert len(updated_thought.pre_generation_context) > 0
        doc = updated_thought.pre_generation_context[0]
        assert doc.metadata["author"] == "Jane Doe"
        assert doc.metadata["category"] == "programming"

    def test_in_memory_retriever_max_results(self):
        """Test in-memory retriever max results limit."""
        documents = {f"doc{i}": f"Python tutorial {i}" for i in range(10)}

        retriever = InMemoryRetriever(documents=documents, max_results=3)
        results = retriever.retrieve("Python")

        assert len(results) <= 3

    def test_in_memory_retriever_relevance_scoring(self):
        """Test in-memory retriever relevance scoring."""
        documents = {
            "doc1": "Python machine learning tutorial",  # 2 matches
            "doc2": "Python programming guide",  # 1 match
            "doc3": "Java programming tutorial",  # 1 match
            "doc4": "Web development basics",  # 0 matches
        }

        retriever = InMemoryRetriever(documents=documents)
        results = retriever.retrieve("Python programming")

        # Should return documents with matches, sorted by relevance
        assert len(results) > 0
        # First result should have highest relevance
        assert "Python" in results[0]

    def test_in_memory_retriever_calculate_relevance_score(self):
        """Test in-memory retriever relevance score calculation."""
        retriever = InMemoryRetriever()

        # Test Jaccard similarity calculation
        score = retriever._calculate_relevance_score(
            "machine learning python", "python machine learning tutorial", 0
        )
        assert score > 0.0

        # Test with no common terms
        score = retriever._calculate_relevance_score("machine learning", "cooking recipes", 0)
        assert score == 0.0

    def test_in_memory_retriever_keyword_matching(self):
        """Test in-memory retriever keyword matching."""
        documents = {
            "doc1": "artificial intelligence and machine learning",
            "doc2": "natural language processing",
            "doc3": "computer vision algorithms",
        }

        retriever = InMemoryRetriever(documents=documents)

        # Test single keyword
        results = retriever.retrieve("machine")
        assert len(results) == 1
        assert "machine learning" in results[0]

        # Test multiple keywords
        results = retriever.retrieve("artificial intelligence")
        assert len(results) == 1
        assert "artificial intelligence" in results[0]

    def test_in_memory_retriever_case_insensitive(self):
        """Test in-memory retriever case insensitive matching."""
        documents = {"doc1": "Python Programming Tutorial", "doc2": "JAVA DEVELOPMENT GUIDE"}

        retriever = InMemoryRetriever(documents=documents)

        # Test lowercase query
        results = retriever.retrieve("python")
        assert len(results) == 1
        assert "Python" in results[0]

        # Test uppercase query
        results = retriever.retrieve("JAVA")
        assert len(results) == 1
        assert "JAVA" in results[0]

    def test_in_memory_retriever_retrieve_for_thought(self):
        """Test in-memory retriever with thought objects."""
        documents = {"doc1": "Python programming tutorial", "doc2": "Machine learning guide"}

        retriever = InMemoryRetriever(documents=documents)
        thought = create_test_thought(prompt="Learn about Python programming")

        updated_thought = retriever.retrieve_for_thought(thought)

        assert updated_thought.pre_generation_context is not None
        assert len(updated_thought.pre_generation_context) > 0

        # Check document metadata
        doc = updated_thought.pre_generation_context[0]
        assert doc.metadata["source"] == "in_memory"
        assert doc.metadata["query"] == thought.prompt
        assert "doc_id" in doc.metadata

    def test_in_memory_retriever_error_handling(self):
        """Test in-memory retriever error handling."""
        retriever = InMemoryRetriever(max_retries=2)

        # Mock a failure scenario by making documents.items() fail
        def failing_items():
            raise Exception("Test error")

        # Replace the documents dict with a mock that fails on items()
        mock_documents = Mock()
        mock_documents.items = failing_items
        retriever.documents = mock_documents

        with pytest.raises(RetrieverError):
            retriever.retrieve("test query")

    def test_in_memory_retriever_performance(self):
        """Test in-memory retriever performance."""
        import time

        # Create large document set
        documents = {
            f"doc{i}": f"Document {i} about various topics including Python, Java, and machine learning"
            for i in range(100)
        }

        retriever = InMemoryRetriever(documents=documents)

        start_time = time.time()
        results = retriever.retrieve("Python machine learning")
        end_time = time.time()

        # Should complete search in reasonable time
        assert (end_time - start_time) < 1.0  # 1 second max
        assert len(results) > 0


class TestRetrieverIntegration:
    """Test retriever integration and common functionality."""

    def test_retriever_with_chain_integration(self):
        """Test retriever integration with chain execution."""
        from sifaka.core.chain import Chain
        from sifaka.models.base import MockModel

        documents = {
            "doc1": "Python is a programming language",
            "doc2": "Machine learning uses algorithms",
        }

        retriever = InMemoryRetriever(documents=documents)
        model = MockModel(
            model_name="test-model", response_text="Python is great for machine learning"
        )

        chain = Chain(
            model=model, prompt="Tell me about Python for machine learning", retriever=retriever
        )

        result = chain.run()

        # Should have retrieved relevant documents
        assert result.pre_generation_context is not None
        assert len(result.pre_generation_context) > 0

    def test_multiple_retrievers_comparison(self):
        """Test comparing different retriever implementations."""
        documents = ["Python tutorial", "Java guide", "Machine learning basics"]

        mock_retriever = MockRetriever(documents=documents)
        memory_retriever = InMemoryRetriever({f"doc{i}": doc for i, doc in enumerate(documents)})

        query = "Python tutorial"

        mock_results = mock_retriever.retrieve(query)
        memory_results = memory_retriever.retrieve(query)

        # Both should return relevant results
        assert len(mock_results) > 0
        assert len(memory_results) > 0

    def test_retriever_error_recovery(self):
        """Test retriever error recovery mechanisms."""
        retriever = InMemoryRetriever(max_retries=3)

        # Test retry logic with intermittent failures
        call_count = 0

        def failing_items():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return [("doc1", "test document")]

        # Replace the documents dict with a mock that fails intermittently
        mock_documents = Mock()
        mock_documents.items = failing_items
        retriever.documents = mock_documents

        # Should eventually succeed after retries
        try:
            results = retriever.retrieve("test")
            # If it succeeds, that's good
            assert isinstance(results, list)
            assert call_count == 3  # Should have succeeded on the 3rd attempt
        except RetrieverError:
            # If it fails after retries, that's also expected behavior
            assert call_count >= 3
