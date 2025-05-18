"""
Detailed tests for the simple retriever.

This module contains more comprehensive tests for the simple retriever
to improve test coverage.
"""

from sifaka.retrievers.simple import SimpleRetriever


# Mock documents for testing
MOCK_DOCUMENTS = [
    "This is the first test document about cats and dogs.",
    "The second document discusses machine learning and AI.",
    "Document three is about natural language processing.",
    "The fourth document talks about cats and their behavior.",
    "Document five covers deep learning techniques.",
]


class TestSimpleRetrieverDetailed:
    """Detailed tests for the SimpleRetriever."""

    def test_init_with_documents(self) -> None:
        """Test initialization with documents."""
        # SimpleRetriever doesn't accept embedding_model or top_k parameters
        retriever = SimpleRetriever(documents=MOCK_DOCUMENTS)

        assert retriever.documents == MOCK_DOCUMENTS
        # SimpleRetriever doesn't store embeddings

    def test_init_with_empty_documents(self) -> None:
        """Test initialization with empty documents list."""
        retriever = SimpleRetriever(documents=[])

        assert retriever.documents == []

    # SimpleRetriever doesn't support custom similarity functions
    # Removing test_init_with_custom_similarity_function

    # SimpleRetriever doesn't compute embeddings
    # Removing test_compute_embeddings

    # SimpleRetriever doesn't have a cosine_similarity method
    # Removing test_cosine_similarity

    def test_retrieve(self) -> None:
        """Test retrieve method."""
        retriever = SimpleRetriever(documents=MOCK_DOCUMENTS)

        # Test retrieving documents about cats
        results = retriever.retrieve("cats")
        assert len(results) > 0
        assert any("cats" in doc.lower() for doc in results)

        # Test retrieving documents about machine learning
        results = retriever.retrieve("machine learning")
        assert len(results) > 0
        assert any("machine learning" in doc.lower() for doc in results)

        # Test retrieving documents about NLP - use "natural" instead of "nlp"
        # since SimpleRetriever filters out words shorter than 3 characters
        results = retriever.retrieve("natural language")
        assert len(results) > 0
        assert any("natural language processing" in doc.lower() for doc in results)

    def test_retrieve_with_empty_documents(self) -> None:
        """Test retrieve method with empty documents."""
        retriever = SimpleRetriever(documents=[])

        results = retriever.retrieve("test query")
        assert results == []

    # SimpleRetriever doesn't support custom similarity functions
    # Removing test_retrieve_with_custom_similarity

    # SimpleRetriever doesn't use embedding models
    # Removing test_retrieve_error
