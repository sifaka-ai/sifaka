"""
Test script to verify that the interfaces are working correctly.
"""

from sifaka.interfaces import (
    ModelProviderProtocol,
    APIClientProtocol,
    TokenCounterProtocol,
    Retriever,
    AsyncRetriever,
    DocumentStore,
    IndexManager,
    QueryProcessor,
)

def test_interfaces():
    """Test that the interfaces can be imported correctly."""
    print("Successfully imported interfaces:")
    print(f"- ModelProviderProtocol: {ModelProviderProtocol}")
    print(f"- APIClientProtocol: {APIClientProtocol}")
    print(f"- TokenCounterProtocol: {TokenCounterProtocol}")
    print(f"- Retriever: {Retriever}")
    print(f"- AsyncRetriever: {AsyncRetriever}")
    print(f"- DocumentStore: {DocumentStore}")
    print(f"- IndexManager: {IndexManager}")
    print(f"- QueryProcessor: {QueryProcessor}")

if __name__ == "__main__":
    test_interfaces()
