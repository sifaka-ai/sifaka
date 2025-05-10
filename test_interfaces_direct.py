"""
Test script to verify that the interfaces are working correctly.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import directly from the interface files
from sifaka.interfaces.model import ModelProviderProtocol, AsyncModelProviderProtocol
from sifaka.interfaces.client import APIClientProtocol
from sifaka.interfaces.counter import TokenCounterProtocol
from sifaka.interfaces.retrieval import Retriever, AsyncRetriever, DocumentStore, IndexManager, QueryProcessor

def test_interfaces():
    """Test that the interfaces can be imported correctly."""
    print("Successfully imported interfaces:")
    print(f"- ModelProviderProtocol: {ModelProviderProtocol}")
    print(f"- AsyncModelProviderProtocol: {AsyncModelProviderProtocol}")
    print(f"- APIClientProtocol: {APIClientProtocol}")
    print(f"- TokenCounterProtocol: {TokenCounterProtocol}")
    print(f"- Retriever: {Retriever}")
    print(f"- AsyncRetriever: {AsyncRetriever}")
    print(f"- DocumentStore: {DocumentStore}")
    print(f"- IndexManager: {IndexManager}")
    print(f"- QueryProcessor: {QueryProcessor}")

if __name__ == "__main__":
    test_interfaces()
