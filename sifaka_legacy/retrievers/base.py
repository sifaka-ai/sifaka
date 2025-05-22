"""
Base retriever for Sifaka.

This module provides the base retriever class that all retrievers should inherit from.
"""

from abc import ABC, abstractmethod
from typing import List


class Retriever(ABC):
    """Base class for retrievers.

    A retriever is responsible for retrieving relevant documents for a query.
    All retrievers should inherit from this class and implement the retrieve method.
    """

    @abstractmethod
    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
