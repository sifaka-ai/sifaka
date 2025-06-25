"""Example of using SelfRAG critic with different retrieval backends."""

import asyncio
from typing import List, Dict, Any


# Mock implementations for demonstration
class MockRedisClient:
    """Mock Redis client for demonstration."""

    def __init__(self):
        self.data = {}
        self.search_index = {}

    def ft(self, index_name):
        return self

    def search(self, query):
        # Simple mock search
        class Result:
            docs = []

        return Result()

    def pipeline(self):
        return self

    def hset(self, key, mapping):
        self.data[key] = mapping
        return self

    def execute(self):
        pass


class SimpleDictRetrievalBackend:
    """Simple dictionary-based retrieval for demonstration."""

    def __init__(self):
        self.documents = []

    async def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval."""
        query_lower = query.lower()
        results = []

        for doc in self.documents:
            content_lower = doc["content"].lower()
            # Simple scoring based on keyword overlap
            score = sum(
                1 for word in query_lower.split() if word in content_lower
            ) / len(query_lower.split())

            if score > 0:
                results.append(
                    {"content": doc["content"], "source": doc["source"], "score": score}
                )

        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add documents to the corpus."""
        self.documents.extend(documents)


async def demo_self_rag_with_retrieval():
    """Demonstrate SelfRAG with actual retrieval."""

    # Create a simple retrieval backend
    retrieval_backend = SimpleDictRetrievalBackend()

    # Add some facts to retrieve
    await retrieval_backend.add_documents(
        [
            {
                "content": "The Great Wall of China is approximately 21,196 kilometers long.",
                "source": "UNESCO World Heritage Site data",
            },
            {
                "content": "The Great Wall was built over many dynasties, primarily during the Ming Dynasty (1368-1644).",
                "source": "Historical records",
            },
            {
                "content": "Python 3.12 was released on October 2, 2023.",
                "source": "Python.org release notes",
            },
            {
                "content": "Machine learning models require training data to learn patterns.",
                "source": "ML textbook",
            },
            {
                "content": "The Eiffel Tower is 330 meters tall and was completed in 1889.",
                "source": "Paris tourist information",
            },
        ]
    )

    # Import and create the enhanced critic
    from sifaka.critics.self_rag_enhanced import SelfRAGEnhancedCritic

    # Create critic with retrieval
    critic = SelfRAGEnhancedCritic(
        retrieval_backend=retrieval_backend,
        retrieval_threshold=0.6,  # Retrieve for claims with >60% confidence
    )

    # Test text with factual claims
    test_text = """
    The Great Wall of China is about 20,000 kilometers long and was built primarily
    during the Ming Dynasty. It's one of the most impressive architectural achievements
    in human history. The wall was built to protect against invasions and took
    centuries to complete.

    In comparison, modern achievements like Python programming language, which was
    released in 1990, show how quickly technology advances. Python has become one
    of the most popular languages for machine learning, which requires large amounts
    of data to train models effectively.
    """

    print("Testing SelfRAG with retrieval backend...")
    print("=" * 80)

    # Get critique
    from sifaka.core.models import SifakaResult

    result = SifakaResult(original_text=test_text, final_text=test_text)

    critique = await critic.critique(test_text, result)

    print(f"Critic: {critique.critic}")
    print(f"Confidence: {critique.confidence:.2f}")
    print(f"\nFeedback:\n{critique.feedback}")
    print("\nSuggestions:")
    for i, suggestion in enumerate(critique.suggestions, 1):
        print(f"{i}. {suggestion}")

    # Show retrieval context
    if "retrieval_context" in critique.metadata:
        context = critique.metadata["retrieval_context"]
        print("\nRetrieval Context:")
        print(f"- Retrieval available: {context.get('retrieval_available', False)}")

        if context.get("verified_claims"):
            print("\nVerified Claims:")
            for claim in context["verified_claims"]:
                print(f"\n  Claim: {claim['claim']}")
                print(f"  Retrieved docs: {claim['retrieved']}")
                print(f"  Sources: {', '.join(claim['sources'])}")


async def demo_self_rag_without_retrieval():
    """Demonstrate SelfRAG without retrieval (fallback mode)."""

    from sifaka.critics.self_rag_enhanced import SelfRAGEnhancedCritic

    # Create critic without retrieval backend
    critic = SelfRAGEnhancedCritic(
        retrieval_backend=None, retrieval_threshold=0.6  # No retrieval available
    )

    test_text = """
    Quantum computers can solve certain problems exponentially faster than classical
    computers. They use quantum bits (qubits) that can exist in superposition.
    Google achieved quantum supremacy in 2019 with their Sycamore processor.
    """

    print("\nTesting SelfRAG without retrieval backend...")
    print("=" * 80)

    from sifaka.core.models import SifakaResult

    result = SifakaResult(original_text=test_text, final_text=test_text)

    critique = await critic.critique(test_text, result)

    print(f"Feedback:\n{critique.feedback}")
    print("\nUnverified claims identified:")

    if "retrieval_context" in critique.metadata:
        context = critique.metadata["retrieval_context"]
        for claim in context.get("unverified_claims", []):
            print(f"- {claim['claim']}")
            print(f"  Suggested query: {claim['query']}")


async def demo_custom_retrieval():
    """Demonstrate with a custom retrieval function."""

    # Create a custom retrieval backend
    class WikipediaRetrievalBackend:
        """Mock Wikipedia retrieval."""

        async def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
            # In reality, this would call Wikipedia API
            # For demo, return mock results based on query
            if "great wall" in query.lower():
                return [
                    {
                        "content": "The Great Wall of China has a total length of 21,196.18 km (13,170.70 mi).",
                        "source": "Wikipedia: Great Wall of China",
                        "score": 0.95,
                    }
                ]
            elif "python" in query.lower():
                return [
                    {
                        "content": "Python was first released in 1991 by Guido van Rossum.",
                        "source": "Wikipedia: Python (programming language)",
                        "score": 0.90,
                    }
                ]
            return []

        async def add_documents(self, documents: List[Dict[str, str]]) -> None:
            pass  # Wikipedia is read-only

    from sifaka.critics.self_rag_enhanced import SelfRAGEnhancedCritic

    critic = SelfRAGEnhancedCritic(
        retrieval_backend=WikipediaRetrievalBackend(), retrieval_threshold=0.5
    )

    test_text = "The Great Wall of China is 30,000 km long. Python was created in 1985."

    print("\nTesting SelfRAG with custom Wikipedia retrieval...")
    print("=" * 80)

    from sifaka.core.models import SifakaResult

    result = SifakaResult(original_text=test_text, final_text=test_text)

    critique = await critic.critique(test_text, result)
    print(f"Feedback:\n{critique.feedback}")

    # The critic should identify the incorrect facts and suggest corrections
    # based on the retrieved Wikipedia data


async def main():
    """Run all demonstrations."""
    await demo_self_rag_with_retrieval()
    print("\n" + "=" * 80 + "\n")

    await demo_self_rag_without_retrieval()
    print("\n" + "=" * 80 + "\n")

    await demo_custom_retrieval()


if __name__ == "__main__":
    asyncio.run(main())
