"""Integration examples showing how to use retrieval with Sifaka's improve() API."""

import asyncio
from typing import List, Dict, Any, Optional
from sifaka import improve
from sifaka.critics.self_rag_enhanced import SelfRAGEnhancedCritic, RetrievalBackend


class FactCheckingBackend(RetrievalBackend):
    """A fact-checking retrieval backend using multiple sources."""

    def __init__(self):
        # In production, these would be real knowledge bases
        self.facts_db = {
            "science": [
                {
                    "content": "Water boils at 100°C (212°F) at sea level.",
                    "source": "Physics textbook",
                },
                {
                    "content": "The Earth orbits the Sun once every 365.25 days.",
                    "source": "Astronomy guide",
                },
                {
                    "content": "DNA contains four bases: adenine, thymine, guanine, and cytosine.",
                    "source": "Biology reference",
                },
            ],
            "history": [
                {
                    "content": "World War II ended in 1945.",
                    "source": "Historical records",
                },
                {
                    "content": "The American Revolution began in 1775.",
                    "source": "US History",
                },
                {
                    "content": "The Berlin Wall fell on November 9, 1989.",
                    "source": "Modern history",
                },
            ],
            "technology": [
                {
                    "content": "HTTP/3 uses QUIC transport protocol.",
                    "source": "IETF standards",
                },
                {
                    "content": "Bitcoin was created in 2009 by Satoshi Nakamoto.",
                    "source": "Cryptocurrency history",
                },
                {
                    "content": "The first iPhone was released in 2007.",
                    "source": "Tech timeline",
                },
            ],
        }

    async def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve facts relevant to the query."""
        results = []
        query_lower = query.lower()

        # Search across all categories
        for category, facts in self.facts_db.items():
            for fact in facts:
                # Simple relevance scoring
                content_lower = fact["content"].lower()
                score = self._calculate_relevance(query_lower, content_lower)

                if score > 0.3:  # Relevance threshold
                    results.append(
                        {
                            "content": fact["content"],
                            "source": f"{fact['source']} ({category})",
                            "score": score,
                        }
                    )

        # Sort by relevance and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate simple relevance score."""
        query_words = set(query.split())
        content_words = set(content.split())

        # Overlap-based scoring
        overlap = len(query_words & content_words)
        return overlap / max(len(query_words), 1)

    async def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add new facts to the database."""
        for doc in documents:
            category = doc.get("category", "general")
            if category not in self.facts_db:
                self.facts_db[category] = []
            self.facts_db[category].append(
                {"content": doc["content"], "source": doc.get("source", "User added")}
            )


# Example 1: Using retrieval-augmented critique with improve()
async def example_with_fact_checking():
    """Example using improve() with fact-checking retrieval."""

    # Create fact-checking backend
    fact_checker = FactCheckingBackend()

    # Add some custom facts
    await fact_checker.add_documents(
        [
            {
                "content": "Sifaka is a text improvement framework using AI critics.",
                "source": "Sifaka documentation",
                "category": "technology",
            },
            {
                "content": "Large Language Models can have billions of parameters.",
                "source": "AI research papers",
                "category": "technology",
            },
        ]
    )

    # Create retrieval-augmented critic
    rag_critic = SelfRAGEnhancedCritic(
        retrieval_backend=fact_checker, retrieval_threshold=0.6
    )

    # Text with some factual claims
    text = """
    World War II ended in 1946 when Germany surrendered. This historical event
    shaped the modern world. In the technology sphere, the first iPhone was
    released in 2008, revolutionizing mobile computing. Water boils at 95°C
    at sea level, which is why cooking times vary at different altitudes.

    Modern AI systems like Large Language Models can have millions of parameters
    and are trained on vast amounts of text data.
    """

    print("Original text:")
    print(text)
    print("\n" + "=" * 80 + "\n")

    # Use improve() with our custom critic
    result = await improve(
        text, critics=[rag_critic], max_iterations=2  # Can mix with other critics too
    )

    print("Improved text:")
    print(result.final_text)
    print(f"\nIterations: {result.iteration}")
    print(f"Improvement achieved: {not result.needs_improvement}")

    # Show what facts were checked
    for critique in result.critiques:
        if (
            critique.critic == "self_rag_enhanced"
            and "retrieval_context" in critique.metadata
        ):
            context = critique.metadata["retrieval_context"]
            print("\nFact-checking results:")
            for claim in context.get("verified_claims", []):
                print(f"- Checked: {claim['claim'][:50]}...")
                print(f"  Sources: {', '.join(claim['sources'])}")


# Example 2: Combining with other critics
async def example_combined_critics():
    """Example combining retrieval-augmented critic with others."""

    # Create a domain-specific fact base
    fact_checker = FactCheckingBackend()

    # Create retrieval-augmented critic
    rag_critic = SelfRAGEnhancedCritic(
        retrieval_backend=fact_checker, retrieval_threshold=0.7
    )

    text = """
    Python is a programming language created in 1985. It's known for being
    easy to learn and having a simple syntax. Python is widely used in data
    science and machine learning applications.
    """

    # Combine with other critics for comprehensive improvement
    result = await improve(
        text,
        critics=[
            rag_critic,  # Fact checking
            "constitutional",  # Principle-based review
            "self_refine",  # General quality improvement
        ],
        max_iterations=3,
    )

    print("Combined critic improvement:")
    print(result.final_text)

    # Show feedback from each critic
    print("\nFeedback summary:")
    for critique in result.critiques:
        print(f"\n{critique.critic} (confidence: {critique.confidence:.2f}):")
        print(f"- {critique.feedback[:200]}...")


# Example 3: Creating a custom retrieval backend for your domain
class CodeDocumentationBackend(RetrievalBackend):
    """Retrieval backend for code documentation."""

    def __init__(self, docs_path: Optional[str] = None):
        self.documentation = {
            "python_stdlib": {
                "os": "The os module provides a way of using operating system dependent functionality.",
                "sys": "The sys module provides access to interpreter variables and functions.",
                "json": "The json module implements JSON encoder and decoder.",
            },
            "popular_packages": {
                "numpy": "NumPy is the fundamental package for scientific computing with Python.",
                "pandas": "pandas provides data structures and data analysis tools.",
                "requests": "requests is an elegant HTTP library for Python.",
            },
        }

    async def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documentation."""
        results = []

        for category, docs in self.documentation.items():
            for module, description in docs.items():
                if module in query.lower() or any(
                    word in description.lower() for word in query.lower().split()
                ):
                    results.append(
                        {
                            "content": f"{module}: {description}",
                            "source": f"Python docs - {category}",
                            "score": 0.8,
                        }
                    )

        return results[:top_k]

    async def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add custom documentation."""
        for doc in documents:
            category = doc.get("category", "custom")
            if category not in self.documentation:
                self.documentation[category] = {}

            module = doc.get("module", "unknown")
            self.documentation[category][module] = doc["content"]


async def example_code_documentation():
    """Example for checking code documentation accuracy."""

    # Create code documentation backend
    code_docs = CodeDocumentationBackend()

    # Add custom package docs
    await code_docs.add_documents(
        [
            {
                "module": "sifaka",
                "content": "Sifaka is a text improvement framework using multiple AI critics.",
                "category": "custom",
            }
        ]
    )

    # Create critic with code documentation
    code_critic = SelfRAGEnhancedCritic(
        retrieval_backend=code_docs, retrieval_threshold=0.5
    )

    text = """
    To read files in Python, you can use the os module which provides file
    reading capabilities. For data analysis, numpy is great for string manipulation
    and pandas excels at numerical computations. The requests library is used for
    file system operations.
    """

    result = await improve(text, critics=[code_critic], max_iterations=2)

    print("Corrected technical documentation:")
    print(result.final_text)


async def main():
    """Run all examples."""
    print("Example 1: Fact-checking with retrieval")
    print("=" * 80)
    await example_with_fact_checking()

    print("\n\nExample 2: Combined critics")
    print("=" * 80)
    await example_combined_critics()

    print("\n\nExample 3: Code documentation checking")
    print("=" * 80)
    await example_code_documentation()


if __name__ == "__main__":
    asyncio.run(main())
