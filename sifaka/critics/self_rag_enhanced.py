"""Enhanced Self-RAG implementation with pluggable retrieval.

Based on: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
Paper: https://arxiv.org/abs/2310.15657
Authors: Asai et al. (2023)

This implementation supports pluggable retrieval backends while maintaining
the core Self-RAG critique functionality.
"""

from typing import Optional, Union, List, Dict, Any, Protocol
from abc import abstractmethod
import json

from ..core.models import SifakaResult, CritiqueResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


class RetrievalBackend(Protocol):
    """Protocol for retrieval backends."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents/facts for a query.

        Returns:
            List of dicts with keys: 'content', 'source', 'score'
        """
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add documents to the retrieval corpus."""
        pass


class SelfRAGEnhancedCritic(BaseCritic):
    """Self-RAG critic with optional retrieval augmentation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        retrieval_backend: Optional[RetrievalBackend] = None,
        retrieval_threshold: float = 0.7,
    ):
        """Initialize with optional retrieval backend.

        Args:
            retrieval_backend: Optional backend for retrieval
            retrieval_threshold: Confidence threshold for when to retrieve (0-1)
        """
        super().__init__(model, temperature, config or Config(), provider, api_key)
        self.retrieval_backend = retrieval_backend
        self.retrieval_threshold = retrieval_threshold

    @property
    def name(self) -> str:
        return "self_rag_enhanced"

    async def _identify_claims_needing_retrieval(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Identify factual claims that need verification."""
        identification_prompt = f"""Analyze this text and identify specific factual claims that would benefit from external verification:

Text: {text}

For each claim, provide:
1. The exact claim text
2. A retrieval query to verify it
3. Why it needs verification (e.g., specific date, statistic, technical fact)
4. Confidence that retrieval would help (0-1)

Format as JSON list:
[
  {{
    "claim": "exact text of claim",
    "query": "retrieval query",
    "reason": "why needs verification",
    "confidence": 0.8
  }}
]
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at identifying factual claims that need verification.",
            },
            {"role": "user", "content": identification_prompt},
        ]

        response = await self.llm_client.complete(messages, temperature=0.3)

        try:
            claims = json.loads(response.content)
            # Filter by confidence threshold
            return [
                c for c in claims if c.get("confidence", 0) >= self.retrieval_threshold
            ]
        except:
            return []

    async def _retrieve_and_verify(
        self, claims: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Retrieve information and verify claims."""
        if not self.retrieval_backend:
            # No backend, just return claims needing verification
            return {
                "retrieval_available": False,
                "unverified_claims": claims,
                "verification_status": "No retrieval backend available",
            }

        verification_results = []

        for claim in claims:
            # Retrieve relevant information
            retrieved_docs = await self.retrieval_backend.retrieve(
                claim["query"], top_k=3
            )

            # Ask LLM to verify claim against retrieved content
            if retrieved_docs:
                verification_prompt = f"""Verify this claim against the retrieved information:

Claim: {claim['claim']}

Retrieved Information:
{self._format_retrieved_docs(retrieved_docs)}

Provide:
1. Is the claim supported, contradicted, or unclear?
2. Relevant evidence from retrieved content
3. Confidence in verification (0-1)
"""

                messages = [
                    {"role": "system", "content": "You are a fact checker."},
                    {"role": "user", "content": verification_prompt},
                ]

                verification = await self.llm_client.complete(messages, temperature=0.1)

                verification_results.append(
                    {
                        "claim": claim["claim"],
                        "query": claim["query"],
                        "retrieved": len(retrieved_docs),
                        "verification": verification.content,
                        "sources": [
                            doc.get("source", "Unknown") for doc in retrieved_docs
                        ],
                    }
                )
            else:
                verification_results.append(
                    {
                        "claim": claim["claim"],
                        "query": claim["query"],
                        "retrieved": 0,
                        "verification": "No relevant information found",
                        "sources": [],
                    }
                )

        return {
            "retrieval_available": True,
            "verified_claims": verification_results,
            "verification_status": f"Verified {len(verification_results)} claims",
        }

    def _format_retrieved_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for prompt."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"{i}. Source: {doc.get('source', 'Unknown')}")
            formatted.append(f"   Content: {doc.get('content', '')}")
            formatted.append(f"   Relevance: {doc.get('score', 0):.2f}")
        return "\n".join(formatted)

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for critique with retrieval context."""
        # First, identify claims needing retrieval
        claims = await self._identify_claims_needing_retrieval(text)

        # Retrieve and verify if we have claims
        retrieval_context = await self._retrieve_and_verify(claims) if claims else {}

        # Store retrieval info in metadata for confidence calculation
        self._current_retrieval_context = retrieval_context

        # Build critique prompt with retrieval context
        critique_prompt = f"""Evaluate this text with a focus on factual accuracy and verifiability:

Text: {text}

{self._format_retrieval_context(retrieval_context)}

Provide critique focusing on:
1. Factual accuracy and claims that need verification
2. Whether sufficient evidence is provided
3. Clarity and logical consistency
4. Suggestions for improvement

Include specific suggestions for:
- Adding citations or sources where needed
- Clarifying ambiguous claims
- Strengthening factual support
"""

        previous_context = self._get_previous_context(result)
        if previous_context:
            critique_prompt += f"\n\n{previous_context}"

        return [
            {
                "role": "system",
                "content": "You are a Self-RAG critic that evaluates text for factual accuracy and provides retrieval-augmented feedback.",
            },
            {"role": "user", "content": critique_prompt},
        ]

    def _format_retrieval_context(self, context: Dict[str, Any]) -> str:
        """Format retrieval context for the prompt."""
        if not context:
            return ""

        sections = []

        if not context.get("retrieval_available"):
            sections.append(
                "**Note**: No retrieval backend available. The following claims should be manually verified:"
            )
            for claim in context.get("unverified_claims", []):
                sections.append(f"- {claim['claim']} (Query: {claim['query']})")
        else:
            sections.append("**Retrieval-Augmented Verification Results**:")
            for result in context.get("verified_claims", []):
                sections.append(f"\nClaim: {result['claim']}")
                sections.append(f"Verification: {result['verification']}")
                if result["sources"]:
                    sections.append(f"Sources: {', '.join(result['sources'])}")

        return "\n".join(sections) if sections else ""

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Perform critique with retrieval augmentation."""
        critique_result = await super().critique(text, result)

        # Add retrieval context to metadata
        if hasattr(self, "_current_retrieval_context"):
            critique_result.metadata["retrieval_context"] = (
                self._current_retrieval_context
            )

        return critique_result


# Example Redis Backend Implementation
class RedisRetrievalBackend:
    """Redis-based retrieval backend using Redis Search."""

    def __init__(self, redis_client, index_name: str = "sifaka_facts"):
        """Initialize with Redis client.

        Args:
            redis_client: Redis client with RediSearch module
            index_name: Name of the search index
        """
        self.redis = redis_client
        self.index_name = index_name
        self._ensure_index()

    def _ensure_index(self):
        """Ensure search index exists."""
        from redis.commands.search.field import TextField, NumericField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        try:
            # Check if index exists
            self.redis.ft(self.index_name).info()
        except:
            # Create index
            schema = [
                TextField("content", weight=1.0),
                TextField("source"),
                NumericField("timestamp"),
                TextField("category"),
            ]

            self.redis.ft(self.index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[f"{self.index_name}:"], index_type=IndexType.HASH
                ),
            )

    async def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using Redis Search."""
        from redis.commands.search.query import Query

        # Create search query
        search_query = Query(query).paging(0, top_k).with_scores()

        # Search
        results = self.redis.ft(self.index_name).search(search_query)

        # Format results
        retrieved = []
        for doc in results.docs:
            retrieved.append(
                {
                    "content": doc.content,
                    "source": doc.source,
                    "score": float(doc.score) if hasattr(doc, "score") else 1.0,
                    "id": doc.id,
                }
            )

        return retrieved

    async def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add documents to Redis."""
        import time

        pipe = self.redis.pipeline()

        for i, doc in enumerate(documents):
            doc_id = f"{self.index_name}:{int(time.time())}:{i}"
            pipe.hset(
                doc_id,
                mapping={
                    "content": doc.get("content", ""),
                    "source": doc.get("source", "Unknown"),
                    "timestamp": int(time.time()),
                    "category": doc.get("category", "general"),
                },
            )

        pipe.execute()


# Example Mem0 Backend Implementation
class Mem0RetrievalBackend:
    """Mem0-based retrieval backend for personalized memory."""

    def __init__(self, mem0_client, user_id: Optional[str] = None):
        """Initialize with Mem0 client.

        Args:
            mem0_client: Initialized Mem0 client
            user_id: Optional user ID for personalized retrieval
        """
        self.memory = mem0_client
        self.user_id = user_id or "default"

    async def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from Mem0."""
        # Search memories
        search_results = self.memory.search(query, user_id=self.user_id, limit=top_k)

        # Format results
        retrieved = []
        for memory in search_results:
            retrieved.append(
                {
                    "content": memory["text"],
                    "source": f"Memory from {memory.get('created_at', 'Unknown time')}",
                    "score": memory.get("score", 0.8),
                    "metadata": memory.get("metadata", {}),
                }
            )

        return retrieved

    async def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add documents as memories to Mem0."""
        for doc in documents:
            # Add as memory with metadata
            self.memory.add(
                doc["content"],
                user_id=self.user_id,
                metadata={
                    "source": doc.get("source", "Unknown"),
                    "category": doc.get("category", "general"),
                    "type": "fact",
                },
            )


# Example usage function
async def create_self_rag_critic_with_redis():
    """Example of creating SelfRAG critic with Redis backend."""
    import redis

    # Initialize Redis client
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

    # Create retrieval backend
    retrieval_backend = RedisRetrievalBackend(redis_client)

    # Add some facts to the corpus
    await retrieval_backend.add_documents(
        [
            {
                "content": "The speed of light is approximately 299,792,458 meters per second.",
                "source": "Physics textbook",
                "category": "science",
            },
            {
                "content": "Python was created by Guido van Rossum and first released in 1991.",
                "source": "Python documentation",
                "category": "technology",
            },
        ]
    )

    # Create critic with retrieval
    critic = SelfRAGEnhancedCritic(
        retrieval_backend=retrieval_backend, retrieval_threshold=0.6
    )

    return critic


async def create_self_rag_critic_with_mem0():
    """Example of creating SelfRAG critic with Mem0 backend."""
    from mem0 import Memory

    # Initialize Mem0
    memory = Memory()

    # Create retrieval backend
    retrieval_backend = Mem0RetrievalBackend(memory, user_id="scientist")

    # Add some memories
    await retrieval_backend.add_documents(
        [
            {
                "content": "Remember: Always cite sources when making scientific claims.",
                "source": "Research guidelines",
            },
            {
                "content": "The user prefers concise explanations with examples.",
                "source": "User preferences",
            },
        ]
    )

    # Create critic with personalized retrieval
    critic = SelfRAGEnhancedCritic(
        retrieval_backend=retrieval_backend, retrieval_threshold=0.5
    )

    return critic
