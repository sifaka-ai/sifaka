"""Self-RAG inspired critic for Sifaka.

This module implements a Self-RAG inspired approach for text critique and improvement,
which combines smart retrieval decisions with reflection-style quality assessment.

Based on "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection":
https://arxiv.org/abs/2310.11511

@misc{asai2023selfraglearningretrievegenerate,
      title={Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
      author={Akari Asai and Zeqiu Wu and Yizhong Wang and Avirup Sil and Hannaneh Hajishirzi},
      year={2023},
      eprint={2310.11511},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.11511},
}

The SelfRAGCritic implements Self-RAG inspired concepts:
1. Smart retrieval decisions based on task analysis
2. Reflection-style tokens for quality assessment
3. Retrieval relevance and factual support evaluation
4. Utility-based scoring for overall text quality

IMPORTANT IMPLEMENTATION CAVEAT:
This is a simplified, production-focused implementation that is approximately 30-40%
faithful to the original Self-RAG paper. Key differences from the original:

WHAT THIS IMPLEMENTATION DOES:
- ✅ Uses Self-RAG reflection token format ([Retrieve], [Relevant], etc.)
- ✅ Makes smart retrieval decisions based on task analysis
- ✅ Provides structured quality assessment with utility scoring
- ✅ Works with any existing language model (no training required)
- ✅ Offers practical production value for RAG applications

WHAT THE ORIGINAL SELF-RAG DOES (that we don't):
- ❌ Requires fine-tuning models with reflection tokens during training
- ❌ Generates reflection tokens inline during text generation (not post-hoc)
- ❌ Uses segment-level beam search with reflection token probabilities
- ❌ Employs separate critic model training on GPT-4 generated data
- ❌ Processes text in segments with retrieval decisions at each step

This implementation prioritizes practical deployment and production value over
research fidelity. It captures the useful aspects of Self-RAG (smart retrieval,
structured assessment) without requiring custom model training or complex inference.
"""

import time
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent

from sifaka.core.interfaces import Retriever
from sifaka.core.thought import Thought
from sifaka.critics.base_pydantic import PydanticAICritic
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class SelfRAGCritic(PydanticAICritic):
    """Critic that implements Self-RAG inspired approach with retrieval and self-reflection and validation awareness.

    This critic uses a Self-RAG inspired approach which combines retrieval-augmented
    generation with self-reflection tokens to critique and improve text quality.
    It evaluates whether retrieval is needed, assesses relevance of retrieved
    content, and provides self-reflective feedback.

    IMPLEMENTATION NOTE: This is a simplified, production-focused implementation
    that captures the practical value of Self-RAG without requiring model training.
    It is approximately 30-40% faithful to the original Self-RAG paper, prioritizing
    ease of deployment and practical utility over research fidelity.

    Enhanced with validation context awareness to prioritize validation constraints
    over conflicting Self-RAG suggestions.

    See module docstring for detailed comparison with the original Self-RAG approach.
    """

    def __init__(
        self,
        model_name: str,
        retriever: Optional[Retriever] = None,
        use_reflection_tokens: bool = True,
        max_retrieved_docs: int = 5,
        reflection_tokens: Optional[Dict[str, List[str]]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the Self-RAG critic.

        Args:
            model_name: The model name for the PydanticAI agent (e.g., "openai:gpt-4")
            retriever: The retriever to use for finding relevant documents.
            use_reflection_tokens: Whether to use Self-RAG reflection tokens.
            max_retrieved_docs: Maximum number of documents to retrieve.
            reflection_tokens: Custom reflection tokens. If None, uses default Self-RAG tokens.
                Expected format: {
                    "retrieve": ["[Retrieve]", "[No Retrieve]"],
                    "relevance": ["[Relevant]", "[Partially Relevant]", "[Irrelevant]"],
                    "support": ["[Fully Supported]", "[Partially Supported]", "[No Support]"],
                    "utility": ["[Utility:5]", "[Utility:4]", "[Utility:3]", "[Utility:2]", "[Utility:1]"]
                }
            **agent_kwargs: Additional arguments passed to the PydanticAI agent.
        """
        # Initialize parent with system prompt
        super().__init__(model_name=model_name, **agent_kwargs)

        self.retriever = retriever
        self.use_reflection_tokens = use_reflection_tokens
        self.max_retrieved_docs = max_retrieved_docs

        # Self-RAG reflection tokens (customizable)
        self.reflection_tokens = reflection_tokens or {
            "retrieve": ["[Retrieve]", "[No Retrieve]"],
            "relevance": ["[Relevant]", "[Partially Relevant]", "[Irrelevant]"],
            "support": ["[Fully Supported]", "[Partially Supported]", "[No Support]"],
            "utility": ["[Utility:5]", "[Utility:4]", "[Utility:3]", "[Utility:2]", "[Utility:1]"],
        }

        logger.info(
            f"Initialized SelfRAGCritic with retriever={retriever is not None}, "
            f"reflection_tokens={use_reflection_tokens}, max_docs={max_retrieved_docs}"
        )

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for Self-RAG evaluation."""
        return """You are a Self-RAG inspired critic that evaluates text using retrieval and self-reflection. Your role is to provide structured feedback using Self-RAG principles.

You must return a CritiqueFeedback object with these REQUIRED fields:
- message: A clear summary of your Self-RAG evaluation (string)
- needs_improvement: Whether the text needs improvement based on Self-RAG assessment (boolean)
- confidence: ConfidenceScore with overall confidence (object with 'overall' field as float 0.0-1.0)
- critic_name: Set this to "SelfRAGCritic" (string)

And these OPTIONAL fields (can be empty lists or null):
- violations: List of ViolationReport objects for identified issues
- suggestions: List of ImprovementSuggestion objects for addressing issues
- processing_time_ms: Time taken in milliseconds (can be null)
- critic_version: Version string (can be null)
- metadata: Additional metadata dictionary (can be empty)

IMPORTANT: Always provide the required fields. For confidence, use a simple object like {"overall": 0.8} where the number is between 0.0 and 1.0.

Focus on Self-RAG principles:
1. RETRIEVAL ASSESSMENT: Was retrieval needed for this task?
2. RELEVANCE ASSESSMENT: How relevant is the retrieved context?
3. SUPPORT ASSESSMENT: How well does the text use the retrieved context?
4. UTILITY ASSESSMENT: Rate the overall utility (1-5)

Use Self-RAG reflection tokens and provide structured, actionable feedback."""

    async def _create_critique_prompt(self, thought: Thought) -> str:
        """Create the critique prompt for Self-RAG evaluation.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            The formatted critique prompt.
        """
        # Step 1: Determine if retrieval would be beneficial
        retrieval_needed = self._should_retrieve(thought)

        # Step 2: Retrieve documents if needed using multiple retriever sources
        context = ""
        retrieved_docs = []
        retriever_used = None

        if retrieval_needed:
            # Try critic-specific retriever first (highest precedence)
            if self.retriever:
                try:
                    query = f"{thought.prompt} {thought.text or ''}".strip()
                    retrieved_docs = self._retrieve_documents(query, self.retriever)
                    retriever_used = "critic_specific"
                    if retrieved_docs:
                        context = "\n\n".join(retrieved_docs[: self.max_retrieved_docs])
                        logger.debug(
                            f"SelfRAGCritic: Retrieved {len(retrieved_docs)} documents using critic-specific retriever"
                        )
                except Exception as e:
                    logger.warning(f"SelfRAGCritic: Critic-specific retrieval failed: {e}")

            # Fallback to chain-level context if no critic-specific retriever or it failed
            if not retrieved_docs:
                chain_context = self._prepare_context(thought)
                if chain_context and chain_context != "No retrieved context available.":
                    context = chain_context
                    retriever_used = "chain_level"
                    logger.debug("SelfRAGCritic: Using chain-level retrieved context")

        # Get validation context if available
        validation_context = self._get_validation_context_dict(thought)
        validation_text = ""
        if validation_context:
            validation_text = f"\n\nValidation Context:\n{validation_context}"

        # Build Self-RAG style prompt
        retrieval_token = self.reflection_tokens["retrieve"][0 if retrieval_needed else 1]

        if self.use_reflection_tokens:
            return f"""Evaluate the following text using Self-RAG principles and reflection tokens.

Original task: {thought.prompt}

Text to evaluate:
{thought.text}

Retrieved context:
{context or "No context retrieved"}
{validation_text}

Provide your assessment using these Self-RAG reflection tokens:
- Retrieval: {retrieval_token}
- Relevance: [Relevant], [Partially Relevant], or [Irrelevant]
- Support: [Fully Supported], [Partially Supported], or [No Support]
- Utility: [Utility:1] through [Utility:5]

Self-RAG Assessment Guidelines:
1. RETRIEVAL: Was retrieval needed for this task? ({retrieval_token})
2. RELEVANCE: How relevant is the retrieved context to the task?
3. SUPPORT: How well does the text use the available context?
4. UTILITY: Rate overall utility from 1 (poor) to 5 (excellent)

Please provide structured feedback with specific issues and suggestions for improvement."""
        else:
            return f"""Evaluate this text for quality and factual accuracy using Self-RAG principles.

Original task: {thought.prompt}

Text to evaluate:
{thought.text}

Retrieved context:
{context or "No context retrieved"}
{validation_text}

Please provide structured feedback focusing on:
1. How well the text addresses the original task
2. Use of available context and factual accuracy
3. Overall quality and utility
4. Specific improvements needed

Retrieval Assessment: {retrieval_token}"""

    def _should_retrieve(self, thought: Thought) -> bool:
        """Determine if retrieval would be beneficial for this task.

        Args:
            thought: The thought to assess.

        Returns:
            True if retrieval would be beneficial, False otherwise.
        """
        prompt_lower = thought.prompt.lower()
        text_lower = (thought.text or "").lower()
        combined = f"{prompt_lower} {text_lower}"

        # Strong indicators that retrieval would help
        factual_indicators = [
            "fact",
            "data",
            "statistic",
            "research",
            "study",
            "evidence",
            "current",
            "recent",
            "latest",
            "when",
            "where",
            "who",
            "what",
            "explain",
            "define",
            "compare",
            "how many",
            "specific",
        ]

        # Strong indicators that retrieval is not needed
        creative_indicators = [
            "opinion",
            "creative",
            "story",
            "poem",
            "fiction",
            "imagine",
            "personal",
            "feeling",
            "think",
            "believe",
            "prefer",
            "write a",
        ]

        factual_score = sum(1 for indicator in factual_indicators if indicator in combined)
        creative_score = sum(1 for indicator in creative_indicators if indicator in combined)

        # Default to retrieval if unclear, but prefer no retrieval for clearly creative tasks
        if creative_score > factual_score:
            return False
        return factual_score > 0 or creative_score == 0

    def _retrieve_documents(self, query: str, retriever: Retriever) -> List[str]:
        """Retrieve documents using the provided retriever.

        Args:
            query: The query to search for.
            retriever: The retriever to use.

        Returns:
            List of retrieved document texts.
        """
        try:
            # Use the retriever's retrieve method
            results = retriever.retrieve(query, limit=self.max_retrieved_docs)
            return [doc.content for doc in results if hasattr(doc, "content")]
        except Exception as e:
            logger.warning(f"SelfRAGCritic: Document retrieval failed: {e}")
            return []
