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

from sifaka.core.interfaces import Model, Retriever
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic

from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger
from sifaka.validators.validation_context import create_validation_context

logger = get_logger(__name__)


class SelfRAGCritic(BaseCritic):
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
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        retriever: Optional[Retriever] = None,
        use_reflection_tokens: bool = True,
        max_retrieved_docs: int = 5,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        reflection_tokens: Optional[Dict[str, List[str]]] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Self-RAG critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            retriever: The retriever to use for finding relevant documents.
            use_reflection_tokens: Whether to use Self-RAG reflection tokens.
            max_retrieved_docs: Maximum number of documents to retrieve.
            critique_prompt_template: Template for the critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            reflection_tokens: Custom reflection tokens. If None, uses default Self-RAG tokens.
                Expected format: {
                    "retrieve": ["[Retrieve]", "[No Retrieve]"],
                    "relevance": ["[Relevant]", "[Partially Relevant]", "[Irrelevant]"],
                    "support": ["[Fully Supported]", "[Partially Supported]", "[No Support]"],
                    "utility": ["[Utility:5]", "[Utility:4]", "[Utility:3]", "[Utility:2]", "[Utility:1]"]
                }
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__(model=model, model_name=model_name, **model_kwargs)

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

        # Set up prompt templates
        self.critique_prompt_template = critique_prompt_template or (
            "You are a Self-RAG critic that evaluates text using retrieval and self-reflection.\n\n"
            "Original task: {prompt}\n\n"
            "Text to critique:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Please evaluate the text using Self-RAG principles:\n\n"
            "1. RETRIEVAL ASSESSMENT: Was retrieval needed for this task?\n"
            "   - [Retrieve] if external knowledge would improve the response\n"
            "   - [No Retrieve] if the task can be answered without external knowledge\n\n"
            "2. RELEVANCE ASSESSMENT: How relevant is the retrieved context?\n"
            "   - [Relevant] if context directly addresses the task\n"
            "   - [Partially Relevant] if context is somewhat related\n"
            "   - [Irrelevant] if context doesn't help with the task\n\n"
            "3. SUPPORT ASSESSMENT: How well does the text use the retrieved context?\n"
            "   - [Fully Supported] if text properly incorporates context\n"
            "   - [Partially Supported] if text uses some context\n"
            "   - [No Support] if text ignores available context\n\n"
            "4. UTILITY ASSESSMENT: Rate the overall utility (1-5):\n"
            "   - [Utility:5] Excellent, comprehensive response\n"
            "   - [Utility:4] Good response with minor issues\n"
            "   - [Utility:3] Adequate response\n"
            "   - [Utility:2] Poor response with major issues\n"
            "   - [Utility:1] Very poor response\n\n"
            "Format your response as:\n"
            "Issues:\n- [List specific issues here]\n\n"
            "Suggestions:\n- [List specific suggestions here]\n\n"
            "Self-RAG Assessment:\n"
            "- Retrieval: [Your assessment]\n"
            "- Relevance: [Your assessment]\n"
            "- Support: [Your assessment]\n"
            "- Utility: [Your assessment]\n\n"
            "Overall Assessment: [Brief summary]"
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Improve the following text using Self-RAG principles and retrieved context.\n\n"
            "Original task: {prompt}\n\n"
            "Current text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Self-RAG critique:\n{critique}\n\n"
            "Please provide an improved version that:\n"
            "1. Better incorporates relevant information from the retrieved context\n"
            "2. Addresses the issues identified in the Self-RAG assessment\n"
            "3. Follows the suggestions for improvement\n"
            "4. Maintains factual accuracy and relevance to the original task\n"
            "5. Uses Self-RAG reflection principles for better quality\n\n"
            "Improved text:"
        )

    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform Self-RAG inspired critique with smart retrieval and reflection-style assessment (async).

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
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

        # Step 3: Generate Self-RAG style critique (async)
        critique_prompt = self._build_critique_prompt(thought, context, retrieval_needed)

        critique_response = await self.model._generate_async(
            prompt=critique_prompt,
            system_prompt="You are a Self-RAG inspired critic that provides structured feedback using reflection tokens.",
        )

        # Step 4: Parse the critique and extract assessments
        issues, suggestions = self._parse_critique(critique_response)
        rag_assessments = self._extract_rag_assessments(critique_response)

        # Determine utility score and improvement need
        utility_score = self._extract_utility_score(rag_assessments.get("utility", "3"))
        needs_improvement = utility_score < 4

        # Always provide feedback if we have issues or suggestions, even if utility score is high
        if not needs_improvement and (issues or suggestions):
            needs_improvement = True
            logger.debug(
                f"SelfRAGCritic: Forcing improvement due to issues/suggestions despite high utility score ({utility_score})"
            )

        logger.debug(f"SelfRAGCritic: Async completed with utility score {utility_score}")

        return {
            "needs_improvement": needs_improvement,
            "message": critique_response,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": 0.8,
            "metadata": {
                "retrieval_needed": retrieval_needed,
                "retrieved_docs_count": len(retrieved_docs),
                "rag_assessments": rag_assessments,
                "utility_score": utility_score,
                "context_length": len(context),
                "retriever_used": retriever_used,
            },
        }

    async def improve_async(self, thought: Thought) -> str:
        """Improve text using Self-RAG approach with retrieval and reflection asynchronously.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text using Self-RAG principles.

        Raises:
            ImproverError: If the improvement fails.
        """
        # Use the enhanced method with validation context from thought
        validation_context = create_validation_context(getattr(thought, "validation_results", None))
        return await self.improve_with_validation_context_async(thought, validation_context)

    async def improve_with_validation_context_async(
        self, thought: Thought, validation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Improve text with validation context awareness asynchronously.

        Args:
            thought: The Thought container with the text to improve and critique.
            validation_context: Optional validation context for constraint awareness.

        Returns:
            The improved text that prioritizes validation constraints.

        Raises:
            ImproverError: If the improvement fails.
        """
        start_time = time.time()

        with critic_context(
            critic_name="SelfRAGCritic",
            operation="improve",
            message_prefix="Failed to improve text with Self-RAG",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="SelfRAGCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            # Get critique from thought
            critique = ""
            if thought.critic_feedback:
                for feedback in thought.critic_feedback:
                    if feedback.critic_name == "SelfRAGCritic":
                        critique = feedback.feedback
                        break

            # If no critique available, generate one using async method
            if not critique:
                logger.debug(
                    "SelfRAGCritic: No critique found in thought, generating new critique for improvement"
                )
                critique_result = await self._perform_critique_async(thought)
                critique = critique_result["message"]
                logger.debug(
                    "SelfRAGCritic: Generated new critique for improvement (not adding to thought feedback)"
                )
                # NOTE: This critique is only used for improvement, not added to thought.critic_feedback
                # to prevent duplicate entries in the visualization

            # Prepare context for improvement (using mixin + retrieval)
            context = self._prepare_context(thought)

            # If we have a retriever or chain context, get additional context for improvement
            if not context.strip():
                # Try critic-specific retriever first
                if self.retriever:
                    try:
                        retrieved_docs = self._retrieve_documents(thought.prompt, self.retriever)
                        if retrieved_docs:
                            context = "\n\n".join(retrieved_docs[: self.max_retrieved_docs])
                            logger.debug(
                                "SelfRAGCritic: Using critic-specific retriever for improvement"
                            )
                    except Exception as e:
                        logger.warning(
                            f"SelfRAGCritic: Critic-specific retrieval failed during improvement: {e}"
                        )

                # Fallback to chain-level context if still no context
                if not context.strip():
                    chain_context = self._prepare_context(thought)
                    if chain_context and chain_context != "No retrieved context available.":
                        context = chain_context
                        logger.debug("SelfRAGCritic: Using chain-level context for improvement")

            # Create improvement prompt with validation awareness
            if validation_context:
                # Use enhanced prompt with validation awareness
                improve_prompt = self._create_enhanced_improvement_prompt(
                    prompt=thought.prompt,
                    text=thought.text,
                    critique=critique,
                    context=context or "No retrieved context available",
                    validation_context=validation_context,
                    critic_suggestions=[],  # SelfRAGCritic doesn't have structured suggestions
                )
            else:
                # Use original prompt template
                improve_prompt = self.improve_prompt_template.format(
                    prompt=thought.prompt,
                    text=thought.text,
                    context=context or "No retrieved context available",
                    critique=critique,
                )

            # Generate improved text (async only)
            improved_text = await self.model._generate_async(
                prompt=improve_prompt,
                system_prompt="You are an expert editor using Self-RAG principles to improve text with retrieved context.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"SelfRAGCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

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

    def _build_critique_prompt(self, thought: Thought, context: str, retrieval_needed: bool) -> str:
        """Build the critique prompt with Self-RAG style structure.

        Args:
            thought: The thought being critiqued.
            context: Retrieved context (if any).
            retrieval_needed: Whether retrieval was deemed necessary.

        Returns:
            Formatted critique prompt.
        """
        retrieval_token = self.reflection_tokens["retrieve"][0 if retrieval_needed else 1]

        if self.use_reflection_tokens:
            prompt = f"""You are a Self-RAG critic. Evaluate the text using reflection tokens.

Original task: {thought.prompt}

Text to evaluate:
{thought.text}

Retrieved context:
{context or "No context retrieved"}

Provide your assessment using these reflection tokens:
- Retrieval: {retrieval_token}
- Relevance: [Relevant], [Partially Relevant], or [Irrelevant]
- Support: [Fully Supported], [Partially Supported], or [No Support]
- Utility: [Utility:1] through [Utility:5]

Format your response as:

Issues:
- [List specific issues if any]

Suggestions:
- [List specific improvement suggestions if any]

Self-RAG Assessment:
- Retrieval: {retrieval_token}
- Relevance: [Your assessment]
- Support: [Your assessment]
- Utility: [Your assessment]"""
        else:
            prompt = f"""Evaluate this text for quality and factual accuracy.

Original task: {thought.prompt}

Text to evaluate:
{thought.text}

Retrieved context:
{context or "No context retrieved"}

Provide structured feedback:

Issues:
- [List specific issues if any]

Suggestions:
- [List specific improvement suggestions if any]

Assessment:
- Retrieval needed: {"yes" if retrieval_needed else "no"}
- Context relevance: [relevant/partially_relevant/irrelevant]
- Factual support: [supported/partially_supported/unsupported]
- Overall quality: [1-5 scale]"""

        return prompt

    def _retrieve_documents(self, query: str, retriever: Optional[Retriever] = None) -> List[str]:
        """Retrieve documents for the given query.

        Args:
            query: The query to search for.
            retriever: Optional specific retriever to use. If None, uses self.retriever.

        Returns:
            List of retrieved document texts.
        """
        target_retriever = retriever or self.retriever
        if not target_retriever:
            return []

        try:
            # Use retriever to get documents
            results = target_retriever.retrieve(query)
            return results[: self.max_retrieved_docs] if results else []
        except Exception as e:
            logger.error(f"SelfRAGCritic: Retrieval error: {e}")
            return []

    def _parse_critique(self, critique: str) -> tuple[List[str], List[str]]:
        """Parse critique text to extract issues and suggestions.

        Args:
            critique: The critique text to parse.

        Returns:
            A tuple of (issues, suggestions) lists.
        """
        issues = []
        suggestions = []

        lines = critique.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if line.lower().startswith("issues:"):
                current_section = "issues"
                continue
            elif line.lower().startswith("suggestions:"):
                current_section = "suggestions"
                continue
            elif line.lower().startswith(("assessment:", "self-rag assessment:")):
                current_section = None
                continue

            # Extract items from current section
            if line.startswith("-") and current_section:
                item = line[1:].strip()
                if current_section == "issues":
                    issues.append(item)
                elif current_section == "suggestions":
                    suggestions.append(item)

        return issues, suggestions

    def _extract_rag_assessments(self, critique: str) -> Dict[str, str]:
        """Extract Self-RAG assessment tokens from critique.

        Args:
            critique: The critique text to parse.

        Returns:
            Dictionary of assessment types to tokens.
        """
        assessments = {}
        lines = critique.split("\n")

        # Look for assessment section
        in_assessment_section = False
        for line in lines:
            line = line.strip()

            # Start of assessment section
            if line.lower().startswith(("self-rag assessment:", "assessment:")):
                in_assessment_section = True
                continue

            # Parse assessment lines
            if in_assessment_section and ":" in line and line.startswith("-"):
                parts = line[1:].split(":", 1)  # Remove leading dash
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    assessments[key] = value

        # Set reasonable defaults
        if self.use_reflection_tokens:
            assessments.setdefault("retrieval", self.reflection_tokens["retrieve"][1])
            assessments.setdefault("relevance", self.reflection_tokens["relevance"][1])
            assessments.setdefault("support", self.reflection_tokens["support"][1])
            assessments.setdefault("utility", self.reflection_tokens["utility"][2])
        else:
            assessments.setdefault("retrieval", "no")
            assessments.setdefault("relevance", "partially_relevant")
            assessments.setdefault("support", "partially_supported")
            assessments.setdefault("utility", "3")

        return assessments

    def _extract_utility_score(self, utility_token: str) -> int:
        """Extract numerical utility score from utility token.

        Args:
            utility_token: The utility assessment token.

        Returns:
            Utility score (1-5).
        """
        import re

        # Try to extract number from various formats
        # [Utility:3], "3", "utility:4", etc.
        patterns = [
            r"\[Utility:(\d+)\]",  # [Utility:3]
            r"utility:?\s*(\d+)",  # utility:3 or utility 3
            r"^(\d+)$",  # Just a number
        ]

        for pattern in patterns:
            match = re.search(pattern, utility_token, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    return max(1, min(5, score))  # Clamp to [1, 5]
                except ValueError:
                    continue

        # Default to middle score
        return 3
