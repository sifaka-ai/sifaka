"""Self-RAG critic for Sifaka.

This module implements the Self-RAG (Self-Reflective Retrieval-Augmented Generation)
approach for text critique and improvement, which combines retrieval with self-reflection
to enhance generation quality.

Based on "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection":
https://arxiv.org/abs/2310.11511

The SelfRAGCritic uses retrieval-augmented generation with self-reflection tokens
to provide comprehensive feedback and improve text quality.
"""

import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model, Retriever
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class SelfRAGCritic(BaseCritic):
    """Critic that implements Self-RAG approach with retrieval and self-reflection.

    This critic uses the Self-RAG approach which combines retrieval-augmented
    generation with self-reflection tokens to critique and improve text quality.
    It evaluates whether retrieval is needed, assesses relevance of retrieved
    content, and provides self-reflective feedback.
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
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__(model=model, model_name=model_name, **model_kwargs)

        self.retriever = retriever
        self.use_reflection_tokens = use_reflection_tokens
        self.max_retrieved_docs = max_retrieved_docs

        # Self-RAG reflection tokens
        self.reflection_tokens = {
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
        """Perform the actual critique logic using Self-RAG approach.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Step 1: Assess if retrieval is needed
        retrieval_needed = self._assess_retrieval_need(thought)

        # Step 2: Get or use existing retrieved context
        context = self._prepare_context(thought)
        retrieved_docs = []

        # If we have a retriever and retrieval is needed, get additional context
        if self.retriever and retrieval_needed and not context.strip():
            try:
                retrieved_docs = await self._retrieve_documents_async(
                    thought.prompt + " " + thought.text
                )
                if retrieved_docs:
                    context = "\n\n".join(retrieved_docs[: self.max_retrieved_docs])
            except Exception as e:
                logger.warning(f"SelfRAGCritic: Retrieval failed: {e}")

        # Step 3: Generate detailed critique (assessments will be extracted from response)
        critique_prompt = self.critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            context=context or "No retrieved context available",
        )

        critique_response = await self.model._generate_async(
            prompt=critique_prompt,
            system_message="You are a Self-RAG critic evaluating text quality using retrieval and self-reflection.",
        )

        # Parse the critique
        issues, suggestions = self._parse_critique(critique_response)

        # Extract Self-RAG assessments from critique
        rag_assessments = self._extract_rag_assessments(critique_response)

        # Determine if improvement is needed based on utility score
        utility_score = self._extract_utility_score(rag_assessments.get("utility", "[Utility:3]"))
        needs_improvement = utility_score < 4

        logger.debug(f"SelfRAGCritic: Completed with utility score {utility_score}")

        return {
            "needs_improvement": needs_improvement,
            "message": critique_response,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": 0.8,  # Default confidence for Self-RAG
            "metadata": {
                "retrieval_needed": retrieval_needed,
                "retrieved_docs_count": len(retrieved_docs),
                "rag_assessments": rag_assessments,
                "utility_score": utility_score,
                "context_length": len(context) if context else 0,
            },
        }

    def improve(self, thought: Thought) -> str:
        """Improve text using Self-RAG approach with retrieval and reflection.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text using Self-RAG principles.

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

            # If no critique available, generate one
            if not critique:
                logger.debug("No critique found in thought, generating new critique")
                import asyncio

                try:
                    asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._perform_critique_async(thought))
                        critique_result = future.result()
                except RuntimeError:
                    critique_result = asyncio.run(self._perform_critique_async(thought))

                critique = critique_result["message"]

            # Prepare context for improvement (using mixin + retrieval)
            context = self._prepare_context(thought)

            # If we have a retriever, get additional context for improvement
            if self.retriever and not context.strip():
                try:
                    import asyncio

                    try:
                        asyncio.get_running_loop()
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run, self._retrieve_documents_async(thought.prompt)
                            )
                            retrieved_docs = future.result()
                    except RuntimeError:
                        retrieved_docs = asyncio.run(self._retrieve_documents_async(thought.prompt))

                    if retrieved_docs:
                        context = "\n\n".join(retrieved_docs[: self.max_retrieved_docs])
                except Exception as e:
                    logger.warning(f"SelfRAGCritic: Retrieval failed during improvement: {e}")

            # Create improvement prompt with context
            improve_prompt = self.improve_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                context=context or "No retrieved context available",
                critique=critique,
            )

            # Generate improved text
            improved_text = self.model.generate(
                prompt=improve_prompt,
                system_prompt="You are an expert editor using Self-RAG principles to improve text with retrieved context.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"SelfRAGCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    def _assess_retrieval_need(self, thought: Thought) -> bool:
        """Assess if retrieval is needed for the given task.

        Args:
            thought: The thought to assess.

        Returns:
            True if retrieval would be beneficial, False otherwise.
        """
        # Simple heuristics for determining retrieval need
        prompt_lower = thought.prompt.lower()
        text_lower = thought.text.lower()

        # Tasks that typically benefit from retrieval
        retrieval_indicators = [
            "fact",
            "information",
            "data",
            "research",
            "study",
            "report",
            "current",
            "recent",
            "latest",
            "update",
            "news",
            "statistics",
            "explain",
            "describe",
            "what is",
            "how does",
            "when did",
            "specific",
            "details",
            "examples",
            "evidence",
        ]

        # Tasks that typically don't need retrieval
        no_retrieval_indicators = [
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
        ]

        # Check for retrieval indicators
        retrieval_score = sum(
            1
            for indicator in retrieval_indicators
            if indicator in prompt_lower or indicator in text_lower
        )

        # Check for no-retrieval indicators
        no_retrieval_score = sum(
            1
            for indicator in no_retrieval_indicators
            if indicator in prompt_lower or indicator in text_lower
        )

        # Default to retrieval if unclear or if retrieval indicators outweigh no-retrieval
        return retrieval_score >= no_retrieval_score

    def _assess_relevance(self, thought: Thought, context: str) -> str:
        """Assess relevance of retrieved context to the task.

        Args:
            thought: The thought being evaluated.
            context: The retrieved context.

        Returns:
            Relevance assessment token.
        """
        if not context or not context.strip():
            return "[Irrelevant]"

        # Simple keyword overlap assessment
        prompt_words = set(thought.prompt.lower().split())
        context_words = set(context.lower().split())

        overlap = len(prompt_words.intersection(context_words))
        overlap_ratio = overlap / len(prompt_words) if prompt_words else 0

        if overlap_ratio > 0.3:
            return "[Relevant]"
        elif overlap_ratio > 0.1:
            return "[Partially Relevant]"
        else:
            return "[Irrelevant]"

    def _assess_support(self, thought: Thought, context: str) -> str:
        """Assess how well the text uses the retrieved context.

        Args:
            thought: The thought being evaluated.
            context: The retrieved context.

        Returns:
            Support assessment token.
        """
        if not context or not context.strip():
            return "[No Support]"

        # Simple assessment based on content overlap
        text_words = set(thought.text.lower().split())
        context_words = set(context.lower().split())

        overlap = len(text_words.intersection(context_words))
        overlap_ratio = overlap / len(text_words) if text_words else 0

        if overlap_ratio > 0.2:
            return "[Fully Supported]"
        elif overlap_ratio > 0.05:
            return "[Partially Supported]"
        else:
            return "[No Support]"

    def _assess_utility(self, thought: Thought, context: str) -> str:
        """Assess overall utility of the text.

        Args:
            thought: The thought being evaluated.
            context: The retrieved context.

        Returns:
            Utility assessment token.
        """
        # Simple utility assessment based on text length and context usage
        text_length = len(thought.text.split())

        # Base score on text length
        if text_length < 10:
            base_score = 2
        elif text_length < 50:
            base_score = 3
        elif text_length < 100:
            base_score = 4
        else:
            base_score = 4

        # Adjust based on context usage
        if context and context.strip():
            support = self._assess_support(thought, context)
            if support == "[Fully Supported]":
                base_score = min(5, base_score + 1)
            elif support == "[No Support]":
                base_score = max(1, base_score - 1)

        return f"[Utility:{base_score}]"

    async def _retrieve_documents_async(self, query: str) -> List[str]:
        """Retrieve documents for the given query.

        Args:
            query: The query to search for.

        Returns:
            List of retrieved document texts.
        """
        if not self.retriever:
            return []

        try:
            # Use retriever to get documents
            results = await self.retriever.retrieve_async(query, limit=self.max_retrieved_docs)
            return [doc.get("content", str(doc)) for doc in results]
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

        # Simple parsing logic
        in_issues = False
        in_suggestions = False

        for line in critique.split("\n"):
            line = line.strip()
            if line.lower().startswith("issues:"):
                in_issues = True
                in_suggestions = False
                continue
            elif line.lower().startswith("suggestions:"):
                in_issues = False
                in_suggestions = True
                continue
            elif line.lower().startswith(("self-rag assessment:", "overall assessment:")):
                in_issues = False
                in_suggestions = False
                continue
            elif not line or line.startswith("#"):
                continue

            if in_issues and line.startswith("-"):
                issues.append(line[1:].strip())
            elif in_suggestions and line.startswith("-"):
                suggestions.append(line[1:].strip())

        # If no structured format found, extract from general content
        if not issues and not suggestions:
            critique_lower = critique.lower()
            if any(word in critique_lower for word in ["issue", "problem", "error", "poor"]):
                issues.append("Issues identified in Self-RAG assessment")
            if any(word in critique_lower for word in ["improve", "suggest", "better", "should"]):
                suggestions.append("See Self-RAG assessment for improvement suggestions")

        return issues, suggestions

    def _extract_rag_assessments(self, critique: str) -> Dict[str, str]:
        """Extract Self-RAG assessment tokens from critique.

        Args:
            critique: The critique text to parse.

        Returns:
            Dictionary of assessment types to tokens.
        """
        assessments = {}

        # Look for Self-RAG assessment section
        lines = critique.split("\n")
        in_assessment = False

        for line in lines:
            line = line.strip()
            if line.lower().startswith("self-rag assessment:"):
                in_assessment = True
                continue
            elif line.lower().startswith("overall assessment:"):
                in_assessment = False
                continue

            if in_assessment and ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip("- ").lower()
                    value = parts[1].strip()
                    assessments[key] = value

        # Set defaults if not found
        assessments.setdefault("retrieval", "[No Retrieve]")
        assessments.setdefault("relevance", "[Partially Relevant]")
        assessments.setdefault("support", "[Partially Supported]")
        assessments.setdefault("utility", "[Utility:3]")

        return assessments

    def _extract_utility_score(self, utility_token: str) -> int:
        """Extract numerical utility score from utility token.

        Args:
            utility_token: The utility assessment token.

        Returns:
            Utility score (1-5).
        """
        import re

        # Look for [Utility:X] pattern
        match = re.search(r"\[Utility:(\d+)\]", utility_token)
        if match:
            try:
                score = int(match.group(1))
                return max(1, min(5, score))  # Clamp to [1, 5]
            except ValueError:
                pass

        # Default to middle score
        return 3
