"""Self-RAG critic for Sifaka.

This module implements the Self-RAG (Self-Reflective Retrieval-Augmented Generation)
approach for text critique and improvement, which combines retrieval with self-reflection
to enhance generation quality.

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

The SelfRAGCritic implements key Self-RAG concepts:
1. Adaptive retrieval based on task requirements
2. Self-reflection tokens for quality assessment
3. Retrieval relevance and support evaluation
4. Utility-based scoring for text quality
5. Learning from retrieval effectiveness patterns (enhanced)
6. Adaptive retrieval decisions based on past success/failure (enhanced)

Note: This implementation captures core Self-RAG principles with enhanced
learning capabilities through integration with the Sifaka thoughts system.
The critic learns when retrieval helps vs. hurts to make smarter decisions.
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
        # Extract learning context from thought for enhanced retrieval decisions
        learning_context = self._extract_retrieval_learning_context(thought)

        # Step 1: Assess if retrieval is needed (enhanced with learning)
        retrieval_needed = self._assess_retrieval_need_with_learning(thought, learning_context)

        # Step 2: Get or use existing retrieved context
        context = self._prepare_context(thought)
        retrieved_docs = []

        # If we have a retriever and retrieval is needed, get additional context
        retrieval_attempted = False
        retrieval_success = False
        if self.retriever and retrieval_needed and not context.strip():
            try:
                retrieval_attempted = True
                retrieved_docs = await self._retrieve_documents_async(
                    thought.prompt + " " + (thought.text or "")
                )
                if retrieved_docs:
                    context = "\n\n".join(retrieved_docs[: self.max_retrieved_docs])
                    retrieval_success = True
                    logger.debug(
                        f"SelfRAGCritic: Successfully retrieved {len(retrieved_docs)} documents"
                    )
            except Exception as e:
                logger.warning(f"SelfRAGCritic: Retrieval failed: {e}")
                retrieval_success = False

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

        # Store retrieval learning outcomes for future decisions
        self._store_retrieval_outcomes(
            thought,
            learning_context,
            retrieval_needed,
            retrieval_attempted,
            retrieval_success,
            utility_score,
            rag_assessments,
        )

        logger.debug(
            f"SelfRAGCritic: Completed with utility score {utility_score} and learning integration"
        )

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
                "learning_applied": bool(learning_context.get("patterns")),
                "retrieval_attempted": retrieval_attempted,
                "retrieval_success": retrieval_success,
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
                        future = executor.submit(
                            lambda: asyncio.run(self._perform_critique_async(thought))
                        )
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

                        # Run async retrieval in thread pool
                        async def _retrieve_docs() -> List[str]:
                            return await self._retrieve_documents_async(thought.prompt)

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, _retrieve_docs())  # type: ignore[arg-type]
                            retrieved_docs: List[str] = future.result()  # type: ignore[assignment]
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
        text_lower = (thought.text or "").lower()

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
        text_words = set((thought.text or "").lower().split())
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
        text_length = len((thought.text or "").split())

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
            # Use retriever to get documents (sync method wrapped in async)
            import asyncio

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.retriever.retrieve, query)
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

    def _extract_retrieval_learning_context(self, thought: Thought) -> Dict[str, Any]:
        """Extract learning context from thought for enhanced retrieval decisions.

        Args:
            thought: The Thought to extract learning context from.

        Returns:
            Dictionary with retrieval learning context.
        """
        learning_context = {
            "retrieval_sessions": 0,
            "effective_retrievals": [],
            "ineffective_retrievals": [],
            "task_retrieval_patterns": {},
            "task_type": self._classify_retrieval_task_type(thought.prompt),
        }

        # Extract from thought metadata
        if thought.metadata:
            self_rag_data = thought.metadata.get("self_rag_memory", {})
            if self_rag_data:
                learning_context["retrieval_sessions"] = len(self_rag_data.get("sessions", []))
                learning_context["effective_retrievals"] = self_rag_data.get(
                    "effective_retrievals", []
                )[
                    -10:
                ]  # Last 10
                learning_context["ineffective_retrievals"] = self_rag_data.get(
                    "ineffective_retrievals", []
                )[
                    -10:
                ]  # Last 10
                learning_context["task_retrieval_patterns"] = self_rag_data.get(
                    "task_retrieval_patterns", {}
                )

        # Extract from thought history
        if thought.history:
            learning_context["previous_attempts"] = len(thought.history)

        # Extract from critic feedback history
        if thought.critic_feedback:
            self_rag_feedback = [
                f for f in thought.critic_feedback if f.critic_name == "SelfRAGCritic"
            ]
            if self_rag_feedback:
                learning_context["previous_feedback_count"] = len(self_rag_feedback)
                # Analyze retrieval effectiveness from previous feedback
                for feedback in self_rag_feedback[-3:]:  # Last 3 feedback instances
                    if feedback.metadata:
                        retrieval_needed = feedback.metadata.get("retrieval_needed", False)
                        utility_score = feedback.metadata.get("utility_score", 3)
                        retrieval_success = feedback.metadata.get("retrieval_success", False)

                        if retrieval_needed and retrieval_success and utility_score >= 4:
                            learning_context["effective_retrievals"].append(
                                {
                                    "task_type": learning_context["task_type"],
                                    "utility_score": utility_score,
                                }
                            )
                        elif retrieval_needed and not retrieval_success:
                            learning_context["ineffective_retrievals"].append(
                                {
                                    "task_type": learning_context["task_type"],
                                    "reason": "retrieval_failed",
                                }
                            )

        return learning_context

    def _classify_retrieval_task_type(self, prompt: str) -> str:
        """Classify the task type for retrieval learning purposes.

        Args:
            prompt: The task prompt to classify.

        Returns:
            String representing the retrieval task type.
        """
        prompt_lower = prompt.lower()

        # Fact-heavy tasks that typically benefit from retrieval
        if any(
            word in prompt_lower
            for word in ["fact", "data", "statistic", "research", "study", "evidence"]
        ):
            return "factual"
        elif any(
            word in prompt_lower for word in ["current", "recent", "latest", "news", "update"]
        ):
            return "current_events"
        elif any(
            word in prompt_lower
            for word in ["technical", "specification", "documentation", "manual"]
        ):
            return "technical"
        elif any(word in prompt_lower for word in ["history", "historical", "past", "timeline"]):
            return "historical"
        elif any(
            word in prompt_lower for word in ["compare", "contrast", "versus", "vs", "difference"]
        ):
            return "comparative"
        elif any(
            word in prompt_lower for word in ["opinion", "creative", "imagine", "story", "poem"]
        ):
            return "creative"
        else:
            return "general"

    def _assess_retrieval_need_with_learning(
        self, thought: Thought, learning_context: Dict[str, Any]
    ) -> bool:
        """Assess if retrieval is needed using learning from past patterns.

        Args:
            thought: The Thought to assess.
            learning_context: Learning context from past retrieval attempts.

        Returns:
            Boolean indicating if retrieval is needed.
        """
        # Start with base assessment
        base_assessment = self._assess_retrieval_need(thought)

        # Apply learning adjustments
        task_type = learning_context.get("task_type", "general")

        # Check task-specific patterns
        task_patterns = learning_context.get("task_retrieval_patterns", {}).get(task_type, {})
        if task_patterns:
            success_rate = task_patterns.get("success_rate", 0.5)
            avg_utility_improvement = task_patterns.get("avg_utility_improvement", 0.0)

            # If retrieval historically helps for this task type, be more likely to retrieve
            if success_rate > 0.7 and avg_utility_improvement > 0.5:
                logger.debug(
                    f"SelfRAGCritic: Learning suggests retrieval beneficial for {task_type} tasks"
                )
                return True
            # If retrieval historically doesn't help, be less likely to retrieve
            elif success_rate < 0.3 or avg_utility_improvement < -0.2:
                logger.debug(
                    f"SelfRAGCritic: Learning suggests retrieval not beneficial for {task_type} tasks"
                )
                return False

        # Check recent effectiveness patterns
        effective_count = len(learning_context.get("effective_retrievals", []))
        ineffective_count = len(learning_context.get("ineffective_retrievals", []))

        if effective_count + ineffective_count > 5:  # Enough data to make decisions
            effectiveness_ratio = effective_count / (effective_count + ineffective_count)
            if effectiveness_ratio > 0.8:
                logger.debug(
                    "SelfRAGCritic: Recent retrieval history very positive, favoring retrieval"
                )
                return True
            elif effectiveness_ratio < 0.2:
                logger.debug("SelfRAGCritic: Recent retrieval history poor, avoiding retrieval")
                return False

        # Fall back to base assessment
        return base_assessment

    def _store_retrieval_outcomes(
        self,
        thought: Thought,
        learning_context: Dict[str, Any],
        retrieval_needed: bool,
        retrieval_attempted: bool,
        retrieval_success: bool,
        utility_score: int,
        rag_assessments: Dict[str, str],
    ) -> None:
        """Store retrieval outcomes in thought metadata for future learning.

        Args:
            thought: The Thought to store outcomes in.
            learning_context: The learning context used.
            retrieval_needed: Whether retrieval was deemed needed.
            retrieval_attempted: Whether retrieval was actually attempted.
            retrieval_success: Whether retrieval was successful.
            utility_score: The utility score achieved.
            rag_assessments: The RAG assessments made.
        """
        if not thought.metadata:
            thought.metadata = {}

        # Initialize self-rag memory if not exists
        if "self_rag_memory" not in thought.metadata:
            thought.metadata["self_rag_memory"] = {
                "sessions": [],
                "effective_retrievals": [],
                "ineffective_retrievals": [],
                "task_retrieval_patterns": {},
            }

        # Analyze this retrieval session
        task_type = learning_context.get("task_type", "general")
        session_data = {
            "session_id": f"rag_session_{int(time.time())}",
            "task_type": task_type,
            "retrieval_needed": retrieval_needed,
            "retrieval_attempted": retrieval_attempted,
            "retrieval_success": retrieval_success,
            "utility_score": utility_score,
            "rag_assessments": rag_assessments,
            "timestamp": time.time(),
        }

        # Update effective/ineffective retrieval lists
        if retrieval_attempted:
            if retrieval_success and utility_score >= 4:
                # Successful retrieval
                thought.metadata["self_rag_memory"]["effective_retrievals"].append(
                    {
                        "task_type": task_type,
                        "utility_score": utility_score,
                        "relevance": rag_assessments.get("relevance", "[Partially Relevant]"),
                        "support": rag_assessments.get("support", "[Partially Supported]"),
                    }
                )
            elif not retrieval_success or utility_score < 3:
                # Unsuccessful retrieval
                thought.metadata["self_rag_memory"]["ineffective_retrievals"].append(
                    {
                        "task_type": task_type,
                        "utility_score": utility_score,
                        "reason": "low_utility" if retrieval_success else "retrieval_failed",
                    }
                )

        # Update task-specific retrieval patterns
        if task_type not in thought.metadata["self_rag_memory"]["task_retrieval_patterns"]:
            thought.metadata["self_rag_memory"]["task_retrieval_patterns"][task_type] = {
                "attempts": 0,
                "successes": 0,
                "total_utility": 0,
                "baseline_utility": 0,
            }

        patterns = thought.metadata["self_rag_memory"]["task_retrieval_patterns"][task_type]
        patterns["attempts"] += 1
        patterns["total_utility"] += utility_score

        if retrieval_attempted and retrieval_success:
            patterns["successes"] += 1

        # Calculate success rate and average utility improvement
        patterns["success_rate"] = patterns["successes"] / patterns["attempts"]
        patterns["avg_utility"] = patterns["total_utility"] / patterns["attempts"]

        # Estimate utility improvement (simplified heuristic)
        baseline_utility = 3  # Assume baseline utility without retrieval
        patterns["avg_utility_improvement"] = patterns["avg_utility"] - baseline_utility

        # Store this session
        thought.metadata["self_rag_memory"]["sessions"].append(session_data)

        # Keep only last 15 sessions
        if len(thought.metadata["self_rag_memory"]["sessions"]) > 15:
            thought.metadata["self_rag_memory"]["sessions"] = thought.metadata["self_rag_memory"][
                "sessions"
            ][-15:]

        # Keep only last 20 effective/ineffective retrievals
        if len(thought.metadata["self_rag_memory"]["effective_retrievals"]) > 20:
            thought.metadata["self_rag_memory"]["effective_retrievals"] = thought.metadata[
                "self_rag_memory"
            ]["effective_retrievals"][-20:]

        if len(thought.metadata["self_rag_memory"]["ineffective_retrievals"]) > 20:
            thought.metadata["self_rag_memory"]["ineffective_retrievals"] = thought.metadata[
                "self_rag_memory"
            ]["ineffective_retrievals"][-20:]
