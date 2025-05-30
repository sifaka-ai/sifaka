"""PydanticAI Chain implementation for Sifaka v0.2.0+

This module provides a simplified PydanticAIChain that uses PydanticAI's native
dependency injection and output validation instead of complex post-processing.

Breaking changes from v0.1.x:
- Removed ChainExecutor and complex orchestration
- Uses PydanticAI dependencies for validators/critics
- Simplified API with fewer configuration options
- Direct integration with PydanticAI features
"""

import asyncio
import uuid
from typing import List, Optional

from sifaka.agents.conversation import ConversationHistoryAdapter
from sifaka.agents.dependencies import SifakaDependencies
from sifaka.core.interfaces import Critic, Validator
from sifaka.core.thought import Thought
from sifaka.storage.protocol import Storage
from sifaka.utils.error_handling import ChainError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# PydanticAI is a required dependency
from pydantic_ai import Agent


class PydanticAIChain:
    """Simplified PydanticAI chain with native dependency injection.

    This class integrates PydanticAI agents with Sifaka validators and critics
    using PydanticAI's native dependency injection and output validation.

    Breaking changes from v0.1.x:
    - Removed complex post-processing orchestration
    - Uses PydanticAI output validators for validation
    - Simplified configuration with fewer options
    - Direct dependency injection approach
    """

    def __init__(
        self,
        agent: "Agent",
        validators: Optional[List[Validator]] = None,
        critics: Optional[List[Critic]] = None,
        model_retrievers: Optional[List] = None,
        critic_retrievers: Optional[List] = None,
        max_improvement_iterations: int = 2,
        always_apply_critics: bool = False,
        analytics_storage: Optional[Storage] = None,
        chain_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the PydanticAI chain with comprehensive retrieval and iteration support.

        Args:
            agent: The PydanticAI agent to use for generation.
            validators: List of Sifaka validators to always run if provided.
            critics: List of Sifaka critics to always run if provided.
            model_retrievers: List of retrievers for pre-generation context injection.
            critic_retrievers: List of retrievers for pre-critic context injection.
            max_improvement_iterations: Maximum number of improvement iterations.
            always_apply_critics: Whether to always apply critics (even on first success).
            analytics_storage: Optional storage backend for analytics/debugging only.
            chain_id: Optional chain identifier. Generated if not provided.
        """

        # Store configuration
        self.agent = agent
        self.validators = validators or []
        self.critics = critics or []
        self.model_retrievers = model_retrievers or []
        self.critic_retrievers = critic_retrievers or []
        self.max_improvement_iterations = max_improvement_iterations
        self.always_apply_critics = always_apply_critics

        self.analytics_storage = analytics_storage

        self.chain_id = chain_id or str(uuid.uuid4())

        # Create Sifaka dependencies for PydanticAI
        self.dependencies = SifakaDependencies(
            validators=self.validators,
            critics=self.critics,
        )

        # Create conversation history adapter
        self.conversation_adapter = ConversationHistoryAdapter(self.agent)

        # Configure PydanticAI agent retry behavior
        if hasattr(self.agent, "retries"):
            # Set PydanticAI retries to match our max_improvement_iterations
            self.agent.retries = self.max_improvement_iterations

        logger.info(
            f"Initialized PydanticAI chain {self.chain_id} with {len(self.validators)} validators, "
            f"{len(self.critics)} critics, {len(self.model_retrievers)} model retrievers, "
            f"{len(self.critic_retrievers)} critic retrievers, max_iterations={self.max_improvement_iterations}"
        )

    def __enter__(self):
        """Enter context manager - activate dependencies."""
        self.dependencies.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup dependencies."""
        return self.dependencies.__exit__(exc_type, exc_val, exc_tb)

    def cleanup(self):
        """Manually cleanup chain resources."""
        self.dependencies._cleanup()

    def get_conversation_history(self):
        """Get the current conversation history.

        Returns:
            List of PydanticAI messages in the conversation history.
        """
        return self.conversation_adapter.get_conversation_history()

    def get_conversation_summary(self):
        """Get a summary of the current conversation.

        Returns:
            Dictionary containing conversation statistics and summary.
        """
        return self.conversation_adapter.get_conversation_summary()

    def clear_conversation_history(self):
        """Clear the agent's conversation history."""
        self.conversation_adapter.clear_conversation_history()

    # NOTE: add_thought_to_conversation() removed - complex bidirectional conversion eliminated
    # PydanticAI conversation history is managed automatically by the agent

    async def run(self, prompt: str, **kwargs) -> Thought:
        """Execute the comprehensive PydanticAI chain with retrieval, validation, and criticism.

        This method implements the full Sifaka workflow:
        1. Pre-generation retrieval context injection
        2. Initial generation with validation
        3. Iterative improvement with critics (up to max_improvement_iterations)
        4. Comprehensive thought logging at each iteration

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional arguments passed to the agent.

        Returns:
            A Thought object containing the final result and complete audit trail.
        """
        logger.info(f"Starting chain execution for prompt: {prompt[:50]}...")

        # Create initial thought (iteration 0)
        thought = Thought(
            prompt=prompt,
            chain_id=self.chain_id,
            iteration=0,
            system_prompt=self._extract_system_prompt(),
            model_name=self._extract_model_name(),
        )

        try:
            # Execute pre-generation retrieval
            thought = await self._execute_model_retrieval(thought, prompt)

            # Build model prompt with retrieval context
            model_prompt = await self._build_model_prompt(thought, prompt)
            thought = thought.set_model_prompt(model_prompt)

            # Save initial thought (iteration 0) with system prompt, user prompt, and retrieval context
            await self._save_thought_for_analytics(thought)

            # Execute iterative generation with validation and criticism
            current_iteration = 0
            validation_passed = False

            while current_iteration <= self.max_improvement_iterations:
                logger.info(f"Starting iteration {current_iteration}")

                # Generate text using PydanticAI
                generation_result = await self._generate_with_agent(model_prompt, **kwargs)

                # Update thought with generated text
                if current_iteration == 0:
                    thought = thought.set_text(generation_result["output"])
                else:
                    # Create new iteration
                    thought = thought.next_iteration()
                    thought = thought.set_text(generation_result["output"])
                    thought = thought.set_model_prompt(model_prompt)

                # Extract and log tool calls
                thought = await self._extract_tool_calls(thought, generation_result)

                # Run validation if provided
                validation_passed, thought = await self._run_validation(thought)

                # Run critics if provided (always run if always_apply_critics=True)
                critic_feedback_applied = False
                if self.critics and (
                    not validation_passed
                    or self.always_apply_critics
                    or current_iteration < self.max_improvement_iterations
                ):
                    critic_feedback_applied, thought = await self._run_critics(thought)

                # Save thought for current iteration
                await self._save_thought_for_analytics(thought)

                # Check if we should continue iterating
                if validation_passed and not critic_feedback_applied:
                    logger.info(
                        f"Chain execution completed successfully at iteration {current_iteration}"
                    )
                    break

                if current_iteration >= self.max_improvement_iterations:
                    logger.warning(
                        f"Reached maximum iterations ({self.max_improvement_iterations})"
                    )
                    break

                # Prepare for next iteration with feedback (validation failures and/or critic feedback)
                # Build improvement prompt if we have validation failures OR critic feedback
                if critic_feedback_applied or not validation_passed:
                    model_prompt = await self._build_improvement_prompt(thought, prompt)

                current_iteration += 1

            return thought

        except Exception as e:
            logger.error(f"Chain execution failed: {e}")
            thought = thought.set_text(f"Error: {str(e)}")
            await self._save_thought_for_analytics(thought)
            raise ChainError(f"Chain execution failed: {e}") from e

    def run_sync(self, prompt: str, **kwargs) -> Thought:
        """Synchronous wrapper for the run method.

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional arguments passed to the agent.

        Returns:
            A Thought object containing the final result.
        """
        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're in an event loop, we can't use asyncio.run()
            # This is a limitation - sync execution from async context not supported
            raise RuntimeError(
                "Cannot run sync method from within an async context. Use await chain.run() instead."
            )
        except RuntimeError:
            # No event loop running, create a new one and run everything in it
            # This ensures all PydanticAI operations stay in the same event loop
            return asyncio.run(self._run_sync_in_loop(prompt, **kwargs))

    async def _run_sync_in_loop(self, prompt: str, **kwargs) -> Thought:
        """Run the chain in a single event loop to avoid event loop conflicts.

        This method ensures that all PydanticAI operations, including agent runs
        and any internal asyncio objects, stay within the same event loop.
        """
        return await self.run(prompt, **kwargs)

    def _extract_system_prompt(self) -> Optional[str]:
        """Extract system prompt from the PydanticAI agent."""
        try:
            # Try to access system prompts from the agent's internal storage
            if hasattr(self.agent, "_system_prompts") and self.agent._system_prompts:
                # _system_prompts is a tuple of system prompt strings
                # Join them with newlines if there are multiple
                return "\n".join(self.agent._system_prompts)

            # Fallback: try the system_prompt method (though this returns a decorator)
            if hasattr(self.agent, "system_prompt"):
                system_prompt = self.agent.system_prompt
                # If it's a callable (method), call it to get the actual prompt
                if callable(system_prompt):
                    system_prompt = system_prompt()
                # Convert to string if not None
                return str(system_prompt) if system_prompt is not None else None

            return None
        except Exception as e:
            logger.warning(f"Failed to extract system prompt: {e}")
            return None

    def _extract_model_name(self) -> Optional[str]:
        """Extract model name from the PydanticAI agent."""
        try:
            # Try to get the model name from the agent's model attribute
            if hasattr(self.agent, "model") and self.agent.model:
                model = self.agent.model
                # Handle different model types
                if hasattr(model, "model_name"):
                    return model.model_name
                elif hasattr(model, "name"):
                    return model.name
                elif hasattr(model, "__class__"):
                    # Use the class name as fallback
                    return f"pydantic-ai-{model.__class__.__name__}"
                else:
                    return str(model)
            return "pydantic-ai-agent"
        except Exception as e:
            logger.warning(f"Failed to extract model name: {e}")
            return "pydantic-ai-agent"

    async def _execute_model_retrieval(self, thought: Thought, prompt: str) -> Thought:
        """Execute pre-generation retrieval and add context to thought."""
        if not self.model_retrievers:
            return thought

        logger.debug(
            f"Executing pre-generation retrieval with {len(self.model_retrievers)} retrievers"
        )

        try:
            from sifaka.core.thought import Document

            all_documents = []
            for retriever in self.model_retrievers:
                try:
                    # Execute retrieval - prefer metadata-aware methods
                    if hasattr(retriever, "retrieve_with_metadata"):
                        # Use metadata-aware retrieval for better Document objects
                        logger.debug(
                            f"Using retrieve_with_metadata for {retriever.__class__.__name__}"
                        )
                        from sifaka.agents.core.async_utils import run_in_thread_pool

                        docs = await run_in_thread_pool(retriever.retrieve_with_metadata, prompt)
                    elif hasattr(retriever, "retrieve_async"):
                        logger.debug(f"Using retrieve_async for {retriever.__class__.__name__}")
                        docs = await retriever.retrieve_async(prompt)
                    else:
                        # Fall back to sync retrieval in thread pool
                        logger.debug(f"Using sync retrieve for {retriever.__class__.__name__}")
                        from sifaka.agents.core.async_utils import run_in_thread_pool

                        docs = await run_in_thread_pool(retriever.retrieve, prompt)

                    # Convert to Document objects if needed
                    for doc in docs:
                        if isinstance(doc, Document):
                            all_documents.append(doc)
                        elif isinstance(doc, dict):
                            all_documents.append(Document(**doc))
                        else:
                            # Assume it's a string
                            all_documents.append(Document(text=str(doc)))

                except Exception as e:
                    logger.error(f"Model retrieval failed for {retriever.__class__.__name__}: {e}")
                    continue

            if all_documents:
                thought = thought.add_pre_generation_context(all_documents)
                logger.debug(f"Added {len(all_documents)} documents to pre-generation context")

            return thought

        except Exception as e:
            logger.error(f"Pre-generation retrieval failed: {e}")
            return thought

    async def _build_model_prompt(self, thought: Thought, original_prompt: str) -> str:
        """Build the final prompt that will be sent to the model with RAG-optimized ordering.

        Order: system_prompt, prompt-retrieved content, prompt, validation results, critic results, original content
        """
        prompt_parts = []

        # 1. Add system prompt if available
        if thought.system_prompt:
            prompt_parts.append(f"System: {thought.system_prompt}")

        # 2. Add prompt-retrieved content first (context before question for better RAG)
        if thought.pre_generation_context:
            context_texts = [doc.text for doc in thought.pre_generation_context]
            context_str = "\n\n".join(context_texts)
            prompt_parts.append(f"Context:\n{context_str}")

        # 3. Add the user prompt (question with context fresh in mind)
        prompt_parts.append(f"User: {original_prompt}")

        # 4. Add validation results (if any from previous iterations)
        if thought.validation_results:
            failed_validations = [
                result for result in thought.validation_results.values() if not result.passed
            ]
            if failed_validations:
                validation_feedback = []
                for result in failed_validations:
                    issues = ", ".join(result.issues or [])
                    validation_feedback.append(f"- {result.validator_name}: {issues}")
                prompt_parts.append("Validation Issues:\n" + "\n".join(validation_feedback))

        # 5. Add critic results (if any from previous iterations)
        if hasattr(thought, "critic_feedback") and thought.critic_feedback:
            critic_feedback = []
            for feedback in thought.critic_feedback[-3:]:  # Last 3 feedbacks
                if hasattr(feedback, "suggestions") and feedback.suggestions:
                    critic_feedback.append(f"- {feedback.critic_name}: {feedback.suggestions}")
                elif hasattr(feedback, "feedback") and feedback.feedback:
                    critic_feedback.append(f"- {feedback.critic_name}: {feedback.feedback}")
            if critic_feedback:
                prompt_parts.append("Critic Feedback:\n" + "\n".join(critic_feedback))

        # 6. Add original content (previous attempt) if this is an improvement iteration
        if thought.iteration > 1 and thought.text:
            prompt_parts.append(f"Previous Attempt:\n{thought.text}")

        return "\n\n".join(prompt_parts)

    async def _generate_with_agent(self, model_prompt: str, **kwargs) -> dict:
        """Generate text using the PydanticAI agent and return rich result data."""
        try:
            # Run the agent with dependencies
            # Handle potential event loop conflicts by catching specific errors
            try:
                result = await self.agent.run(model_prompt, deps=self.dependencies, **kwargs)
            except RuntimeError as e:
                if "This event loop is already running" in str(e):
                    logger.error(f"Event loop conflict detected: {e}")
                    # For now, return a placeholder result to avoid breaking the chain
                    # This is a temporary workaround until PydanticAI fixes event loop handling
                    return {
                        "output": f"Error: Event loop conflict - {str(e)}",
                        "tool_calls": [],
                        "usage": {},
                    }
                else:
                    raise

            # Extract output text from result
            # Handle both real PydanticAI results and mock results
            if hasattr(result, "data"):
                # This is likely a PydanticAI AgentRunResult or our mock
                output = str(result.data)
            elif hasattr(result, "text"):
                output = result.text
            elif isinstance(result, str):
                output = result
            else:
                # Fallback - convert to string
                output = str(result)

            # Create rich data structure
            rich_data = {
                "output": output,
                "tool_calls": [],  # TODO: Extract actual tool calls from PydanticAI result
                "usage": {},  # TODO: Extract usage info from PydanticAI result
            }

            return rich_data

        except Exception as e:
            logger.error(f"Agent generation failed: {e}")
            raise

    async def _extract_tool_calls(self, thought: Thought, generation_result: dict) -> Thought:
        """Extract tool calls from generation result and add to thought."""
        try:
            # Extract tool calls from PydanticAI result if available
            # This is a placeholder - actual implementation depends on PydanticAI's result structure
            tool_calls = generation_result.get("tool_calls", [])

            from datetime import datetime

            from sifaka.core.thought import ToolCall

            for tool_call_data in tool_calls:
                tool_call = ToolCall(
                    tool_name=tool_call_data.get("tool_name", "unknown"),
                    arguments=tool_call_data.get("arguments", {}),
                    result=tool_call_data.get("result"),
                    timestamp=datetime.now(),
                    success=tool_call_data.get("success", True),
                    error_message=tool_call_data.get("error_message"),
                )
                thought = thought.add_tool_call(tool_call)

            return thought

        except Exception as e:
            logger.warning(f"Failed to extract tool calls: {e}")
            return thought

    async def _save_thought_for_analytics(self, thought: Thought):
        """Save thought to analytics storage if provided."""
        logger.info(f"_save_thought_for_analytics called with thought {thought.id}")
        logger.info(f"Analytics storage: {self.analytics_storage}")

        if self.analytics_storage is None:
            # No analytics storage configured - skip saving
            logger.warning("No analytics storage configured, skipping thought save")
            return

        logger.info(f"Attempting to save thought {thought.id} to analytics storage")

        try:
            # Storage methods are sync - run in thread pool to avoid blocking
            from sifaka.agents.core.async_utils import run_in_thread_pool

            if hasattr(self.analytics_storage, "save_thought"):
                logger.info("Using save_thought method")
                await run_in_thread_pool(self.analytics_storage.save_thought, thought)
            elif hasattr(self.analytics_storage, "save"):
                # Use thought.id as key to save each iteration separately
                logger.info(f"Using save method with key: {thought.id}")
                await run_in_thread_pool(self.analytics_storage.save, thought.id, thought)
                logger.info("Successfully saved thought to analytics storage")
            else:
                # Fallback for simple storage implementations
                logger.warning("Analytics storage does not support saving thoughts")
        except Exception as e:
            logger.error(f"Failed to save thought to analytics storage: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't fail the chain execution for analytics storage errors

    async def _run_validation(self, thought: Thought) -> tuple[bool, Thought]:
        """Run all validators on the thought and update it with results. Returns (all_passed, updated_thought)."""
        if not self.validators:
            return True, thought

        logger.debug(f"Running validation with {len(self.validators)} validators")

        all_passed = True
        updated_thought = thought

        try:
            for validator in self.validators:
                try:
                    # Run validation - prefer async methods, but use thread pool for sync methods
                    # that don't involve PydanticAI operations
                    if hasattr(validator, "_validate_async"):
                        result = await validator._validate_async(updated_thought)
                    elif self._validator_uses_pydantic_ai(validator):
                        # For validators that use PydanticAI models, run directly to avoid event loop issues
                        result = validator.validate(updated_thought)
                    else:
                        # For simple validators, use thread pool
                        from sifaka.agents.core.async_utils import run_in_thread_pool

                        result = await run_in_thread_pool(validator.validate, updated_thought)

                    # Add result to thought
                    updated_thought = updated_thought.add_validation_result(result)

                    if not result.passed:
                        all_passed = False
                        logger.debug(
                            f"Validation failed: {validator.__class__.__name__}: {', '.join(result.issues or [])}"
                        )
                    else:
                        logger.debug(f"Validation passed: {validator.__class__.__name__}")

                except Exception as e:
                    logger.error(f"Validation error for {validator.__class__.__name__}: {e}")
                    # Create error result
                    from sifaka.core.thought import ValidationResult

                    error_result = ValidationResult(
                        validator_name=validator.__class__.__name__,
                        passed=False,
                        issues=[str(e)],
                        suggestions=["Please check the validator configuration"],
                    )
                    updated_thought = updated_thought.add_validation_result(error_result)
                    all_passed = False

            return all_passed, updated_thought

        except Exception as e:
            logger.error(f"Validation execution failed: {e}")
            return False, thought

    async def _run_critics(self, thought: Thought) -> tuple[bool, Thought]:
        """Run all critics on the thought and update it with feedback. Returns (feedback_applied, updated_thought)."""
        if not self.critics:
            return False, thought

        logger.debug(f"Running criticism with {len(self.critics)} critics")

        feedback_applied = False
        updated_thought = thought

        try:
            # Execute pre-critic retrieval if configured
            updated_thought = await self._execute_critic_retrieval(updated_thought)

            for critic in self.critics:
                try:
                    # Run criticism using async methods to avoid event loop conflicts
                    if hasattr(critic, "critique_async"):
                        result = await critic.critique_async(updated_thought)
                    else:
                        # Fall back to sync method in thread pool
                        from sifaka.agents.core.async_utils import run_in_thread_pool

                        result = await run_in_thread_pool(critic.critique, updated_thought)

                    # Convert dict result to CriticFeedback if needed
                    from sifaka.core.thought import CriticFeedback

                    if isinstance(result, dict):
                        feedback = CriticFeedback(
                            critic_name=critic.__class__.__name__,
                            feedback=result.get("feedback", ""),
                            confidence=result.get("confidence", 0.0),
                            violations=result.get("violations", []),
                            suggestions=result.get("suggestions", []),
                            needs_improvement=result.get("needs_improvement", False),
                        )
                    else:
                        feedback = result

                    # Add feedback to thought
                    updated_thought = updated_thought.add_critic_feedback(feedback)

                    if feedback.needs_improvement:
                        feedback_applied = True
                        logger.debug(
                            f"Critic suggests improvement: {critic.__class__.__name__} (confidence: {feedback.confidence})"
                        )
                    else:
                        logger.debug(
                            f"Critic satisfied: {critic.__class__.__name__} (confidence: {feedback.confidence})"
                        )

                except Exception as e:
                    logger.error(f"Criticism error for {critic.__class__.__name__}: {e}")
                    # Create error feedback
                    from sifaka.core.thought import CriticFeedback

                    error_feedback = CriticFeedback(
                        critic_name=critic.__class__.__name__,
                        feedback="Criticism failed due to an error",
                        confidence=0.0,
                        violations=[str(e)],
                        suggestions=["Please check the critic configuration"],
                        needs_improvement=False,
                    )
                    updated_thought = updated_thought.add_critic_feedback(error_feedback)

            # If always_apply_critics is True, force feedback_applied to True
            # This ensures iteration continues even if critics say no improvement needed
            if self.always_apply_critics:
                feedback_applied = True
                logger.debug("always_apply_critics=True: forcing feedback_applied=True")

            return feedback_applied, updated_thought

        except Exception as e:
            logger.error(f"Criticism execution failed: {e}")
            return False, thought

    async def _execute_critic_retrieval(self, thought: Thought) -> Thought:
        """Execute pre-critic retrieval and add context to thought."""
        if not self.critic_retrievers:
            return thought

        logger.debug(
            f"Executing pre-critic retrieval with {len(self.critic_retrievers)} retrievers"
        )

        try:
            from sifaka.core.thought import Document

            # Use the current thought text as the query for critic retrieval
            query = thought.text or thought.prompt

            all_documents = []
            for retriever in self.critic_retrievers:
                try:
                    # Execute retrieval - prefer metadata-aware methods
                    if hasattr(retriever, "retrieve_with_metadata"):
                        # Use metadata-aware retrieval for better Document objects
                        logger.debug(
                            f"Using retrieve_with_metadata for critic retriever {retriever.__class__.__name__}"
                        )
                        from sifaka.agents.core.async_utils import run_in_thread_pool

                        docs = await run_in_thread_pool(retriever.retrieve_with_metadata, query)
                    elif hasattr(retriever, "retrieve_async"):
                        logger.debug(
                            f"Using retrieve_async for critic retriever {retriever.__class__.__name__}"
                        )
                        docs = await retriever.retrieve_async(query)
                    else:
                        # Fall back to sync retrieval in thread pool
                        logger.debug(
                            f"Using sync retrieve for critic retriever {retriever.__class__.__name__}"
                        )
                        from sifaka.agents.core.async_utils import run_in_thread_pool

                        docs = await run_in_thread_pool(retriever.retrieve, query)

                    # Convert to Document objects if needed
                    for doc in docs:
                        if isinstance(doc, Document):
                            all_documents.append(doc)
                        elif isinstance(doc, dict):
                            all_documents.append(Document(**doc))
                        else:
                            # Assume it's a string
                            all_documents.append(Document(text=str(doc)))

                except Exception as e:
                    logger.error(f"Critic retrieval failed for {retriever.__class__.__name__}: {e}")
                    continue

            if all_documents:
                thought = thought.add_post_generation_context(all_documents)
                logger.debug(
                    f"Added {len(all_documents)} documents to post-generation context for critics"
                )

            return thought

        except Exception as e:
            logger.error(f"Pre-critic retrieval failed: {e}")
            return thought

    async def _build_improvement_prompt(self, thought: Thought, original_prompt: str) -> str:
        """Build an improvement prompt with RAG-optimized ordering.

        Order: system_prompt, prompt-retrieved content, prompt, validation results, critic results, original content
        """
        prompt_parts = []

        # 1. Add system prompt if available
        if thought.system_prompt:
            prompt_parts.append(f"System: {thought.system_prompt}")

        # 2. Add prompt-retrieved content first (context before question for better RAG)
        if thought.pre_generation_context:
            context_texts = [doc.text for doc in thought.pre_generation_context]
            context_str = "\n\n".join(context_texts)
            prompt_parts.append(f"Context:\n{context_str}")

        # 3. Add the user prompt (question with context fresh in mind)
        prompt_parts.append(f"Original Request: {original_prompt}")

        # 4. Add validation feedback (most important for model to address)
        if thought.validation_results:
            failed_validations = [
                result for result in thought.validation_results.values() if not result.passed
            ]
            if failed_validations:
                validation_feedback = []
                for result in failed_validations:
                    issues = ", ".join(result.issues or [])
                    validation_feedback.append(f"- {result.validator_name}: {issues}")
                prompt_parts.append("Validation Issues:\n" + "\n".join(validation_feedback))

        # 5. Add critic feedback (additional improvement suggestions)
        if thought.critic_feedback:
            improvement_feedback = [fb for fb in thought.critic_feedback if fb.needs_improvement]
            if improvement_feedback:
                critic_suggestions = []
                for feedback in improvement_feedback:
                    suggestions = ", ".join(feedback.suggestions)
                    critic_suggestions.append(
                        f"- {feedback.critic_name}: {feedback.feedback} (Suggestions: {suggestions})"
                    )
                prompt_parts.append("Improvement Suggestions:\n" + "\n".join(critic_suggestions))

        # 6. Add original content (previous attempt for reference)
        if thought.text:
            prompt_parts.append(f"Previous Attempt:\n{thought.text}")

        # Add improvement instruction
        prompt_parts.append(
            "Please provide an improved response that addresses the validation issues and incorporates the improvement suggestions."
        )

        return "\n\n".join(prompt_parts)

    def _validator_uses_pydantic_ai(self, validator) -> bool:
        """Check if a validator uses PydanticAI models that need to stay in the same event loop."""
        try:
            # Check if the validator has a model attribute that's a PydanticAI model
            if hasattr(validator, "model"):
                model = validator.model
                # Check for PydanticAI model types
                if hasattr(model, "agent") or "pydantic" in str(type(model)).lower():
                    return True
            return False
        except Exception:
            # If we can't determine, assume it doesn't use PydanticAI
            return False

    def _critic_uses_pydantic_ai(self, critic) -> bool:
        """Check if a critic uses PydanticAI models that need to stay in the same event loop."""
        try:
            # Check if the critic has a model attribute that's a PydanticAI model
            if hasattr(critic, "model"):
                model = critic.model
                # Check for PydanticAI model types
                if hasattr(model, "agent") or "pydantic" in str(type(model)).lower():
                    return True
            return False
        except Exception:
            # If we can't determine, assume it doesn't use PydanticAI
            return False
