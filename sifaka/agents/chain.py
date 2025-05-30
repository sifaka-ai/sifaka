"""PydanticAI Chain implementation for Sifaka.

This module provides the main PydanticAIChain class that orchestrates
PydanticAI agents with Sifaka's validation and criticism framework.
"""

import asyncio
import uuid
from functools import wraps
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Critic, Validator
from sifaka.core.thought import Thought
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.protocol import Storage
from sifaka.utils.error_handling import ChainError
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation

logger = get_logger(__name__)


def async_to_sync(async_method):
    """Decorator to create sync wrapper for async methods."""

    @wraps(async_method)
    def sync_wrapper(self, *args, **kwargs):
        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we get here, we're in an async context, run in thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, async_method(self, *args, **kwargs))
                return future.result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(async_method(self, *args, **kwargs))

    return sync_wrapper


# Import PydanticAI with availability check
try:
    from pydantic_ai import Agent

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None


class PydanticAIChain:
    """Hybrid chain that orchestrates PydanticAI agents with Sifaka components.

    This class provides a composition-based approach to integrating PydanticAI
    agents with Sifaka's validation and criticism framework. It manages the
    workflow of generation, validation, and improvement while allowing
    PydanticAI to handle its own tool calling and internal feedback loops.
    """

    def __init__(
        self,
        agent: "Agent",
        storage: Optional[Storage] = None,
        validators: Optional[List[Validator]] = None,
        critics: Optional[List[Critic]] = None,
        model_retrievers: Optional[List] = None,
        critic_retrievers: Optional[List] = None,
        max_improvement_iterations: int = 2,
        always_apply_critics: bool = False,
        enable_critic_tools: bool = True,
        enable_validator_tools: bool = True,
        chain_id: Optional[str] = None,
    ):
        """Initialize the PydanticAI chain.

        Args:
            agent: The PydanticAI agent to use for generation.
            storage: Storage backend for thoughts. Defaults to MemoryStorage.
            validators: List of Sifaka validators to apply.
            critics: List of Sifaka critics to apply.
            model_retrievers: List of retrievers for pre-generation context.
            critic_retrievers: List of retrievers for critic-specific context.
            max_improvement_iterations: Maximum number of improvement iterations.
            always_apply_critics: Whether to always apply critics, even if validation passes.
            enable_critic_tools: Whether to register critics as PydanticAI tools.
            enable_validator_tools: Whether to register validators as PydanticAI tools.
            chain_id: Optional chain identifier. Generated if not provided.
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ChainError("PydanticAI is not available")

        self.agent = agent

        if storage is not None:
            self.storage = storage
        else:
            self.storage = MemoryStorage()

        self.validators = validators or []
        self.critics = critics or []
        self.model_retrievers = model_retrievers or []
        self.critic_retrievers = critic_retrievers or []
        self.max_improvement_iterations = max_improvement_iterations
        self.always_apply_critics = always_apply_critics
        self.chain_id = chain_id or str(uuid.uuid4())

        # Setup agent tools if enabled
        if enable_critic_tools:
            self._setup_critic_tools()
        if enable_validator_tools:
            self._setup_validator_tools()

    @async_to_sync
    async def run(self, prompt: str, **kwargs) -> Thought:
        """Execute the hybrid chain (sync wrapper for async implementation).

        This method provides a sync interface that wraps the async implementation,
        making it consistent with traditional Sifaka Chain API.

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional arguments passed to the agent.

        Returns:
            A Thought object containing the final result.
        """
        return await self.run_async(prompt, **kwargs)

    async def run_async(self, prompt: str, **kwargs) -> Thought:
        """Execute the hybrid chain asynchronously.

        This method orchestrates the complete workflow:
        1. Initial generation with PydanticAI
        2. Sifaka validation
        3. Improvement loop if validation fails

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional arguments passed to the agent.

        Returns:
            A Thought object containing the final result.
        """
        logger.info(f"Starting PydanticAI chain execution for prompt: {prompt[:50]}...")

        with time_operation("pydantic_ai_chain_execution"):
            # Create initial thought
            thought = Thought(prompt=prompt, chain_id=self.chain_id, iteration=0)

            # Phase 1: Pre-generation retrieval
            thought = await self._execute_model_retrieval_async(thought)

            # Phase 2: Initial generation with PydanticAI
            thought = await self._execute_agent_generation(thought, **kwargs)

            # Save initial iteration
            await self._save_intermediate_thought(thought)

            # Phase 3: Sifaka validation
            thought = await self._execute_validation_async(thought)

            # Phase 4: Improvement loop if validation fails OR always_apply_critics is True
            validation_passed = self._validation_passed(thought)
            should_improve = (not validation_passed and self.max_improvement_iterations > 0) or (
                getattr(self, "always_apply_critics", False) and self.max_improvement_iterations > 0
            )

            if should_improve:
                thought = await self._execute_improvement_loop(thought, **kwargs)

            # Save final result
            thought_key = f"thought_{thought.chain_id}_{thought.iteration}"
            try:
                logger.debug(f"Saving thought with key: {thought_key}")
                await self.storage._set_async(thought_key, thought)
                logger.info(
                    f"Successfully saved thought to: {getattr(self.storage, 'file_path', 'storage')}"
                )
            except Exception as e:
                logger.error(f"Failed to save thought: {e}")
                # Don't raise the exception to avoid breaking the chain execution

            logger.info(f"Chain execution completed. Final iteration: {thought.iteration}")
            return thought

    async def _execute_agent_generation(self, thought: Thought, **kwargs) -> Thought:
        """Execute initial generation using the PydanticAI agent.

        Args:
            thought: The current thought state.
            **kwargs: Additional arguments for the agent.

        Returns:
            Updated thought with generated text.
        """
        logger.debug("Executing PydanticAI agent generation")

        with time_operation("agent_generation"):
            try:
                # Run the PydanticAI agent asynchronously
                result = await self.agent.run(thought.prompt, **kwargs)

                # Extract output and update thought with comprehensive data
                output, updated_thought = self._extract_agent_output(result, thought)
                thought = (updated_thought or thought).set_text(output)

                # Extract and set comprehensive metadata
                model_name = self._extract_model_name()
                model_prompt = self._extract_model_prompt(result, thought.prompt)
                system_prompt = self._extract_system_prompt(result)

                thought = thought.model_copy(
                    update={
                        "iteration": 1,
                        "model_name": model_name,
                        "model_prompt": model_prompt,
                        "system_prompt": system_prompt,
                    }
                )

                logger.debug(f"Agent generated {len(output)} characters")
                return thought

            except Exception as e:
                logger.error(f"Agent generation failed: {e}")
                raise ChainError(f"PydanticAI agent generation failed: {e}")

    def _execute_validation(self, thought: Thought) -> Thought:
        """Execute Sifaka validation on the generated text (sync version for backward compatibility).

        Args:
            thought: The thought with generated text.

        Returns:
            Updated thought with validation results.
        """
        if not self.validators:
            logger.debug("No validators configured, skipping validation")
            return thought

        logger.debug(f"Running validation with {len(self.validators)} validators")

        with time_operation("validation"):
            for validator in self.validators:
                try:
                    result = validator.validate(thought)
                    thought = thought.add_validation_result(validator.__class__.__name__, result)
                    logger.debug(
                        f"Validation by {validator.__class__.__name__}: {'PASSED' if result.passed else 'FAILED'}"
                    )

                except Exception as e:
                    logger.error(f"Validation error for {validator.__class__.__name__}: {e}")
                    # Continue with other validators

            return thought

    async def _execute_validation_async(self, thought: Thought) -> Thought:
        """Execute Sifaka validation on the generated text asynchronously.

        Args:
            thought: The thought with generated text.

        Returns:
            Updated thought with validation results.
        """
        if not self.validators:
            logger.debug("No validators configured, skipping validation")
            return thought

        logger.debug(f"Running async validation with {len(self.validators)} validators")

        with time_operation("validation"):
            # Run all validators concurrently
            validation_tasks = []
            for validator in self.validators:
                validation_tasks.append(self._validate_with_validator_async(validator, thought))

            # Wait for all validations to complete
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(validation_results):
                validator = self.validators[i]
                validator_name = validator.__class__.__name__

                if isinstance(result, Exception):
                    logger.error(f"Validation error for {validator_name}: {result}")
                    # Continue with other validators - don't add failed validation to thought
                else:
                    # Add successful validation result to thought
                    thought = thought.add_validation_result(validator_name, result)
                    logger.debug(
                        f"Async validation by {validator_name}: {'PASSED' if result.passed else 'FAILED'}"
                    )

            return thought

    async def _validate_with_validator_async(self, validator, thought: Thought):
        """Run a single validator asynchronously with error handling."""
        try:
            # Check if validator has async method, otherwise use sync in thread pool
            if hasattr(validator, "_validate_async"):
                return await validator._validate_async(thought)  # type: ignore
            else:
                # Fall back to sync validation in thread pool to avoid blocking
                import concurrent.futures

                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return await loop.run_in_executor(executor, validator.validate, thought)
        except Exception as e:
            logger.error(f"Async validation failed for {validator.__class__.__name__}: {e}")
            raise

    async def _execute_improvement_loop(self, thought: Thought, **kwargs) -> Thought:
        """Execute improvement iterations using critics and agent feedback.

        Args:
            thought: The thought with validation results.
            **kwargs: Additional arguments for the agent.

        Returns:
            Improved thought after iterations.
        """
        logger.debug(
            f"Starting improvement loop (max {self.max_improvement_iterations} iterations)"
        )

        current_thought = thought

        for iteration in range(self.max_improvement_iterations):
            logger.debug(f"Improvement iteration {iteration + 1}")

            # Apply critic retrieval and then critics to get feedback
            current_thought = await self._execute_critic_retrieval_async(current_thought)
            current_thought = await self._execute_criticism_async(current_thought)

            # Create improvement prompt based on feedback
            improvement_prompt = self._create_improvement_prompt(current_thought)
            logger.debug(f"Improvement prompt: {improvement_prompt[:200]}...")

            # Generate improved text using PydanticAI
            try:
                logger.debug(
                    f"Running improvement iteration {iteration + 1} with prompt length: {len(improvement_prompt)}"
                )
                result = await self.agent.run(improvement_prompt, **kwargs)
                improved_text, updated_thought = self._extract_agent_output(result, current_thought)
                logger.debug(
                    f"Improvement iteration {iteration + 1} completed, generated {len(improved_text)} characters"
                )

                # Create new iteration (use updated thought if tool calls were captured)
                current_thought = (updated_thought or current_thought).next_iteration()
                current_thought = current_thought.set_text(improved_text)

                # Extract and update comprehensive metadata for improvement iteration
                model_name = self._extract_model_name()
                model_prompt = self._extract_model_prompt(result, improvement_prompt)
                system_prompt = self._extract_system_prompt(result)

                current_thought = current_thought.model_copy(
                    update={
                        "model_name": model_name,
                        "model_prompt": model_prompt,
                        "system_prompt": system_prompt,
                    }
                )

                # Re-validate
                current_thought = await self._execute_validation_async(current_thought)

                # Save intermediate iteration
                await self._save_intermediate_thought(current_thought)

                # Check if validation now passes
                if self._validation_passed(current_thought):
                    logger.debug(f"Validation passed after iteration {iteration + 1}")
                    break

            except Exception as e:
                logger.error(f"Improvement iteration {iteration + 1} failed: {e}")
                break

        return current_thought

    async def _execute_criticism_async(self, thought: Thought) -> Thought:
        """Execute criticism using configured critics asynchronously.

        Args:
            thought: The thought to critique.

        Returns:
            Updated thought with critic feedback.
        """
        if not self.critics:
            logger.debug("No critics configured, skipping criticism")
            return thought

        logger.debug(f"Running async criticism with {len(self.critics)} critics")

        # Run all critics concurrently
        criticism_tasks = []
        for critic in self.critics:
            criticism_tasks.append(self._critique_with_critic_async(critic, thought))

        # Wait for all criticisms to complete
        criticism_results = await asyncio.gather(*criticism_tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(criticism_results):
            critic = self.critics[i]
            critic_name = critic.__class__.__name__

            if isinstance(result, Exception):
                logger.error(f"Criticism error for {critic_name}: {result}")
                # Create error feedback
                from sifaka.core.thought import CriticFeedback

                error_feedback = CriticFeedback(
                    critic_name=critic_name,
                    feedback="Please try again or check the critic configuration",
                    confidence=0.0,
                    issues=[str(result)],
                    suggestions=["Please try again or check the critic configuration"],
                    needs_improvement=False,
                )
                thought = thought.add_critic_feedback(error_feedback)
            elif isinstance(result, dict):
                # Convert to CriticFeedback object and add to thought
                from sifaka.core.thought import CriticFeedback

                feedback = CriticFeedback(
                    critic_name=critic_name,
                    feedback=result.get("feedback", ""),
                    confidence=result.get("confidence", 0.0),
                    issues=result.get("issues", []),
                    suggestions=result.get("suggestions", []),
                    needs_improvement=result.get("needs_improvement", False),
                )
                thought = thought.add_critic_feedback(feedback)
                logger.debug(f"Added async feedback from {critic_name}")

        return thought

    async def _critique_with_critic_async(self, critic: Critic, thought: Thought) -> Dict[str, Any]:
        """Run a single critic asynchronously with error handling."""
        try:
            # Check if critic has async method, otherwise use sync in thread pool
            if hasattr(critic, "_critique_async"):
                return await critic._critique_async(thought)  # type: ignore
            else:
                # Fall back to sync criticism in thread pool to avoid blocking
                import concurrent.futures

                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return await loop.run_in_executor(executor, critic.critique, thought)
        except Exception as e:
            logger.error(f"Async criticism failed for {critic.__class__.__name__}: {e}")
            # Return error feedback dict
            return {"error": str(e), "confidence": 0.0, "issues": [str(e)], "suggestions": []}

    def _create_improvement_prompt(self, thought: Thought) -> str:
        """Create an improvement prompt based on validation and critic feedback.

        Args:
            thought: The thought with feedback.

        Returns:
            Improvement prompt string.
        """
        # Build feedback text from validation and critics
        feedback_parts = []
        has_length_constraint = False

        # Add validation feedback with priority emphasis
        if hasattr(thought, "validation_results") and thought.validation_results:
            validation_issues = []
            for name, result in thought.validation_results.items():
                if not result.passed:
                    validation_issues.append(f"- {name}: {result.message}")
                    if result.suggestions:
                        validation_issues.append(f"  Suggestions: {', '.join(result.suggestions)}")

                    # Check if this is a length constraint
                    if "length" in name.lower() or "too long" in result.message.lower():
                        has_length_constraint = True

            if validation_issues:
                if has_length_constraint:
                    feedback_parts.append("CRITICAL VALIDATION REQUIREMENTS (MUST BE ADDRESSED):")
                else:
                    feedback_parts.append("Validation Issues:")
                feedback_parts.extend(validation_issues)

        # Add critic feedback with conditional filtering for length constraints
        if hasattr(thought, "critic_feedback") and thought.critic_feedback:
            critic_suggestions = []
            for feedback in thought.critic_feedback:
                # Include feedback if it needs improvement OR has useful suggestions
                if feedback.needs_improvement or (
                    feedback.suggestions and any(s.strip() for s in feedback.suggestions)
                ):
                    critic_line = f"- {feedback.critic_name}"
                    if feedback.feedback and feedback.feedback.strip():
                        critic_line += f": {feedback.feedback}"
                    critic_suggestions.append(critic_line)

                    if feedback.suggestions:
                        # Filter out generic suggestions
                        useful_suggestions = [
                            s
                            for s in feedback.suggestions
                            if s.strip() and s != "See critique for improvement suggestions"
                        ]

                        # If we have length constraints, filter out suggestions that would add content
                        if has_length_constraint:
                            filtered_suggestions = []
                            for suggestion in useful_suggestions:
                                # Skip suggestions that clearly ask for more content
                                if not any(
                                    phrase in suggestion.lower()
                                    for phrase in [
                                        "provide more",
                                        "include more",
                                        "add more",
                                        "incorporate",
                                        "enhance",
                                        "expand",
                                        "elaborate",
                                        "examples",
                                        "case studies",
                                    ]
                                ):
                                    filtered_suggestions.append(suggestion)
                            useful_suggestions = filtered_suggestions

                        if useful_suggestions:
                            critic_suggestions.append(
                                f"  Suggestions: {', '.join(useful_suggestions)}"
                            )

            if critic_suggestions:
                if has_length_constraint:
                    feedback_parts.append(
                        "\nSecondary Feedback (only if compatible with length requirements):"
                    )
                else:
                    feedback_parts.append("\nCritic Feedback:")
                feedback_parts.extend(critic_suggestions)

        combined_feedback = (
            "\n".join(feedback_parts) if feedback_parts else "No specific feedback available."
        )

        # Create improvement prompt with emphasis on validation constraints
        if has_length_constraint:
            improvement_prompt = (
                "Please improve the following text based on the feedback provided.\n\n"
                "⚠️  IMPORTANT: The text MUST meet the length requirements. This is a hard constraint "
                "that takes priority over all other suggestions. Focus on reducing content while "
                "maintaining the core message.\n\n"
                f"Original task: {thought.prompt}\n\n"
                f"Current text:\n{thought.text}\n\n"
                f"Feedback:\n{combined_feedback}\n\n"
                "Please provide an improved version that FIRST addresses the validation requirements "
                "(especially length constraints), then incorporates other feedback only if it doesn't "
                "conflict with the validation requirements. You may use your tools if needed.\n\n"
                "Improved text:"
            )
        else:
            improvement_prompt = (
                "Please improve the following text based on the feedback provided.\n\n"
                f"Original task: {thought.prompt}\n\n"
                f"Current text:\n{thought.text}\n\n"
                f"Feedback:\n{combined_feedback}\n\n"
                "Please provide an improved version that addresses the issues identified "
                "in the feedback while maintaining the core message and staying true to "
                "the original task. You may use your tools if needed to gather additional "
                "information or validate your improvements.\n\n"
                "Improved text:"
            )

        return improvement_prompt

    def _validation_passed(self, thought: Thought) -> bool:
        """Check if all validations passed.

        Args:
            thought: The thought to check.

        Returns:
            True if all validations passed, False otherwise.
        """
        if not hasattr(thought, "validation_results") or not thought.validation_results:
            return True  # No validations means pass

        return all(result.passed for result in thought.validation_results.values())

    def _extract_agent_output(
        self, result, thought: Optional[Thought] = None
    ) -> tuple[str, Optional[Thought]]:
        """Extract text output from PydanticAI agent result and log tool calls.

        Args:
            result: The agent result object.
            thought: Optional thought to add tool calls to.

        Returns:
            Tuple of (extracted_text_output, updated_thought_with_tool_calls).
        """
        updated_thought = thought

        try:
            # Extract tool calls if available
            if hasattr(result, "all_messages") and updated_thought is not None:
                updated_thought = self._extract_tool_calls(result, updated_thought)

            # Extract output text
            if hasattr(result, "output"):
                output = result.output
                if isinstance(output, str):
                    return output, updated_thought
                else:
                    return str(output), updated_thought
            else:
                return str(result), updated_thought

        except Exception as e:
            logger.warning(f"Failed to extract agent output: {e}")
            return str(result), updated_thought

    def _extract_model_name(self) -> str:
        """Extract model name from PydanticAI agent.

        Returns:
            The model name string.
        """
        try:
            # Try to get model name from the agent
            if hasattr(self.agent, "model"):
                model = self.agent.model
                if hasattr(model, "_model_name"):
                    return f"pydantic-ai:{model._model_name}"
                elif hasattr(model, "name"):
                    return f"pydantic-ai:{model.name}"
                else:
                    return f"pydantic-ai:{str(model)}"
            else:
                return "pydantic-ai-agent"
        except Exception as e:
            logger.warning(f"Failed to extract model name: {e}")
            return "pydantic-ai-agent"

    def _extract_model_prompt(self, result, fallback_prompt: str) -> str:
        """Extract the actual prompt sent to the model from PydanticAI result.

        Args:
            result: The agent result object.
            fallback_prompt: Fallback prompt if extraction fails.

        Returns:
            The model prompt string.
        """
        try:
            # Try to get the actual prompt from messages
            if hasattr(result, "all_messages"):
                messages = result.all_messages()
                for message in messages:
                    if hasattr(message, "content") and hasattr(message, "role"):
                        if message.role == "user":
                            return str(message.content)

            # Fallback to the original prompt
            return fallback_prompt

        except Exception as e:
            logger.warning(f"Failed to extract model prompt: {e}")
            return fallback_prompt

    def _extract_system_prompt(self, result) -> str:
        """Extract system prompt from PydanticAI result.

        Args:
            result: The agent result object.

        Returns:
            The system prompt string.
        """
        try:
            # Try to get system prompt from agent's internal storage first
            if hasattr(self.agent, "_system_prompts") and self.agent._system_prompts:
                # PydanticAI stores system prompts in a tuple
                return str(self.agent._system_prompts[0])

            # Fallback: Try to get system prompt from agent
            if hasattr(self.agent, "system_prompt"):
                system_prompt = self.agent.system_prompt
                if callable(system_prompt):
                    # For callable system prompts, try to call it or get its string representation
                    try:
                        # Try calling with no arguments first
                        return str(system_prompt())
                    except Exception:
                        try:
                            # Try to get the actual prompt text from the agent's internal state
                            if hasattr(self.agent, "_system_prompt"):
                                return str(self.agent._system_prompt)
                            # Try to extract from the function if it's a simple string wrapper
                            elif (
                                hasattr(system_prompt, "__closure__") and system_prompt.__closure__
                            ):
                                for cell in system_prompt.__closure__:
                                    if isinstance(cell.cell_contents, str):
                                        return cell.cell_contents
                            # Fallback to a descriptive string
                            return "Dynamic system prompt (callable)"
                        except Exception:
                            return "Dynamic system prompt (callable)"
                else:
                    return str(system_prompt) if system_prompt else ""

            # Try to get from messages
            if hasattr(result, "all_messages"):
                messages = result.all_messages()
                for message in messages:
                    if hasattr(message, "content") and hasattr(message, "role"):
                        if message.role == "system":
                            return str(message.content)

            return ""

        except Exception as e:
            logger.warning(f"Failed to extract system prompt: {e}")
            return ""

    def _extract_tool_calls(self, result, thought: Thought) -> Thought:
        """Extract tool calls from PydanticAI result and add to thought.

        Args:
            result: The agent result object.
            thought: The thought to add tool calls to.

        Returns:
            Updated thought with tool calls.
        """
        try:
            from datetime import datetime

            from sifaka.core.thought import ToolCall

            if hasattr(result, "all_messages"):
                for message in result.all_messages():
                    # Check for tool calls in the message
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            # Extract tool call information based on PydanticAI structure
                            tool_name = "unknown"
                            arguments = {}

                            if hasattr(tool_call, "function"):
                                func = tool_call.function
                                if hasattr(func, "name"):
                                    tool_name = func.name
                                if hasattr(func, "arguments"):
                                    arguments = func.arguments
                            elif hasattr(tool_call, "name"):
                                tool_name = tool_call.name
                                if hasattr(tool_call, "arguments"):
                                    arguments = tool_call.arguments

                            tool_record = ToolCall(
                                tool_name=tool_name,
                                arguments=arguments,
                                timestamp=datetime.now(),
                                success=True,  # Assume success if we got here
                            )
                            thought = thought.add_tool_call(tool_record)

            return thought

        except Exception as e:
            logger.warning(f"Failed to extract tool calls: {e}")
            return thought

    async def _save_intermediate_thought(self, thought: Thought):
        """Save an intermediate thought iteration asynchronously.

        Args:
            thought: The thought to save.
        """
        try:
            thought_key = f"thought_{thought.chain_id}_{thought.iteration}"
            logger.debug(f"Saving intermediate thought with key: {thought_key}")
            await self.storage._set_async(thought_key, thought)
            logger.debug(f"Successfully saved intermediate thought: iteration {thought.iteration}")
        except Exception as e:
            logger.error(f"Failed to save intermediate thought: {e}")
            # Don't raise the exception to avoid breaking the chain execution

    def _execute_model_retrieval(self, thought: Thought) -> Thought:
        """Execute pre-generation retrieval using model retrievers (sync version for backward compatibility).

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with retrieved context.
        """
        if not self.model_retrievers:
            logger.debug("No model retrievers configured, skipping model retrieval")
            return thought

        logger.debug(f"Running model retrieval with {len(self.model_retrievers)} retrievers")

        with time_operation("model_retrieval"):
            for retriever in self.model_retrievers:
                try:
                    # Use the retriever's retrieve_for_thought method for pre-generation
                    thought = retriever.retrieve_for_thought(thought, is_pre_generation=True)
                    logger.debug(f"Applied model retriever: {retriever.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Model retrieval error for {retriever.__class__.__name__}: {e}")
                    # Continue with other retrievers

            return thought

    async def _execute_model_retrieval_async(self, thought: Thought) -> Thought:
        """Execute pre-generation retrieval using model retrievers asynchronously.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with retrieved context.
        """
        if not self.model_retrievers:
            logger.debug("No model retrievers configured, skipping model retrieval")
            return thought

        logger.debug(f"Running async model retrieval with {len(self.model_retrievers)} retrievers")

        with time_operation("model_retrieval"):
            # Run all retrievers concurrently
            retrieval_tasks = []
            for retriever in self.model_retrievers:
                retrieval_tasks.append(
                    self._retrieve_with_retriever_async(retriever, thought, is_pre_generation=True)
                )

            # Wait for all retrievals to complete
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

            # Process results sequentially to maintain thought state consistency
            for i, result in enumerate(retrieval_results):
                retriever = self.model_retrievers[i]
                retriever_name = retriever.__class__.__name__

                if isinstance(result, Exception):
                    logger.error(f"Model retrieval error for {retriever_name}: {result}")
                    # Continue with other retrievers
                else:
                    # Update thought with retrieved context
                    thought = result
                    logger.debug(f"Applied async model retriever: {retriever_name}")

            return thought

    def _execute_critic_retrieval(self, thought: Thought) -> Thought:
        """Execute retrieval for critics using critic retrievers (sync version for backward compatibility).

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with critic-specific context.
        """
        if not self.critic_retrievers:
            logger.debug("No critic retrievers configured, skipping critic retrieval")
            return thought

        logger.debug(f"Running critic retrieval with {len(self.critic_retrievers)} retrievers")

        with time_operation("critic_retrieval"):
            for retriever in self.critic_retrievers:
                try:
                    # Use the retriever's retrieve_for_thought method for post-generation/critic context
                    thought = retriever.retrieve_for_thought(thought, is_pre_generation=False)
                    logger.debug(f"Applied critic retriever: {retriever.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Critic retrieval error for {retriever.__class__.__name__}: {e}")
                    # Continue with other retrievers

            return thought

    async def _execute_critic_retrieval_async(self, thought: Thought) -> Thought:
        """Execute retrieval for critics using critic retrievers asynchronously.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with critic-specific context.
        """
        if not self.critic_retrievers:
            logger.debug("No critic retrievers configured, skipping critic retrieval")
            return thought

        logger.debug(
            f"Running async critic retrieval with {len(self.critic_retrievers)} retrievers"
        )

        with time_operation("critic_retrieval"):
            # Run all retrievers concurrently
            retrieval_tasks = []
            for retriever in self.critic_retrievers:
                retrieval_tasks.append(
                    self._retrieve_with_retriever_async(retriever, thought, is_pre_generation=False)
                )

            # Wait for all retrievals to complete
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

            # Process results sequentially to maintain thought state consistency
            for i, result in enumerate(retrieval_results):
                retriever = self.critic_retrievers[i]
                retriever_name = retriever.__class__.__name__

                if isinstance(result, Exception):
                    logger.error(f"Critic retrieval error for {retriever_name}: {result}")
                    # Continue with other retrievers
                else:
                    # Update thought with retrieved context
                    thought = result
                    logger.debug(f"Applied async critic retriever: {retriever_name}")

            return thought

    async def _retrieve_with_retriever_async(
        self, retriever, thought: Thought, is_pre_generation: bool
    ) -> Thought:
        """Run a single retriever asynchronously with error handling."""
        try:
            # Check if retriever has async method, otherwise use sync in thread pool
            if hasattr(retriever, "_retrieve_for_thought_async"):
                return await retriever._retrieve_for_thought_async(thought, is_pre_generation)  # type: ignore
            else:
                # Fall back to sync retrieval in thread pool to avoid blocking
                import concurrent.futures

                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return await loop.run_in_executor(
                        executor, retriever.retrieve_for_thought, thought, is_pre_generation
                    )
        except Exception as e:
            logger.error(f"Async retrieval failed for {retriever.__class__.__name__}: {e}")
            raise

    def _setup_critic_tools(self):
        """Setup critics as PydanticAI tools (placeholder for Phase 2)."""
        # This will be implemented in Phase 2
        logger.debug("Critic tools setup (placeholder)")

    def _setup_validator_tools(self):
        """Setup validators as PydanticAI tools (placeholder for Phase 2)."""
        # This will be implemented in Phase 2
        logger.debug("Validator tools setup (placeholder)")
