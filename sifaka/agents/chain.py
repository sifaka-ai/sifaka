"""PydanticAI Chain implementation for Sifaka.

This module provides the main PydanticAIChain class that orchestrates
PydanticAI agents with Sifaka's validation and criticism framework.
"""

import uuid
from typing import List, Optional

from sifaka.core.interfaces import Critic, Validator
from sifaka.core.thought import Thought
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.protocol import Storage
from sifaka.utils.error_handling import ChainError
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation

logger = get_logger(__name__)

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
        max_improvement_iterations: int = 2,
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
            max_improvement_iterations: Maximum number of improvement iterations.
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
        self.max_improvement_iterations = max_improvement_iterations
        self.chain_id = chain_id or str(uuid.uuid4())

        # Setup agent tools if enabled
        if enable_critic_tools:
            self._setup_critic_tools()
        if enable_validator_tools:
            self._setup_validator_tools()

    def run(self, prompt: str, **kwargs) -> Thought:
        """Execute the hybrid chain.

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

            # Phase 1: Initial generation with PydanticAI
            thought = self._execute_agent_generation(thought, **kwargs)

            # Save initial iteration
            self._save_intermediate_thought(thought)

            # Phase 2: Sifaka validation
            thought = self._execute_validation(thought)

            # Phase 3: Improvement loop if validation fails
            if not self._validation_passed(thought) and self.max_improvement_iterations > 0:
                thought = self._execute_improvement_loop(thought, **kwargs)

            # Save final result
            thought_key = f"thought_{thought.chain_id}_{thought.iteration}"
            try:
                logger.debug(f"Saving thought with key: {thought_key}")
                self.storage.save(thought_key, thought)
                logger.info(
                    f"Successfully saved thought to: {getattr(self.storage, 'file_path', 'storage')}"
                )
            except Exception as e:
                logger.error(f"Failed to save thought: {e}")
                # Don't raise the exception to avoid breaking the chain execution

            logger.info(f"Chain execution completed. Final iteration: {thought.iteration}")
            return thought

    def _execute_agent_generation(self, thought: Thought, **kwargs) -> Thought:
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
                # Run the PydanticAI agent
                result = self.agent.run_sync(thought.prompt, **kwargs)

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
        """Execute Sifaka validation on the generated text.

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

    def _execute_improvement_loop(self, thought: Thought, **kwargs) -> Thought:
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

            # Apply critics to get feedback
            current_thought = self._execute_criticism(current_thought)

            # Create improvement prompt based on feedback
            improvement_prompt = self._create_improvement_prompt(current_thought)
            logger.debug(f"Improvement prompt: {improvement_prompt[:200]}...")

            # Generate improved text using PydanticAI
            try:
                logger.debug(
                    f"Running improvement iteration {iteration + 1} with prompt length: {len(improvement_prompt)}"
                )
                result = self.agent.run_sync(improvement_prompt, **kwargs)
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
                current_thought = self._execute_validation(current_thought)

                # Save intermediate iteration
                self._save_intermediate_thought(current_thought)

                # Check if validation now passes
                if self._validation_passed(current_thought):
                    logger.debug(f"Validation passed after iteration {iteration + 1}")
                    break

            except Exception as e:
                logger.error(f"Improvement iteration {iteration + 1} failed: {e}")
                break

        return current_thought

    def _execute_criticism(self, thought: Thought) -> Thought:
        """Execute criticism using configured critics.

        Args:
            thought: The thought to critique.

        Returns:
            Updated thought with critic feedback.
        """
        if not self.critics:
            logger.debug("No critics configured, skipping criticism")
            return thought

        logger.debug(f"Running criticism with {len(self.critics)} critics")

        for critic in self.critics:
            try:
                feedback_dict = critic.critique(thought)
                # Convert to CriticFeedback object and add to thought
                from sifaka.core.thought import CriticFeedback

                feedback = CriticFeedback(
                    critic_name=critic.__class__.__name__,
                    feedback=feedback_dict.get("feedback", ""),
                    confidence=feedback_dict.get("confidence", 0.0),
                    issues=feedback_dict.get("issues", []),
                    suggestions=feedback_dict.get("suggestions", []),
                )
                thought = thought.add_critic_feedback(feedback)

            except Exception as e:
                logger.error(f"Criticism error for {critic.__class__.__name__}: {e}")

        return thought

    def _create_improvement_prompt(self, thought: Thought) -> str:
        """Create an improvement prompt based on validation and critic feedback.

        Args:
            thought: The thought with feedback.

        Returns:
            Improvement prompt string.
        """
        # Build feedback text from validation and critics
        feedback_parts = []

        # Add validation feedback
        if hasattr(thought, "validation_results") and thought.validation_results:
            feedback_parts.append("Validation Issues:")
            for name, result in thought.validation_results.items():
                if not result.passed:
                    feedback_parts.append(f"- {name}: {result.message}")
                    if result.suggestions:
                        feedback_parts.append(f"  Suggestions: {', '.join(result.suggestions)}")

        # Add critic feedback
        if hasattr(thought, "critic_feedback") and thought.critic_feedback:
            feedback_parts.append("\nCritic Feedback:")
            for feedback in thought.critic_feedback:
                if feedback.confidence < 0.7:  # Only include concerning feedback
                    feedback_parts.append(f"- {feedback.critic_name}: {feedback.feedback}")
                    if feedback.suggestions:
                        feedback_parts.append(f"  Suggestions: {', '.join(feedback.suggestions)}")

        combined_feedback = (
            "\n".join(feedback_parts) if feedback_parts else "No specific feedback available."
        )

        # Use the same pattern as SelfRefineCritic
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

    def _save_intermediate_thought(self, thought: Thought):
        """Save an intermediate thought iteration.

        Args:
            thought: The thought to save.
        """
        try:
            thought_key = f"thought_{thought.chain_id}_{thought.iteration}"
            logger.debug(f"Saving intermediate thought with key: {thought_key}")
            self.storage.save(thought_key, thought)
            logger.debug(f"Successfully saved intermediate thought: iteration {thought.iteration}")
        except Exception as e:
            logger.error(f"Failed to save intermediate thought: {e}")
            # Don't raise the exception to avoid breaking the chain execution

    def _setup_critic_tools(self):
        """Setup critics as PydanticAI tools (placeholder for Phase 2)."""
        # This will be implemented in Phase 2
        logger.debug("Critic tools setup (placeholder)")

    def _setup_validator_tools(self):
        """Setup validators as PydanticAI tools (placeholder for Phase 2)."""
        # This will be implemented in Phase 2
        logger.debug("Validator tools setup (placeholder)")
