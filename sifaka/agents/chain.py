"""Simplified PydanticAI Chain implementation for Sifaka v0.3.0+

This module provides a clean, modular PydanticAIChain that orchestrates
specialized execution modules for different chain phases.

Key improvements in v0.3.0:
- Modular architecture with focused execution modules
- Configuration object instead of scattered parameters
- Clean async-only interface
- No backward compatibility code
"""

import uuid
from typing import Optional

from pydantic_ai import Agent

from sifaka.agents.config import ChainConfig
from sifaka.agents.conversation import ConversationHistoryAdapter
from sifaka.agents.dependencies import SifakaDependencies
from sifaka.agents.execution.criticism import CriticismExecutor
from sifaka.agents.execution.generation import GenerationExecutor
from sifaka.agents.execution.retrieval import RetrievalExecutor
from sifaka.agents.execution.validation import ValidationExecutor
from sifaka.agents.prompt.builder import PromptBuilder
from sifaka.core.thought import Thought
from sifaka.utils.error_handling import ChainError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class PydanticAIChain:
    """Simplified PydanticAI chain with modular execution architecture.

    This class orchestrates specialized execution modules for different chain phases:
    - Generation: Text generation using PydanticAI agents
    - Validation: Running validators on generated text
    - Criticism: Running critics for improvement feedback
    - Retrieval: Context retrieval for both generation and criticism
    - Prompt Building: Constructing prompts for different phases

    Key improvements in v0.3.0:
    - Clean modular architecture
    - Configuration object instead of scattered parameters
    - Async-only interface
    - No backward compatibility code
    """

    def __init__(self, agent: Agent, config: ChainConfig):
        """Initialize the PydanticAI chain with modular execution architecture.

        Args:
            agent: The PydanticAI agent to use for generation.
            config: Configuration object containing all chain settings.
        """
        # Store core components
        self.agent = agent
        self.config = config
        self.chain_id = config.chain_id or str(uuid.uuid4())

        # Initialize execution modules
        self.generator = GenerationExecutor(agent)
        self.validator = ValidationExecutor(config.validators)
        self.critic = CriticismExecutor(config.critics)
        self.retriever = RetrievalExecutor(config.model_retrievers, config.critic_retrievers)
        self.prompt_builder = PromptBuilder()

        # Create Sifaka dependencies for PydanticAI
        self.dependencies = SifakaDependencies(
            validators=config.validators,
            critics=config.critics,
        )

        # Create conversation history adapter
        self.conversation_adapter = ConversationHistoryAdapter(self.agent)

        # Configure PydanticAI agent retry behavior
        if hasattr(self.agent, "retries"):
            # Set PydanticAI retries to match our max_improvement_iterations
            self.agent.retries = config.max_improvement_iterations

        logger.info(
            f"Initialized PydanticAI chain {self.chain_id} with {len(config.validators)} validators, "
            f"{len(config.critics)} critics, {len(config.model_retrievers)} model retrievers, "
            f"{len(config.critic_retrievers)} critic retrievers, max_iterations={config.max_improvement_iterations}"
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
        """Execute the simplified PydanticAI chain with modular execution.

        This method implements a clean, linear workflow:
        1. Initialize thought with metadata
        2. Execute pre-generation retrieval
        3. Iterative generation, validation, and improvement
        4. Return final thought with complete audit trail

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional arguments passed to the agent.

        Returns:
            A Thought object containing the final result and complete audit trail.
        """
        logger.info(f"Starting chain execution for prompt: {prompt[:50]}...")

        # Create initial thought with extracted metadata
        thought = Thought(
            prompt=prompt,
            chain_id=self.chain_id,
            iteration=0,
            system_prompt=self._extract_system_prompt(),
            model_name=self._extract_model_name(),
        )

        try:
            # Execute pre-generation retrieval
            thought = await self.retriever.execute_model_retrieval(thought)

            # Build initial model prompt
            model_prompt = self.prompt_builder.build_model_prompt(thought, prompt)
            thought = thought.set_model_prompt(model_prompt)

            # Save initial thought
            await self._save_thought_for_analytics(thought)

            # Execute iterative generation with validation and criticism
            for iteration in range(self.config.max_improvement_iterations + 1):
                logger.info(f"Starting iteration {iteration}")

                # Generate text (unless we already have improved text from critic)
                if not hasattr(thought, "_critic_improved") or not thought._critic_improved:
                    thought = await self.generator.execute(thought, **kwargs)
                else:
                    # Reset the flag after using critic-improved text
                    thought._critic_improved = False

                # Run validation
                thought = await self.validator.execute(thought)
                validation_passed = self.validator.validation_passed(thought)

                # Run critics if needed
                critic_feedback_applied = await self._should_run_critics(
                    validation_passed, iteration
                )
                if critic_feedback_applied:
                    # Execute critic retrieval first
                    thought = await self.retriever.execute_critic_retrieval(thought)
                    # Then run critics
                    thought = await self.critic.execute(thought)

                # Save thought for current iteration
                await self._save_thought_for_analytics(thought)

                # Check if we should continue
                if validation_passed and not critic_feedback_applied:
                    logger.info(f"Chain execution completed successfully at iteration {iteration}")
                    break

                if iteration >= self.config.max_improvement_iterations:
                    logger.warning(
                        f"Reached maximum iterations ({self.config.max_improvement_iterations})"
                    )
                    break

                # Prepare for next iteration
                if iteration < self.config.max_improvement_iterations:
                    # Try to use critic's improvement method BEFORE calling next_iteration()
                    # (since next_iteration() clears the text field)
                    improved_text = None
                    if thought.critic_feedback and self.config.critics:
                        for critic in self.config.critics:
                            if hasattr(critic, "improve_async"):
                                try:
                                    logger.debug(
                                        f"Using {critic.__class__.__name__} improve_async method"
                                    )
                                    improved_text = await critic.improve_async(thought)
                                    logger.debug(
                                        f"Critic improvement completed: {len(improved_text)} characters"
                                    )
                                    break
                                except Exception as e:
                                    logger.warning(
                                        f"Critic {critic.__class__.__name__} improvement failed: {e}"
                                    )
                                    continue

                    # Now create next iteration
                    thought = thought.next_iteration()

                    if improved_text is not None:
                        # Use critic-improved text directly
                        thought = thought.set_text(improved_text)
                        # Mark that this thought has critic-improved text to skip generation
                        thought._critic_improved = True
                        # Still need a model prompt for consistency, but it won't be used for generation
                        improvement_prompt = (
                            f"Improved text from {critic.__class__.__name__}: {improved_text}"
                        )
                        thought = thought.set_model_prompt(improvement_prompt)
                    else:
                        # Fallback to prompt builder
                        improvement_prompt = self.prompt_builder.build_improvement_prompt(
                            thought, prompt
                        )
                        thought = thought.set_model_prompt(improvement_prompt)

            return thought

        except Exception as e:
            logger.error(f"Chain execution failed: {e}")
            thought = thought.set_text(f"Error: {str(e)}")
            await self._save_thought_for_analytics(thought)
            raise ChainError(f"Chain execution failed: {e}") from e

    async def _should_run_critics(self, validation_passed: bool, iteration: int) -> bool:
        """Determine if critics should be run based on configuration and state.

        Args:
            validation_passed: Whether validation passed.
            iteration: Current iteration number.

        Returns:
            True if critics should be run, False otherwise.
        """
        if not self.config.critics:
            return False

        # Always run critics if configured to do so
        if self.config.always_apply_critics:
            return True

        # Run critics if validation failed
        if not validation_passed:
            return True

        # Run critics if we're not on the last iteration
        if iteration < self.config.max_improvement_iterations:
            return True

        return False

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

    async def _save_thought_for_analytics(self, thought: Thought):
        """Save thought to analytics storage if provided."""
        if self.config.analytics_storage is None:
            return

        try:
            # Use standard async storage protocol (all storage implementations support this)
            await self.config.analytics_storage.set(thought.id, thought)
            logger.debug(f"Saved thought {thought.id} to analytics storage")
        except Exception as e:
            logger.error(f"Failed to save thought to analytics storage: {e}")
            # Don't fail the chain execution for analytics storage errors
