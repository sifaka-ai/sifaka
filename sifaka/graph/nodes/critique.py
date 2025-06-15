"""Critique node for Sifaka workflow.

This module contains the CritiqueNode which handles the critique phase
of the Sifaka workflow with parallel critic execution and memory optimization.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Union, TYPE_CHECKING

from pydantic_graph import End, GraphRunContext

from sifaka.core.thought import SifakaThought
from sifaka.graph.nodes.base import SifakaNode
from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from sifaka.graph.nodes.generate import GenerateNode

logger = get_logger(__name__)


@dataclass
class CritiqueNode(SifakaNode):
    """Graph node for parallel critic execution.

    This node handles the critique phase of the Sifaka workflow:
    1. Runs all configured critics in parallel
    2. Aggregates critique feedback and suggestions
    3. Determines whether to continue iterating
    4. Updates thought with critique results
    5. Performs memory optimization if configured

    Decision logic:
    - If critics provide suggestions and can continue: iterate
    - If no suggestions or max iterations reached: end workflow
    """

    async def run(
        self, ctx: GraphRunContext[SifakaThought, Any]
    ) -> Union["GenerateNode", End[SifakaThought]]:
        """Execute parallel critique for the current thought.

        Args:
            ctx: Graph run context containing thought state and dependencies

        Returns:
            Next node based on critique results:
            - GenerateNode if critics suggest improvements and should continue
            - End if no improvements needed or max iterations reached
        """
        from sifaka.graph.nodes.generate import GenerateNode

        logger.log_thought_event(
            "critique_start",
            ctx.state.id,
            iteration=ctx.state.iteration,
            extra={
                "critic_count": len(ctx.deps.critics),
                "text_length": len(ctx.state.current_text) if ctx.state.current_text else 0,
                "never_apply_critics": ctx.deps.never_apply_critics,
                "always_apply_critics": ctx.deps.always_apply_critics,
            },
        )

        text = ctx.state.current_text
        if not text:
            # No text to critique - this shouldn't happen in normal flow
            ctx.state.finalize()
            logger.error(
                "No text available for critique",
                extra={"thought_id": ctx.state.id, "iteration": ctx.state.iteration},
            )
            return End(ctx.state)

        # Check if critics should run based on configuration
        has_suggestions = False
        failed_critics = []

        if ctx.deps.never_apply_critics:
            # Critics are disabled - skip critic execution
            logger.log_thought_event(
                "critics_skipped",
                ctx.state.id,
                iteration=ctx.state.iteration,
                extra={"reason": "never_apply_critics_enabled"},
            )
        else:
            # Run all critics in parallel
            critic_tasks = []
            for name, critic in ctx.deps.critics.items():
                task = self._run_critic(name, critic, ctx.state)
                critic_tasks.append(task)

            # Wait for all critiques to complete
            if critic_tasks:
                with logger.performance_timer(
                    "critique", thought_id=ctx.state.id, iteration=ctx.state.iteration
                ):
                    results = await asyncio.gather(*critic_tasks, return_exceptions=True)

                # Process results and update thought
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        # Handle critic exceptions
                        critic_names = list(ctx.deps.critics.keys())
                        critic_name = critic_names[i] if i < len(critic_names) else f"critic_{i}"
                        ctx.state.add_critique(
                            critic=critic_name,
                            feedback=f"Critic failed: {str(result)}",
                            suggestions=[],
                        )
                        failed_critics.append(critic_name)
                    # Note: Critics add their results directly to the thought in their critique_async method
                    # No need to process successful results here

                # Check if any critique from this iteration has suggestions
                current_critiques = ctx.state.get_current_iteration_critiques()
                for critique in current_critiques:
                    if critique.suggestions:
                        has_suggestions = True
                        break

        logger.log_thought_event(
            "critique_complete",
            ctx.state.id,
            iteration=ctx.state.iteration,
            extra={
                "has_suggestions": has_suggestions,
                "failed_critics": failed_critics,
                "critic_count": len(ctx.deps.critics),
                "total_critiques": len(ctx.state.get_current_iteration_critiques()),
            },
        )

        # Determine next step based on validation results, critique results, and iteration limits
        current_validations = ctx.state.get_current_iteration_validations()
        validation_passed = current_validations and all(v.passed for v in current_validations)

        # Check if we should continue iterating
        should_continue = False
        continue_reason = ""

        if not validation_passed and ctx.state.should_continue():
            # Validations failed and we can continue - iterate to fix validation issues
            should_continue = True
            continue_reason = "validation_failed"
        elif validation_passed and ctx.deps.always_apply_critics and ctx.state.should_continue():
            # Validations passed but always_apply_critics is enabled - force iteration
            should_continue = True
            continue_reason = "always_apply_critics_enabled"
        elif validation_passed and has_suggestions and ctx.state.should_continue():
            # Validations passed and critics suggest improvements - iterate for enhancement
            should_continue = True
            continue_reason = "critic_suggestions"

        # Check if memory optimization should be performed
        if (
            ctx.deps.auto_optimize_memory
            and ctx.state.iteration % ctx.deps.memory_optimization_interval == 0
        ):
            try:
                optimization_results = ctx.state.optimize_memory(
                    keep_last_n_iterations=ctx.deps.keep_last_n_iterations,
                    max_messages_per_iteration=ctx.deps.max_messages_per_iteration,
                    max_tool_result_size_bytes=ctx.deps.max_tool_result_size_bytes,
                    preserve_current=True,
                )
                logger.log_thought_event(
                    "memory_optimized",
                    ctx.state.id,
                    iteration=ctx.state.iteration,
                    extra=optimization_results,
                )
            except Exception as e:
                logger.error(
                    "Memory optimization failed",
                    extra={
                        "thought_id": ctx.state.id,
                        "iteration": ctx.state.iteration,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )

        if should_continue:
            # Continue to next iteration
            ctx.state.iteration += 1
            logger.log_thought_event(
                "iteration_continue",
                ctx.state.id,
                iteration=ctx.state.iteration,
                extra={"reason": continue_reason},
            )
            return GenerateNode()
        else:
            # Finalize - either max iterations reached or no improvements needed
            ctx.state.finalize()
            finalize_reason = (
                "max_iterations"
                if ctx.state.iteration >= ctx.state.max_iterations
                else "no_improvements_needed"
            )
            logger.log_thought_event(
                "thought_finalized",
                ctx.state.id,
                iteration=ctx.state.iteration,
                extra={"reason": finalize_reason},
            )
            return End(ctx.state)

    async def _run_critic(self, name: str, critic, thought: SifakaThought):
        """Run a single critic with error handling.

        Args:
            name: Name of the critic
            critic: The BaseCritic instance
            thought: The current thought (for context)

        Returns:
            None (critics add results directly to the thought)
        """
        try:
            # Run the critic - it will add results directly to the thought
            await critic.critique_async(thought)

        except Exception as e:
            # Critic execution failed - add error critique
            thought.add_critique(
                critic=name,
                feedback=f"Critic failed: {str(e)}",
                suggestions=[],
                confidence=0.0,
                reasoning=f"Error occurred during critique: {str(e)}",
                needs_improvement=False,
                critic_metadata={"error": str(e), "error_type": type(e).__name__},
                model_name=getattr(critic, "model_name", "unknown"),
                paper_reference=getattr(critic, "paper_reference", "unknown"),
                methodology=getattr(critic, "methodology", "unknown"),
            )

    def _build_critique_prompt(self, text: str, thought: SifakaThought) -> str:
        """Build critique prompt with context.

        Args:
            text: The text to critique
            thought: The current thought for context

        Returns:
            Formatted critique prompt
        """
        parts = [
            f"Critique this text and suggest specific improvements:",
            f"",
            f"Original prompt: {thought.prompt}",
            f"Current iteration: {thought.iteration}",
            f"",
            f"Text to critique:",
            text,
            f"",
            f"If no improvements are needed, respond with 'NO_IMPROVEMENTS_NEEDED'.",
            f"Otherwise, provide 1-3 specific, actionable suggestions.",
        ]

        # Add validation context if available
        current_validations = thought.get_current_iteration_validations()
        if current_validations:
            failed_validations = [v for v in current_validations if not v.passed]
            if failed_validations:
                parts.extend(
                    [
                        f"",
                        f"Note: The following validations failed:",
                    ]
                )
                for validation in failed_validations:
                    parts.append(f"- {validation.validator}: {validation.details}")

        return "\n".join(parts)
