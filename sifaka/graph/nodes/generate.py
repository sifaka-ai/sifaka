"""Text generation node for Sifaka workflow.

This module contains the GenerateNode which handles the text generation phase
of the Sifaka workflow using PydanticAI agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from pydantic_graph import GraphRunContext

from sifaka.core.thought import SifakaThought, ValidationContext
from sifaka.graph.nodes.base import SifakaNode
from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from sifaka.graph.nodes.validate import ValidateNode

logger = get_logger(__name__)


@dataclass
class GenerateNode(SifakaNode):
    """Graph node for text generation using PydanticAI agents.

    This node handles the generation phase of the Sifaka workflow:
    1. Builds context from thought history and validation/critique feedback
    2. Runs the PydanticAI generator agent
    3. Updates the thought with generation results
    4. Proceeds to validation

    The node is designed to be stateless and reusable across different thoughts.
    """

    async def run(self, ctx: GraphRunContext[SifakaThought, Any]) -> "ValidateNode":
        """Execute text generation for the current thought.

        Args:
            ctx: Graph run context containing thought state and dependencies

        Returns:
            ValidateNode to proceed to validation phase
        """
        from sifaka.graph.nodes.validate import ValidateNode

        logger.log_thought_event(
            "generation_start",
            ctx.state.id,
            iteration=ctx.state.iteration,
            extra={
                "model": str(ctx.deps.generator_agent.model),
                "prompt_length": len(ctx.state.prompt),
            },
        )

        # Build context from thought history and feedback
        context = self._build_context(ctx.state, ctx.deps)

        # Get message history for conversation continuity
        message_history = self._get_message_history(ctx.state)

        try:
            # Use PydanticAI agent for generation
            with logger.performance_timer(
                "generation", thought_id=ctx.state.id, iteration=ctx.state.iteration
            ):
                result = await ctx.deps.generator_agent.run(
                    context, message_history=message_history
                )

            # Update thought with generation results
            ctx.state.add_generation(
                text=result.output,
                model=str(ctx.deps.generator_agent.model),
                pydantic_result=result,
            )

            logger.log_thought_event(
                "generation_complete",
                ctx.state.id,
                iteration=ctx.state.iteration,
                extra={
                    "model": str(ctx.deps.generator_agent.model),
                    "text_length": len(result.output),
                    "cost": getattr(result, "cost", None),
                },
            )

            # Track any tool calls made during generation
            # Note: Tool call tracking would be handled by PydanticAI
            # This is a placeholder for future tool call extraction

        except Exception as e:
            # Handle generation errors gracefully
            error_text = f"Generation failed: {str(e)}"
            ctx.state.add_generation(
                text=error_text,
                model=str(ctx.deps.generator_agent.model),
                pydantic_result=None,  # No result object for errors
            )

            logger.error(
                "Generation failed",
                extra={
                    "thought_id": ctx.state.id,
                    "iteration": ctx.state.iteration,
                    "model": str(ctx.deps.generator_agent.model),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

        return ValidateNode()

    def _build_context(self, thought: SifakaThought, deps) -> str:
        """Build context prompt from thought history and feedback with validation awareness.

        Args:
            thought: The current thought state
            deps: SifakaDependencies containing configuration options

        Returns:
            Formatted context string for the generation agent with validation prioritization
        """
        parts = [f"Generate content for: {thought.prompt}"]

        if thought.iteration > 0:
            parts.append(f"Iteration {thought.iteration}:")

            # Add previous generation for context continuity
            if thought.generations:
                last_generation = thought.generations[-1]
                parts.extend(
                    [
                        "",
                        "Previous attempt:",
                        last_generation.text,
                        "",
                        "Improve the above based on the following feedback:",
                    ]
                )

            # Include validation results with weighting
            prev_iteration = thought.iteration - 1
            prev_validations = [v for v in thought.validations if v.iteration == prev_iteration]

            if prev_validations and deps.always_include_validation_results:
                # Apply validation weight to determine prominence
                validation_prominence = self._get_prominence_level(deps.validation_weight)
                parts.append(
                    f"{validation_prominence} Validation Results (Weight: {deps.validation_weight:.0%}):"
                )

                for validation in prev_validations:
                    status = "✅ PASSED" if validation.passed else "❌ FAILED"
                    parts.append(f"- {validation.validator}: {status}")
                    if not validation.passed and "message" in validation.details:
                        parts.append(f"  → {validation.details['message']}")
                parts.append("")  # Add spacing

            # Get validation context for priority-aware prompting (for failures only)
            validation_context = ValidationContext.extract_constraints(thought)

            if validation_context:
                # Use validation-aware formatting for failures
                feedback_categories = ValidationContext.categorize_feedback(validation_context)
                priority_notice = ValidationContext.create_validation_priority_notice(
                    validation_context
                )
                validation_issues = ValidationContext.format_validation_issues(
                    validation_context, feedback_categories
                )

                if priority_notice:
                    parts.append(priority_notice.strip())
                if validation_issues:
                    parts.append(validation_issues.strip())
            elif not deps.always_include_validation_results:
                # Fallback to simple validation failure listing (only if not already included above)
                failures = [
                    v for v in thought.validations if not v.passed and v.iteration == prev_iteration
                ]
                if failures:
                    parts.append("Fix these validation issues:")
                    for failure in failures[-3:]:  # Limit to last 3 failures
                        parts.append(f"- {failure.validator}: {failure.details}")

            # Add critic suggestions from previous iteration with weighting
            prev_iteration = thought.iteration - 1
            suggestions = [
                c for c in thought.critiques if c.iteration == prev_iteration and c.suggestions
            ]
            if suggestions:
                # Apply critic weight to determine prominence
                critic_prominence = self._get_prominence_level(deps.critic_weight)

                # Get header from validation context if available
                if validation_context:
                    feedback_categories = ValidationContext.categorize_feedback(validation_context)
                    critic_header = feedback_categories["critic_header"]
                    parts.append(
                        f"{critic_prominence} {critic_header} (Weight: {deps.critic_weight:.0%})"
                    )
                else:
                    parts.append(
                        f"{critic_prominence} Critic Suggestions (Weight: {deps.critic_weight:.0%}):"
                    )

                for critique in suggestions:
                    for suggestion in critique.suggestions[:2]:  # Limit to 2 per critic
                        parts.append(f"- {suggestion}")

        return "\n".join(parts)

    def _get_message_history(self, thought: SifakaThought):
        """Extract message history for conversation continuity.

        Args:
            thought: The current thought state

        Returns:
            Message history for PydanticAI agent, or None if no history

        Note:
            We don't pass serialized message history back to PydanticAI since
            the messages have been converted to dicts/strings and are no longer
            proper PydanticAI message objects. Instead, we rely on context building
            to provide conversation continuity.
        """
        # Always return None - conversation continuity is handled through context building
        # This avoids the issue of trying to pass serialized message objects back to PydanticAI
        return None
