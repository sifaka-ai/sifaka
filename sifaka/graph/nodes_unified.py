"""Unified graph nodes for Sifaka workflow orchestration.

This module contains all the core graph nodes in a single file to avoid
circular import issues with PydanticAI's graph type resolution system.

Nodes:
- GenerateNode: Text generation using PydanticAI agents
- ValidateNode: Parallel validation execution
- CritiqueNode: Parallel critic execution
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Union

from pydantic_graph import End, GraphRunContext

from sifaka.core.thought import SifakaThought, ValidationContext
from sifaka.graph.base_node import SifakaNode
from sifaka.utils.logging import get_logger

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

    async def run(self, ctx: GraphRunContext[SifakaThought, Any]) -> ValidateNode:
        """Execute text generation for the current thought.

        Args:
            ctx: Graph run context containing thought state and dependencies

        Returns:
            ValidateNode to proceed to validation phase
        """
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

    def _get_prominence_level(self, weight: float) -> str:
        """Get prominence indicator based on weight.

        Args:
            weight: Weight value between 0.0 and 1.0

        Returns:
            Prominence indicator string
        """
        if weight >= 0.7:
            return "🔥 CRITICAL"
        elif weight >= 0.5:
            return "⚠️ IMPORTANT"
        elif weight >= 0.3:
            return "📝 MODERATE"
        else:
            return "💡 MINOR"

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


@dataclass
class ValidateNode(SifakaNode):
    """Graph node for parallel validation execution.

    This node handles the validation phase of the Sifaka workflow:
    1. Runs all configured validators in parallel
    2. Aggregates validation results
    3. Determines next step based on validation outcomes
    4. Updates thought with validation results

    Decision logic:
    - If all validations pass: proceed to critique
    - If any validation fails: check if should continue iterating
    - If max iterations reached: end workflow
    """

    async def run(
        self, ctx: GraphRunContext[SifakaThought, Any]
    ) -> Union[CritiqueNode, GenerateNode, End[SifakaThought]]:
        """Execute parallel validation for the current thought.

        Args:
            ctx: Graph run context containing thought state and dependencies

        Returns:
            Next node based on validation results:
            - CritiqueNode if validation passed
            - GenerateNode if validation failed but should continue
            - End if max iterations reached or other termination condition
        """
        logger.log_thought_event(
            "validation_start",
            ctx.state.id,
            iteration=ctx.state.iteration,
            extra={
                "validator_count": len(ctx.deps.validators),
                "text_length": len(ctx.state.current_text) if ctx.state.current_text else 0,
            },
        )

        text = ctx.state.current_text
        if not text:
            # No text to validate - this shouldn't happen in normal flow
            ctx.state.add_validation(
                validator="system",
                passed=False,
                details={"error": "No text available for validation"},
            )
            logger.error(
                "No text available for validation",
                extra={"thought_id": ctx.state.id, "iteration": ctx.state.iteration},
            )
            return End(ctx.state)

        # Run all validators in parallel
        validation_tasks = []
        for validator in ctx.deps.validators:
            task = self._run_validator(validator, text, ctx.state)
            validation_tasks.append(task)

        # Initialize validation results
        all_passed = True
        failed_validators = []

        # Wait for all validations to complete
        if validation_tasks:
            with logger.performance_timer(
                "validation", thought_id=ctx.state.id, iteration=ctx.state.iteration
            ):
                results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Process results and update thought
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle validator exceptions
                    validator_name = getattr(ctx.deps.validators[i], "name", f"validator_{i}")
                    ctx.state.add_validation(
                        validator=validator_name,
                        passed=False,
                        details={"error": str(result), "exception_type": type(result).__name__},
                    )
                    all_passed = False
                    failed_validators.append(validator_name)
                else:
                    validator_name, passed, details = result
                    ctx.state.add_validation(
                        validator=validator_name, passed=passed, details=details
                    )
                    if not passed:
                        all_passed = False
                        failed_validators.append(validator_name)

        logger.log_thought_event(
            "validation_complete",
            ctx.state.id,
            iteration=ctx.state.iteration,
            extra={
                "all_passed": all_passed,
                "failed_validators": failed_validators,
                "validator_count": len(ctx.deps.validators),
            },
        )

        # Always proceed to critique phase regardless of validation results
        # Critics can provide valuable feedback even when validations fail
        return CritiqueNode()

    async def _run_validator(self, validator, text: str, thought: SifakaThought):
        """Run a single validator with comprehensive error handling.

        Args:
            validator: The validator instance to run
            text: The text to validate
            thought: The current thought (for context-aware validators)

        Returns:
            Tuple of (validator_name, passed, details)
        """
        validator_name = getattr(validator, "name", validator.__class__.__name__)

        try:
            # Try async validation first
            if hasattr(validator, "validate_async"):
                result = await validator.validate_async(thought)
            elif hasattr(validator, "validate"):
                # Fallback to sync validation (run in thread pool if needed)
                result = validator.validate(thought)
            else:
                # Invalid validator - no validate method
                return (
                    validator_name,
                    False,
                    {"error": "Validator has no validate or validate_async method"},
                )

            # Handle ValidationResult objects (the expected format)
            if hasattr(result, "passed") and hasattr(result, "metadata"):
                # This is a ValidationResult object
                passed = result.passed
                details = {
                    "message": result.message,
                    "score": result.score,
                    "issues": result.issues,
                    "suggestions": result.suggestions,
                    "metadata": result.metadata,
                    "processing_time_ms": result.processing_time_ms,
                }
            elif isinstance(result, dict):
                # Legacy dict format
                passed = result.get("passed", False)
                details = result.get("details", result)
            elif isinstance(result, bool):
                # Simple boolean result
                passed = result
                details = {"result": result}
            else:
                # Unexpected result format
                passed = False
                details = {
                    "error": f"Unexpected result format: {type(result)}",
                    "result": str(result),
                }

            return validator_name, passed, details

        except Exception as e:
            # Validator execution failed
            return validator_name, False, {"error": str(e), "exception_type": type(e).__name__}


@dataclass
class CritiqueNode(SifakaNode):
    """Graph node for parallel critic execution.

    This node handles the critique phase of the Sifaka workflow:
    1. Runs all configured critics in parallel
    2. Aggregates critique feedback and suggestions
    3. Determines whether to continue iterating
    4. Updates thought with critique results

    Decision logic:
    - If critics provide suggestions and can continue: iterate
    - If no suggestions or max iterations reached: end workflow
    """

    async def run(
        self, ctx: GraphRunContext[SifakaThought, Any]
    ) -> Union[GenerateNode, End[SifakaThought]]:
        """Execute parallel critique for the current thought.

        Args:
            ctx: Graph run context containing thought state and dependencies

        Returns:
            Next node based on critique results:
            - GenerateNode if critics suggest improvements and should continue
            - End if no improvements needed or max iterations reached
        """
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

    def _extract_suggestions(self, feedback: str) -> list[str]:
        """Extract actionable suggestions from critic feedback.

        Args:
            feedback: The raw feedback text from the critic

        Returns:
            List of extracted suggestions
        """
        if "NO_IMPROVEMENTS_NEEDED" in feedback:
            return []

        # Simple extraction logic - look for bullet points or numbered lists
        suggestions = []
        lines = feedback.split("\n")

        for line in lines:
            line = line.strip()
            # Look for bullet points, numbers, or suggestion keywords
            if (
                line.startswith("-")
                or line.startswith("*")
                or line.startswith("•")
                or any(line.startswith(f"{i}.") for i in range(1, 10))
                or "suggest" in line.lower()
                or "recommend" in line.lower()
                or "improve" in line.lower()
            ):

                # Clean up the suggestion text
                cleaned = line.lstrip("-*•0123456789. ").strip()
                if cleaned and len(cleaned) > 10:  # Ignore very short suggestions
                    suggestions.append(cleaned)

        # Limit to 3 suggestions per critic
        return suggestions[:3]
