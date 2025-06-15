"""Validation node for Sifaka workflow.

This module contains the ValidateNode which handles the validation phase
of the Sifaka workflow with parallel validator execution.
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
    from sifaka.graph.nodes.critique import CritiqueNode
    from sifaka.graph.nodes.generate import GenerateNode

logger = get_logger(__name__)


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
    ) -> Union["CritiqueNode", "GenerateNode", End[SifakaThought]]:
        """Execute parallel validation for the current thought.

        Args:
            ctx: Graph run context containing thought state and dependencies

        Returns:
            Next node based on validation results:
            - CritiqueNode if validation passed
            - GenerateNode if validation failed but should continue
            - End if max iterations reached or other termination condition
        """
        from sifaka.graph.nodes.critique import CritiqueNode

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
