"""SifakaThought: Core state container for Sifaka.

This module implements the central thought model that tracks the complete audit trail
of a thought's evolution through the Sifaka workflow. Based on the migration specification,
this is a pure Pydantic model designed for complete type safety and rich observability.

The SifakaThought model captures:
- Complete generation history with PydanticAI metadata
- Validation results from all validators
- Critique feedback and suggestions
- Tool call records with execution metrics
- Thought connectivity for conversations
- Research technique tracking
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ToolCall(BaseModel):
    """Record of a tool call made during generation.

    Tracks tool execution with timing and result information for observability.
    """

    iteration: int
    tool_name: str
    args: Dict[str, Any]
    result: Any
    execution_time: float  # in seconds
    timestamp: datetime


class Generation(BaseModel):
    """Record of a text generation operation.

    Captures the generated text along with PydanticAI metadata for complete traceability.
    """

    iteration: int
    text: str
    model: str
    timestamp: datetime
    conversation_history: List[Union[Dict, str]] = Field(
        default_factory=list,
        description="Complete conversation history including requests and responses",
    )
    cost: Optional[float] = None
    usage: Optional[Dict] = None


class ValidationResult(BaseModel):
    """Result of a validation operation.

    Records validation outcome with detailed information for debugging and improvement.
    """

    iteration: int
    validator: str
    passed: bool
    details: Dict[str, Any]
    timestamp: datetime


class CritiqueResult(BaseModel):
    """Result of a critique operation.

    Captures critic feedback with structured suggestions for iterative improvement.
    Enhanced with rich metadata for observability and debugging.
    """

    iteration: int
    critic: str
    feedback: str
    suggestions: List[str]
    timestamp: datetime

    # Rich metadata for observability
    confidence: Optional[float] = None  # 0.0 to 1.0
    reasoning: Optional[str] = None
    needs_improvement: bool = True

    # Critic-specific metadata
    critic_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Performance metrics
    processing_time_ms: Optional[float] = None
    model_name: Optional[str] = None

    # Research paper information
    paper_reference: Optional[str] = None
    methodology: Optional[str] = None

    # Tool usage tracking
    tools_used: List[str] = Field(default_factory=list)
    retrieval_context: Optional[Dict[str, Any]] = None


class SifakaThought(BaseModel):
    """Central state container for Sifaka with complete audit trail.

    This is the core model that tracks a thought's complete evolution through
    the Sifaka workflow. It maintains immutable history while allowing updates
    during active processing.

    Key features:
    - Complete audit trail of all operations
    - PydanticAI integration metadata
    - Thought connectivity for conversations
    - Research technique tracking
    - Type-safe operations

    Example:
        ```python
        thought = SifakaThought(
            prompt="Explain renewable energy",
            max_iterations=3
        )

        # Add generation result
        thought.add_generation(
            text="Renewable energy comes from...",
            model="openai:gpt-4",
            pydantic_result=result
        )

        # Add validation
        thought.add_validation("length", True, {"word_count": 150})

        # Check if should continue
        if thought.should_continue():
            thought.iteration += 1
        ```
    """

    # Identity and core prompt
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str

    # Current state
    current_text: Optional[str] = None
    final_text: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 3

    # Complete audit trail
    generations: List[Generation] = Field(default_factory=list)
    validations: List[ValidationResult] = Field(default_factory=list)
    critiques: List[CritiqueResult] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)

    # Research tracking
    techniques_applied: List[str] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Thought connectivity
    parent_thought_id: Optional[str] = None
    child_thought_ids: List[str] = Field(default_factory=list)
    conversation_id: Optional[str] = None

    def add_generation(self, text: str, model: str, pydantic_result) -> None:
        """Add a generation result to the thought's audit trail.

        Args:
            text: The generated text
            model: The model name used for generation
            pydantic_result: The PydanticAI AgentRunResult object (can be None for errors)
        """
        # Handle None pydantic_result (e.g., for error cases)
        if pydantic_result is not None:
            # Try to serialize conversation history safely - handle different PydanticAI versions
            conversation_history = []
            try:
                for msg in pydantic_result.new_messages():
                    try:
                        # Try to serialize as dict first
                        if hasattr(msg, "model_dump"):
                            conversation_history.append(msg.model_dump())
                        elif hasattr(msg, "dict"):
                            conversation_history.append(msg.dict())
                        else:
                            # Fallback to string representation (now allowed by Union type)
                            conversation_history.append(str(msg))
                    except Exception as e:
                        # If individual message serialization fails, store error info as dict
                        conversation_history.append(
                            {
                                "error": f"Failed to serialize message: {str(e)}",
                                "message_type": type(msg).__name__,
                                "message_str": str(msg),
                            }
                        )
            except Exception as e:
                # If accessing new_messages() fails, store error info
                conversation_history = [
                    {
                        "error": f"Failed to access messages: {str(e)}",
                        "result_type": type(pydantic_result).__name__,
                    }
                ]

            cost = getattr(pydantic_result, "cost", None)

            # Handle usage - it might be a method or property
            usage_attr = getattr(pydantic_result, "usage", None)
            if usage_attr is not None:
                try:
                    # If it's callable (method), call it
                    if callable(usage_attr):
                        usage_obj = usage_attr()
                    else:
                        usage_obj = usage_attr

                    # Convert to dict - try multiple methods
                    if hasattr(usage_obj, "model_dump"):
                        usage = usage_obj.model_dump()
                    elif hasattr(usage_obj, "dict"):
                        usage = usage_obj.dict()
                    elif hasattr(usage_obj, "__dict__"):
                        usage = usage_obj.__dict__.copy()
                    else:
                        # Last resort - try to convert to dict manually
                        usage = (
                            dict(usage_obj) if hasattr(usage_obj, "__iter__") else str(usage_obj)
                        )
                except Exception as e:
                    # If usage extraction fails, store error info
                    usage = {
                        "error": f"Failed to extract usage: {str(e)}",
                        "usage_type": type(usage_attr).__name__,
                        "usage_str": str(usage_attr),
                    }
            else:
                usage = None
        else:
            conversation_history = []
            cost = None
            usage = None

        self.generations.append(
            Generation(
                iteration=self.iteration,
                text=text,
                model=model,
                timestamp=datetime.now(),
                conversation_history=conversation_history,
                cost=cost,
                usage=usage,
            )
        )
        self.current_text = text
        self.updated_at = datetime.now()

    def add_validation(self, validator: str, passed: bool, details: Dict[str, Any]) -> None:
        """Add a validation result to the thought's audit trail.

        Args:
            validator: Name of the validator
            passed: Whether validation passed
            details: Detailed validation information
        """
        self.validations.append(
            ValidationResult(
                iteration=self.iteration,
                validator=validator,
                passed=passed,
                details=details,
                timestamp=datetime.now(),
            )
        )
        self.updated_at = datetime.now()

    def add_critique(
        self,
        critic: str,
        feedback: str,
        suggestions: List[str],
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        needs_improvement: bool = True,
        critic_metadata: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[float] = None,
        model_name: Optional[str] = None,
        paper_reference: Optional[str] = None,
        methodology: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        retrieval_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add critique feedback to the thought's audit trail.

        Args:
            critic: Name of the critic
            feedback: The critique feedback text
            suggestions: List of improvement suggestions
            confidence: Confidence score (0.0 to 1.0)
            reasoning: Reasoning behind the critique
            needs_improvement: Whether the text needs improvement
            critic_metadata: Critic-specific metadata
            processing_time_ms: Time taken for critique in milliseconds
            model_name: Model used for critique
            paper_reference: Research paper reference
            methodology: Methodology description
            tools_used: List of tools used during critique
            retrieval_context: Context from retrieval tools
        """
        self.critiques.append(
            CritiqueResult(
                iteration=self.iteration,
                critic=critic,
                feedback=feedback,
                suggestions=suggestions,
                timestamp=datetime.now(),
                confidence=confidence,
                reasoning=reasoning,
                needs_improvement=needs_improvement,
                critic_metadata=critic_metadata or {},
                processing_time_ms=processing_time_ms,
                model_name=model_name,
                paper_reference=paper_reference,
                methodology=methodology,
                tools_used=tools_used or [],
                retrieval_context=retrieval_context,
            )
        )
        if critic not in self.techniques_applied:
            self.techniques_applied.append(critic)
        self.updated_at = datetime.now()

    def add_tool_call(
        self, tool_name: str, args: Dict[str, Any], result: Any, execution_time: float
    ) -> None:
        """Add a tool call record to the thought's audit trail.

        Args:
            tool_name: Name of the tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool
            execution_time: Time taken to execute the tool (in seconds)
        """
        self.tool_calls.append(
            ToolCall(
                iteration=self.iteration,
                tool_name=tool_name,
                args=args,
                result=result,
                execution_time=execution_time,
                timestamp=datetime.now(),
            )
        )
        self.updated_at = datetime.now()

    def should_continue(self) -> bool:
        """Determine if the thought should continue to the next iteration.

        Returns:
            True if the thought should continue, False if it should end.

        Logic:
        - Stop if max iterations reached
        - Stop if validation passed and no critic suggestions
        - Continue otherwise
        """
        if self.iteration >= self.max_iterations:
            return False

        current_validations = [v for v in self.validations if v.iteration == self.iteration]
        if current_validations and all(v.passed for v in current_validations):
            current_critiques = [c for c in self.critiques if c.iteration == self.iteration]
            if not current_critiques or not any(c.suggestions for c in current_critiques):
                return False

        return True

    def connect_to(self, parent: "SifakaThought") -> None:
        """Connect this thought to a parent thought for conversation continuity.

        Args:
            parent: The parent thought to connect to
        """
        self.parent_thought_id = parent.id
        self.conversation_id = parent.conversation_id or parent.id
        parent.child_thought_ids.append(self.id)

    def get_current_iteration_validations(self) -> List[ValidationResult]:
        """Get validation results for the current iteration.

        Returns:
            List of validation results for the current iteration
        """
        return [v for v in self.validations if v.iteration == self.iteration]

    def get_current_iteration_critiques(self) -> List[CritiqueResult]:
        """Get critique results for the current iteration.

        Returns:
            List of critique results for the current iteration
        """
        return [c for c in self.critiques if c.iteration == self.iteration]

    def get_current_iteration_tool_calls(self) -> List[ToolCall]:
        """Get tool calls for the current iteration.

        Returns:
            List of tool calls for the current iteration
        """
        return [t for t in self.tool_calls if t.iteration == self.iteration]

    def validation_passed(self) -> bool:
        """Check if all validations passed for the current iteration.

        Returns:
            True if all current iteration validations passed, False otherwise
        """
        current_validations = self.get_current_iteration_validations()
        if not current_validations:
            # No validations means we can't say they passed
            return False

        # All validations must pass
        return all(v.passed for v in current_validations)

    def finalize(self) -> None:
        """Finalize the thought by setting final_text and marking as complete.

        This should be called when the thought processing is complete.
        """
        self.final_text = self.current_text
        self.updated_at = datetime.now()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the thought's current state.

        Returns:
            Dictionary containing summary information
        """
        return {
            "id": self.id,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "has_text": self.current_text is not None,
            "text_length": len(self.current_text) if self.current_text else 0,
            "generations_count": len(self.generations),
            "validations_count": len(self.validations),
            "critiques_count": len(self.critiques),
            "tool_calls_count": len(self.tool_calls),
            "techniques_applied": self.techniques_applied,
            "is_finalized": self.final_text is not None,
            "conversation_id": self.conversation_id,
            "has_parent": self.parent_thought_id is not None,
            "children_count": len(self.child_thought_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    # Memory Management Methods
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information for this thought.

        Returns:
            Dictionary containing memory usage statistics
        """
        import sys

        # Calculate sizes of major components
        generations_size = sum(sys.getsizeof(gen.model_dump()) for gen in self.generations)
        validations_size = sum(sys.getsizeof(val.model_dump()) for val in self.validations)
        critiques_size = sum(sys.getsizeof(crit.model_dump()) for crit in self.critiques)
        tool_calls_size = sum(sys.getsizeof(tool.model_dump()) for tool in self.tool_calls)

        # Calculate conversation history size (often the largest component)
        conversation_size = 0
        for gen in self.generations:
            if gen.conversation_history:
                conversation_size += sys.getsizeof(gen.conversation_history)
                for msg in gen.conversation_history:
                    conversation_size += sys.getsizeof(msg)

        total_size = sys.getsizeof(self.model_dump())

        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "components": {
                "generations": {
                    "count": len(self.generations),
                    "size_bytes": generations_size,
                    "size_mb": round(generations_size / (1024 * 1024), 2),
                },
                "conversation_history": {
                    "size_bytes": conversation_size,
                    "size_mb": round(conversation_size / (1024 * 1024), 2),
                },
                "validations": {
                    "count": len(self.validations),
                    "size_bytes": validations_size,
                    "size_mb": round(validations_size / (1024 * 1024), 2),
                },
                "critiques": {
                    "count": len(self.critiques),
                    "size_bytes": critiques_size,
                    "size_mb": round(critiques_size / (1024 * 1024), 2),
                },
                "tool_calls": {
                    "count": len(self.tool_calls),
                    "size_bytes": tool_calls_size,
                    "size_mb": round(tool_calls_size / (1024 * 1024), 2),
                },
            },
            "largest_component": max(
                [
                    ("generations", generations_size),
                    ("conversation_history", conversation_size),
                    ("validations", validations_size),
                    ("critiques", critiques_size),
                    ("tool_calls", tool_calls_size),
                ],
                key=lambda x: x[1],
            )[0],
        }

    def cleanup_history(
        self, keep_last_n: int = 3, preserve_current: bool = True
    ) -> Dict[str, int]:
        """Remove old history entries to save memory.

        Args:
            keep_last_n: Number of most recent entries to keep for each type
            preserve_current: Whether to always preserve current iteration data

        Returns:
            Dictionary with counts of items removed
        """
        removed_counts = {
            "generations": 0,
            "validations": 0,
            "critiques": 0,
            "tool_calls": 0,
        }

        # Determine which iterations to preserve
        if preserve_current:
            # Always keep current iteration + last N-1 iterations
            min_iteration_to_keep = max(0, self.iteration - keep_last_n + 1)
            preserve_iterations = set(range(min_iteration_to_keep, self.iteration + 1))
        else:
            # Keep only the last N iterations
            all_iterations = set()
            for gen in self.generations:
                all_iterations.add(gen.iteration)
            for val in self.validations:
                all_iterations.add(val.iteration)
            for crit in self.critiques:
                all_iterations.add(crit.iteration)
            for tool in self.tool_calls:
                all_iterations.add(tool.iteration)

            sorted_iterations = sorted(all_iterations, reverse=True)
            preserve_iterations = set(sorted_iterations[:keep_last_n])

        # Clean up generations
        original_count = len(self.generations)
        self.generations = [gen for gen in self.generations if gen.iteration in preserve_iterations]
        removed_counts["generations"] = original_count - len(self.generations)

        # Clean up validations
        original_count = len(self.validations)
        self.validations = [val for val in self.validations if val.iteration in preserve_iterations]
        removed_counts["validations"] = original_count - len(self.validations)

        # Clean up critiques
        original_count = len(self.critiques)
        self.critiques = [crit for crit in self.critiques if crit.iteration in preserve_iterations]
        removed_counts["critiques"] = original_count - len(self.critiques)

        # Clean up tool calls
        original_count = len(self.tool_calls)
        self.tool_calls = [
            tool for tool in self.tool_calls if tool.iteration in preserve_iterations
        ]
        removed_counts["tool_calls"] = original_count - len(self.tool_calls)

        self.updated_at = datetime.now()

        logger.info(
            f"Cleaned up thought {self.id} history",
            extra={
                "keep_last_n": keep_last_n,
                "preserve_current": preserve_current,
                "preserved_iterations": sorted(preserve_iterations),
                "removed_counts": removed_counts,
            },
        )

        return removed_counts

    def compress_conversation_history(self, max_messages_per_iteration: int = 10) -> int:
        """Compress conversation history by keeping only the most recent messages per iteration.

        Args:
            max_messages_per_iteration: Maximum number of messages to keep per iteration

        Returns:
            Number of messages removed
        """
        messages_removed = 0

        for generation in self.generations:
            if (
                generation.conversation_history
                and len(generation.conversation_history) > max_messages_per_iteration
            ):
                original_count = len(generation.conversation_history)
                # Keep the most recent messages
                generation.conversation_history = generation.conversation_history[
                    -max_messages_per_iteration:
                ]
                messages_removed += original_count - len(generation.conversation_history)

        if messages_removed > 0:
            self.updated_at = datetime.now()
            logger.info(
                f"Compressed conversation history for thought {self.id}",
                extra={
                    "messages_removed": messages_removed,
                    "max_messages_per_iteration": max_messages_per_iteration,
                },
            )

        return messages_removed

    def remove_large_tool_results(self, max_result_size_bytes: int = 10240) -> int:
        """Remove large tool call results to save memory.

        Args:
            max_result_size_bytes: Maximum size for tool results (default: 10KB)

        Returns:
            Number of tool results removed
        """
        import sys

        results_removed = 0

        for tool_call in self.tool_calls:
            if tool_call.result is not None:
                result_size = sys.getsizeof(tool_call.result)
                if result_size > max_result_size_bytes:
                    # Replace with summary
                    tool_call.result = {
                        "_removed_large_result": True,
                        "original_size_bytes": result_size,
                        "summary": f"Large result removed (was {result_size} bytes)",
                        "type": type(tool_call.result).__name__,
                    }
                    results_removed += 1

        if results_removed > 0:
            self.updated_at = datetime.now()
            logger.info(
                f"Removed large tool results for thought {self.id}",
                extra={
                    "results_removed": results_removed,
                    "max_result_size_bytes": max_result_size_bytes,
                },
            )

        return results_removed

    def optimize_memory(
        self,
        keep_last_n_iterations: int = 3,
        max_messages_per_iteration: int = 10,
        max_tool_result_size_bytes: int = 10240,
        preserve_current: bool = True,
    ) -> Dict[str, Any]:
        """Comprehensive memory optimization for the thought.

        Args:
            keep_last_n_iterations: Number of iterations to keep in history
            max_messages_per_iteration: Maximum conversation messages per iteration
            max_tool_result_size_bytes: Maximum size for tool results
            preserve_current: Whether to always preserve current iteration

        Returns:
            Dictionary with optimization results
        """
        # Get memory usage before optimization
        memory_before = self.get_memory_usage()

        # Apply optimizations
        removed_counts = self.cleanup_history(keep_last_n_iterations, preserve_current)
        messages_removed = self.compress_conversation_history(max_messages_per_iteration)
        tool_results_removed = self.remove_large_tool_results(max_tool_result_size_bytes)

        # Get memory usage after optimization
        memory_after = self.get_memory_usage()

        # Calculate savings
        bytes_saved = memory_before["total_size_bytes"] - memory_after["total_size_bytes"]
        mb_saved = round(bytes_saved / (1024 * 1024), 2)
        percent_saved = (
            round((bytes_saved / memory_before["total_size_bytes"]) * 100, 1)
            if memory_before["total_size_bytes"] > 0
            else 0
        )

        optimization_results = {
            "memory_before_mb": memory_before["total_size_mb"],
            "memory_after_mb": memory_after["total_size_mb"],
            "memory_saved_mb": mb_saved,
            "percent_saved": percent_saved,
            "optimizations_applied": {
                "history_cleanup": removed_counts,
                "conversation_compression": {"messages_removed": messages_removed},
                "tool_result_cleanup": {"results_removed": tool_results_removed},
            },
            "settings": {
                "keep_last_n_iterations": keep_last_n_iterations,
                "max_messages_per_iteration": max_messages_per_iteration,
                "max_tool_result_size_bytes": max_tool_result_size_bytes,
                "preserve_current": preserve_current,
            },
        }

        logger.info(
            f"Memory optimization completed for thought {self.id}", extra=optimization_results
        )

        return optimization_results

    def get_conversation_messages_for_iteration(self, iteration: int) -> List[str]:
        """Get all conversation messages (requests and responses) from a specific iteration.

        Args:
            iteration: The iteration number to get messages for

        Returns:
            List of conversation messages from that iteration (includes both requests TO model and responses FROM model)
        """
        messages = []
        iteration_generations = [g for g in self.generations if g.iteration == iteration]

        for generation in iteration_generations:
            if generation.conversation_history:
                for msg in generation.conversation_history:
                    if isinstance(msg, dict):
                        # Extract content from structured message
                        content = msg.get("content", str(msg))
                        messages.append(content)
                    else:
                        # Handle string message
                        messages.append(str(msg))

        return messages

    def get_latest_conversation_messages(self) -> List[str]:
        """Get conversation messages from the most recent iteration.

        Returns:
            List of conversation messages from the current iteration (includes both requests and responses)
        """
        return self.get_conversation_messages_for_iteration(self.iteration)

    def print_iteration_details(self, iteration: int = None) -> None:
        """Print detailed information for a specific iteration or current iteration.

        Args:
            iteration: Iteration to print details for. If None, uses current iteration.
        """
        if iteration is None:
            iteration = self.iteration

        print(f"🔄 ITERATION {iteration} DETAILS")
        print("-" * 50)

        # Show generations
        iteration_generations = [g for g in self.generations if g.iteration == iteration]
        for gen in iteration_generations:
            print(f"🤖 Model: {gen.model}")
            print(f"📄 Generated text ({len(gen.text)} chars): {gen.text[:100]}...")

            # Show conversation messages (requests and responses)
            if gen.conversation_history:
                print("💬 Conversation messages:")
                for i, msg in enumerate(gen.conversation_history):
                    content = msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg)
                    truncated = content[:150] + "..." if len(content) > 150 else content
                    print(f"   {i+1}. {truncated}")

        # Show validations
        iteration_validations = [v for v in self.validations if v.iteration == iteration]
        if iteration_validations:
            print("\n✅ Validations:")
            for validation in iteration_validations:
                status = "✅ PASSED" if validation.passed else "❌ FAILED"
                print(f"   {validation.validator}: {status}")
                if not validation.passed and "error" in validation.details:
                    print(f"      Error: {validation.details['error']}")

        # Show critiques
        iteration_critiques = [c for c in self.critiques if c.iteration == iteration]
        if iteration_critiques:
            print("\n🔍 Critics:")
            for critique in iteration_critiques:
                print(f"   🎯 {critique.critic}:")
                print(f"      Needs Improvement: {critique.needs_improvement}")
                print(f"      Confidence: {critique.confidence}")
                print(f"      Feedback: {critique.feedback[:100]}...")
                if critique.suggestions:
                    print(f"      Suggestions ({len(critique.suggestions)}):")
                    for i, suggestion in enumerate(critique.suggestions[:2]):
                        print(f"         {i+1}. {suggestion[:80]}...")

        print()

    def extract_validation_constraints(self) -> Optional[Dict[str, Any]]:
        """Extract constraint information from validation results.

        Returns:
            Dictionary containing constraint information, or None if no constraints found
        """
        current_validations = self.get_current_iteration_validations()
        if not current_validations:
            return None

        # Get all failed validations
        failed_validations = [v for v in current_validations if not v.passed]
        if not failed_validations:
            return None

        # Convert to constraint format
        constraint_data = {
            "type": "validation_failures",
            "failed_validations": [
                {
                    "validator_name": validation.validator,
                    "message": str(validation.details.get("error", "Validation failed")),
                    "suggestions": validation.details.get("suggestions", []),
                    "issues": validation.details.get("issues", []),
                    "score": validation.details.get("score", 0.0),
                    "details": validation.details,
                }
                for validation in failed_validations
            ],
            "total_failures": len(failed_validations),
        }

        return constraint_data


class ValidationContext:
    """Helper class for managing validation context in improvement operations.

    This class provides sophisticated validation-aware prompting that ensures
    validation constraints take priority over critic suggestions, preventing
    conflicts and wasted iterations.

    Key features:
    - Constraint extraction and categorization
    - Suggestion filtering to prevent conflicts
    - Priority-based prompt engineering
    - Critical vs regular failure handling
    """

    @staticmethod
    def extract_constraints(thought: SifakaThought) -> Optional[Dict[str, Any]]:
        """Extract constraint information from thought validation results.

        Args:
            thought: The SifakaThought containing validation results

        Returns:
            Dictionary containing constraint information, or None if no constraints found
        """
        return thought.extract_validation_constraints()

    @staticmethod
    def has_critical_constraints(constraints: Optional[Dict[str, Any]]) -> bool:
        """Check if constraints contain critical failures (like length limits).

        Args:
            constraints: Constraint information from extract_constraints()

        Returns:
            True if critical constraints are present, False otherwise
        """
        if not constraints or constraints.get("type") != "validation_failures":
            return False

        failed_validations = constraints.get("failed_validations", [])

        for validation in failed_validations:
            message = validation.get("message", "").lower()
            validator_name = validation.get("validator_name", "").lower()

            # Check for critical constraint indicators
            critical_keywords = [
                "too long",
                "too short",
                "maximum",
                "minimum",
                "limit",
                "characters",
                "words",
                "length",
                "exceeds",
                "below",
            ]

            if "length" in validator_name or any(
                keyword in message for keyword in critical_keywords
            ):
                return True

        return False

    @staticmethod
    def filter_conflicting_suggestions(
        suggestions: List[str], constraints: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Filter critic suggestions that conflict with validation constraints.

        Args:
            suggestions: List of critic suggestions
            constraints: Constraint information from extract_constraints()

        Returns:
            Filtered list of suggestions that don't conflict with constraints
        """
        if not constraints or not suggestions:
            return suggestions

        # Check if we have length constraints that require content reduction
        if ValidationContext._has_length_reduction_constraint(constraints):
            return ValidationContext._filter_content_expansion_suggestions(suggestions)

        # Add other constraint type filtering here as needed
        return suggestions

    @staticmethod
    def create_validation_priority_notice(constraints: Optional[Dict[str, Any]]) -> str:
        """Create priority notice for validation constraints.

        Args:
            constraints: Constraint information from extract_constraints()

        Returns:
            Priority notice string for inclusion in improvement prompts
        """
        if not constraints or constraints.get("type") != "validation_failures":
            return ""

        failed_validations = constraints.get("failed_validations", [])
        if not failed_validations:
            return ""

        # Check if any validation failure is critical
        if ValidationContext.has_critical_constraints(constraints):
            return (
                "🚨 CRITICAL VALIDATION FAILURES: The following requirements MUST be met and take "
                "ABSOLUTE PRIORITY over all other suggestions:\n\n"
            )
        else:
            return "⚠️ VALIDATION REQUIREMENTS: The text must be corrected to meet the following requirements:\n\n"

    @staticmethod
    def categorize_feedback(constraints: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Categorize feedback based on validation constraints.

        Args:
            constraints: Constraint information from extract_constraints()

        Returns:
            Dictionary with categorized feedback templates
        """
        if not constraints or constraints.get("type") != "validation_failures":
            return {
                "validation_header": "Validation Issues:",
                "critic_header": "Critic Feedback:",
                "priority_instructions": (
                    "addresses the issues identified in the feedback while maintaining the core message "
                    "and staying true to the original task"
                ),
            }

        # Check if any validation failure is critical
        if ValidationContext.has_critical_constraints(constraints):
            return {
                "validation_header": "🚨 CRITICAL VALIDATION REQUIREMENTS (MUST BE ADDRESSED FIRST):",
                "critic_header": "Secondary Feedback (IGNORE if it conflicts with validation requirements):",
                "priority_instructions": (
                    "IMMEDIATELY addresses the validation requirements listed above. "
                    "These requirements take ABSOLUTE PRIORITY over all other feedback. "
                    "Only consider other suggestions if they don't conflict with validation requirements"
                ),
            }
        else:
            return {
                "validation_header": "⚠️ VALIDATION REQUIREMENTS:",
                "critic_header": "Additional Feedback:",
                "priority_instructions": (
                    "first addresses the validation requirements, then incorporates other feedback "
                    "while maintaining the core message and staying true to the original task"
                ),
            }

    @staticmethod
    def format_validation_issues(
        constraints: Dict[str, Any], feedback_categories: Dict[str, str]
    ) -> str:
        """Format validation issues for inclusion in improvement prompts.

        Args:
            constraints: Validation context with constraint information
            feedback_categories: Categorized feedback templates

        Returns:
            Formatted validation issues string
        """
        if not constraints:
            return ""

        header = feedback_categories["validation_header"]

        if constraints.get("type") == "validation_failures":
            failed_validations = constraints.get("failed_validations", [])

            if not failed_validations:
                return f"{header}\n- Validation requirements must be met\n"

            issues_text = f"{header}\n"
            for validation in failed_validations:
                validator_name = validation.get("validator_name", "Unknown Validator")
                message = validation.get("message", "Validation failed")
                suggestions = validation.get("suggestions", [])

                issues_text += f"- {validator_name}: {message}\n"

                # Include specific suggestions if available
                if suggestions:
                    for suggestion in suggestions[:2]:  # Limit to first 2 suggestions
                        issues_text += f"  → {suggestion}\n"

            issues_text += "\n"
            return issues_text

        return f"{header}\n- General validation requirements must be met\n\n"

    @staticmethod
    def _has_length_reduction_constraint(constraints: Dict[str, Any]) -> bool:
        """Check if constraints require content reduction."""
        failed_validations = constraints.get("failed_validations", [])

        for validation in failed_validations:
            message = validation.get("message", "").lower()
            validator_name = validation.get("validator_name", "").lower()

            # Check for length reduction indicators
            if "length" in validator_name and any(
                keyword in message for keyword in ["too long", "maximum", "exceeds", "limit"]
            ):
                return True

        return False

    @staticmethod
    def _filter_content_expansion_suggestions(suggestions: List[str]) -> List[str]:
        """Filter suggestions that would expand content when length reduction is needed."""
        CONTENT_EXPANSION_PHRASES = [
            "provide more",
            "include more",
            "add more",
            "incorporate",
            "enhance",
            "expand",
            "elaborate",
            "examples",
            "case studies",
            "provide additional",
            "include additional",
            "add details",
            "give more",
            "offer more",
            "present more",
            "show more",
            "develop further",
            "go deeper",
            "more comprehensive",
            "more thorough",
            "additional information",
        ]

        filtered_suggestions = []
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            if not any(phrase in suggestion_lower for phrase in CONTENT_EXPANSION_PHRASES):
                filtered_suggestions.append(suggestion)

        return filtered_suggestions


def create_validation_context(thought: SifakaThought) -> Optional[Dict[str, Any]]:
    """Convenience function to create validation context from a thought.

    Args:
        thought: The SifakaThought containing validation results

    Returns:
        Validation context dictionary, or None if no constraints found
    """
    return ValidationContext.extract_constraints(thought)
