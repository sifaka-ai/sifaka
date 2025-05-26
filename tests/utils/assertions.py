"""Custom assertions for Sifaka testing.

This module provides specialized assertion functions for validating
Sifaka components and their behavior.
"""

from datetime import datetime
from typing import Any, List, Optional

from sifaka.core.thought import CriticFeedback, Thought, ValidationResult


def assert_thought_valid(
    thought: Thought,
    expected_text: Optional[str] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    required_fields: Optional[List[str]] = None,
) -> None:
    """Assert that a Thought object is valid and meets criteria."""
    # Basic structure validation
    assert isinstance(thought, Thought), f"Expected Thought, got {type(thought)}"
    assert thought.id is not None and len(thought.id) > 0, "Thought must have a valid ID"
    assert thought.prompt is not None and len(thought.prompt) > 0, "Thought must have a prompt"
    assert (
        isinstance(thought.iteration, int) and thought.iteration >= 0
    ), "Iteration must be non-negative integer"
    assert isinstance(thought.timestamp, datetime), "Timestamp must be datetime object"

    # Text validation
    if thought.text is not None:
        if expected_text:
            assert (
                thought.text == expected_text
            ), f"Expected text '{expected_text}', got '{thought.text}'"

        if min_length:
            assert (
                len(thought.text) >= min_length
            ), f"Text too short: {len(thought.text)} < {min_length}"

        if max_length:
            assert (
                len(thought.text) <= max_length
            ), f"Text too long: {len(thought.text)} > {max_length}"

    # Required fields validation
    if required_fields:
        for field in required_fields:
            assert hasattr(thought, field), f"Thought missing required field: {field}"
            value = getattr(thought, field)
            assert value is not None, f"Required field {field} is None"


def assert_validation_results(
    thought: Thought,
    expected_count: Optional[int] = None,
    expected_passed: Optional[bool] = None,
    validator_names: Optional[List[str]] = None,
) -> None:
    """Assert validation results meet expectations."""
    assert thought.validation_results is not None, "Thought must have validation results"

    results = thought.validation_results

    if expected_count is not None:
        assert (
            len(results) == expected_count
        ), f"Expected {expected_count} validation results, got {len(results)}"

    if validator_names:
        for name in validator_names:
            assert name in results, f"Missing validation result for validator: {name}"

    if expected_passed is not None:
        all_passed = all(result.passed for result in results.values())
        if expected_passed:
            assert (
                all_passed
            ), f"Expected all validations to pass, but some failed: {[name for name, result in results.items() if not result.passed]}"
        else:
            assert not all_passed, "Expected some validations to fail, but all passed"

    # Validate structure of each result
    for name, result in results.items():
        assert isinstance(result, ValidationResult), f"Invalid validation result type for {name}"
        assert isinstance(
            result.passed, bool
        ), f"Validation result 'passed' must be boolean for {name}"
        assert isinstance(
            result.message, str
        ), f"Validation result 'message' must be string for {name}"
        # ValidationResult stores validator name in metadata, not as a direct field
        if hasattr(result, "metadata") and result.metadata:
            validator_name = result.metadata.get("validator", name)
            assert (
                validator_name == name
            ), f"Validator name mismatch: expected {name}, got {validator_name}"


def assert_critic_feedback(
    thought: Thought,
    expected_count: Optional[int] = None,
    expected_improvement: Optional[bool] = None,
    critic_names: Optional[List[str]] = None,
) -> None:
    """Assert critic feedback meets expectations."""
    assert thought.critic_feedback is not None, "Thought must have critic feedback"

    feedback_list = thought.critic_feedback

    if expected_count is not None:
        assert (
            len(feedback_list) == expected_count
        ), f"Expected {expected_count} critic feedback items, got {len(feedback_list)}"

    if critic_names:
        feedback_names = [fb.critic_name for fb in feedback_list]
        for name in critic_names:
            assert name in feedback_names, f"Missing critic feedback for: {name}"

    # Validate structure of each feedback
    for feedback in feedback_list:
        assert isinstance(feedback, CriticFeedback), "Invalid critic feedback type"
        assert isinstance(feedback.critic_name, str), "Critic name must be string"
        assert isinstance(feedback.feedback, dict), "Feedback must be dict"
        assert isinstance(feedback.confidence, (int, float)), "Confidence must be numeric"

        if hasattr(feedback, "score") and feedback.score is not None:
            assert (
                0.0 <= feedback.score <= 1.0
            ), f"Score must be between 0 and 1, got {feedback.score}"


def assert_performance_within_bounds(
    execution_time: float,
    memory_usage: Optional[float] = None,
    max_time: float = 10.0,
    max_memory: float = 200.0,
    operation_name: str = "operation",
) -> None:
    """Assert that performance metrics are within acceptable bounds."""
    assert execution_time >= 0, f"Execution time cannot be negative: {execution_time}"
    assert (
        execution_time <= max_time
    ), f"{operation_name} took too long: {execution_time:.2f}s > {max_time}s"

    if memory_usage is not None:
        assert memory_usage >= 0, f"Memory usage cannot be negative: {memory_usage}"
        assert (
            memory_usage <= max_memory
        ), f"{operation_name} used too much memory: {memory_usage:.1f}MB > {max_memory}MB"


def assert_chain_execution_success(
    result: Thought,
    expected_iterations: Optional[int] = None,
    min_iterations: int = 0,
    max_iterations: int = 10,
) -> None:
    """Assert that chain execution completed successfully."""
    assert_thought_valid(result)

    # Check iterations
    assert (
        result.iteration >= min_iterations
    ), f"Too few iterations: {result.iteration} < {min_iterations}"
    assert (
        result.iteration <= max_iterations
    ), f"Too many iterations: {result.iteration} > {max_iterations}"

    if expected_iterations is not None:
        assert (
            result.iteration == expected_iterations
        ), f"Expected {expected_iterations} iterations, got {result.iteration}"

    # Check that we have generated text
    assert result.text is not None and len(result.text) > 0, "Chain must produce non-empty text"


def assert_async_performance(
    concurrent_results: List[Any],
    sequential_time: float,
    concurrent_time: float,
    min_speedup: float = 1.2,
) -> None:
    """Assert that async execution provides performance benefits."""
    assert len(concurrent_results) > 0, "Must have concurrent results to compare"

    # Check that all results are valid
    for result in concurrent_results:
        if isinstance(result, Thought):
            assert_thought_valid(result)

    # Check performance improvement
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
    assert (
        speedup >= min_speedup
    ), f"Insufficient speedup: {speedup:.2f}x < {min_speedup}x (sequential: {sequential_time:.2f}s, concurrent: {concurrent_time:.2f}s)"


def assert_error_handling(
    error_occurred: bool,
    expected_error_type: Optional[type] = None,
    error_message: Optional[str] = None,
    should_recover: bool = False,
) -> None:
    """Assert proper error handling behavior."""
    if should_recover:
        assert not error_occurred, "Operation should have recovered from error but didn't"
    else:
        assert error_occurred, "Expected error to occur but operation succeeded"

    # Additional error validation would be done in the calling test
    # This is a placeholder for more sophisticated error checking


def assert_storage_consistency(
    storage: Any, thought: Thought, operation: str = "save_and_load"
) -> None:
    """Assert that storage operations maintain data consistency."""
    if operation == "save_and_load":
        # Save the thought
        storage.set(thought.id, thought)

        # Load it back
        loaded_thought = storage.get(thought.id)

        # Verify consistency
        assert loaded_thought is not None, "Failed to load saved thought"
        assert loaded_thought.id == thought.id, "ID mismatch after save/load"
        assert loaded_thought.text == thought.text, "Text mismatch after save/load"
        assert loaded_thought.prompt == thought.prompt, "Prompt mismatch after save/load"


def assert_concurrent_safety(
    results: List[Any], expected_count: int, operation_name: str = "concurrent operation"
) -> None:
    """Assert that concurrent operations completed safely."""
    assert (
        len(results) == expected_count
    ), f"{operation_name}: expected {expected_count} results, got {len(results)}"

    # Check for any None results (indicating failures)
    none_count = sum(1 for result in results if result is None)
    assert none_count == 0, f"{operation_name}: {none_count} operations returned None"

    # Check for unique IDs if results are Thoughts
    if results and isinstance(results[0], Thought):
        ids = [result.id for result in results]
        unique_ids = set(ids)
        assert len(unique_ids) == len(
            ids
        ), f"{operation_name}: duplicate IDs found in concurrent results"
