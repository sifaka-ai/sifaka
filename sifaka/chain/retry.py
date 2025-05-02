"""
Retry manager module for Sifaka.

This module provides the RetryManager class which is responsible for managing
retry logic and attempt tracking.
"""

from typing import Generic, TypeVar

from .executor import ChainExecutor
from .result import ChainResult

OutputType = TypeVar("OutputType")


class RetryManager:
    """
    Manages retry logic.
    
    This class is responsible for tracking attempt counts, determining when
    to stop retrying, and handling retry-related errors.
    """
    
    def __init__(self, max_attempts: int = 3):
        """
        Initialize a RetryManager instance.
        
        Args:
            max_attempts: Maximum number of attempts
        """
        self.max_attempts = max_attempts
        
    def run_with_retries(
        self, executor: ChainExecutor[OutputType], prompt: str
    ) -> ChainResult[OutputType]:
        """
        Run with retries.
        
        Args:
            executor: The executor to use for each attempt
            prompt: The original prompt
            
        Returns:
            ChainResult containing the output and validation details
            
        Raises:
            ValueError: If validation fails after max attempts
            RuntimeError: If an unexpected condition occurs
        """
        attempts = 0
        current_prompt = prompt
        last_critique_details = None
        
        while attempts < self.max_attempts:
            # Execute a single cycle
            output, validation_result, critique_details = executor.execute(current_prompt)
            
            # If validation passed, return result
            if validation_result.all_passed:
                return ChainResult(
                    output=output,
                    rule_results=validation_result.rule_results,
                    critique_details=last_critique_details,
                )
                
            # If we're on the last attempt, raise error
            if attempts >= self.max_attempts - 1:
                error_messages = executor.validator.get_error_messages(validation_result)
                raise ValueError(
                    f"Validation failed after {attempts + 1} attempts. Errors:\n"
                    + "\n".join(error_messages)
                )
                
            # If we have critique details, update for next attempt
            if critique_details:
                last_critique_details = critique_details
                feedback = executor.improver.get_feedback(critique_details)
                current_prompt = executor.feedback_formatter.create_prompt_with_feedback(
                    prompt, feedback
                )
                
            attempts += 1
            
        # Should never reach here due to while loop condition
        raise RuntimeError("Unexpected end of retry execution")
