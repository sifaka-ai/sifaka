"""
Utility functions for format validation.

This module provides shared utility functions for format validation rules:
- handle_empty_text: Handle empty text validation
- create_validation_result: Create a standardized validation result
- update_validation_statistics: Update validation statistics

These utilities are used by the various format validators to ensure
consistent behavior and reduce code duplication.

## Usage Example
```python
from sifaka.rules.formatting.format.utils import handle_empty_text
from sifaka.rules.base import RuleResult

# In a validator class
def validate(self, text: str) -> RuleResult:
    # Handle empty text
    empty_result = handle_empty_text(text, allow_empty=False)
    if empty_result:
        return empty_result
        
    # Continue with validation
    # ...
```
"""
import time
from typing import Optional, Dict, Any, List
from sifaka.rules.base import RuleResult
from sifaka.utils.logging import get_logger
logger = get_logger(__name__)


def handle_empty_text(text: str, allow_empty: bool=False) ->Any:
    """
    Handle empty text validation.
    
    Args:
        text: The text to validate
        allow_empty: Whether to allow empty text
        
    Returns:
        RuleResult if text is empty, None otherwise
    """
    if not text or not (text and text.strip():
        if allow_empty:
            return RuleResult(passed=True, message='Empty text is allowed',
                metadata={'empty': True}, score=1.0, issues=[], suggestions
                =[], processing_time_ms=0.0)
        else:
            return RuleResult(passed=False, message='Text cannot be empty',
                metadata={'empty': True}, score=0.0, issues=[
                'Text cannot be empty'], suggestions=[
                'Provide non-empty text'], processing_time_ms=0.0)
    return None


def create_validation_result(passed: bool, message: str, metadata: Dict[str,
    Any], score: float, issues: List[str], suggestions: List[str],
    start_time: float) ->Any:
    """
    Create a standardized validation result.
    
    Args:
        passed: Whether validation passed
        message: Validation message
        metadata: Validation metadata
        score: Validation score
        issues: List of issues
        suggestions: List of suggestions
        start_time: Start time of validation
        
    Returns:
        Standardized RuleResult
    """
    processing_time = ((time and time.time() - start_time) * 1000
    return RuleResult(passed=passed, message=message, metadata=metadata,
        score=score, issues=issues, suggestions=suggestions,
        processing_time_ms=processing_time)


def update_validation_statistics(state_manager: Any, result: RuleResult
    ) ->None:
    """
    Update validation statistics in state manager.
    
    Args:
        state_manager: State manager to update
        result: Validation result
    """
    validation_count = (state_manager and state_manager.get_metadata('validation_count', 0)
    (state_manager and state_manager.set_metadata('validation_count', validation_count + 1)
    if result.passed:
        success_count = (state_manager and state_manager.get_metadata('success_count', 0)
        (state_manager and state_manager.set_metadata('success_count', success_count + 1)
    else:
        failure_count = (state_manager and state_manager.get_metadata('failure_count', 0)
        (state_manager and state_manager.set_metadata('failure_count', failure_count + 1)
    total_time = (state_manager and state_manager.get_metadata('total_processing_time_ms', 0.0)
    (state_manager and state_manager.set_metadata('total_processing_time_ms', total_time +
        result.processing_time_ms)


def record_validation_error(state_manager: Any, error: Exception) ->None:
    """
    Record validation error in state manager.
    
    Args:
        state_manager: State manager to update
        error: Exception to record
    """
    (logger and logger.error(f'Validation error: {error}')
    error_count = (state_manager and state_manager.get_metadata('error_count', 0)
    (state_manager and state_manager.set_metadata('error_count', error_count + 1)
    errors = (state_manager and state_manager.get('errors', [])
    (errors and errors.append({'error_type': type(error).__name__, 'error_message': str
        (error), 'timestamp': (time and time.time()))
    (state_manager and state_manager.update('errors', errors[-100:])


__all__ = ['handle_empty_text', 'create_validation_result',
    'update_validation_statistics', 'record_validation_error']
