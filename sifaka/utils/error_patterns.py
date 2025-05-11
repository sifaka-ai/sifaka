"""
Error handling patterns for Sifaka components.

This module is deprecated and has been consolidated into sifaka.utils.errors.
Import directly from sifaka.utils.errors instead.

## Usage Examples

```python
# Old usage (deprecated)
from sifaka.utils.error_patterns import handle_chain_error, handle_model_error

# New usage (recommended)
from sifaka.utils.errors import handle_chain_error, handle_model_error
```
"""

from sifaka.utils.errors import (
    # Error result classes
    ErrorResult,
    # Component-specific error handlers
    handle_component_error,
    create_error_handler,
    handle_chain_error,
    handle_model_error,
    handle_rule_error,
    handle_critic_error,
    handle_classifier_error,
    handle_retrieval_error,
    # Error result creation functions
    create_error_result,
    create_error_result_factory,
    create_chain_error_result,
    create_model_error_result,
    create_rule_error_result,
    create_critic_error_result,
    create_classifier_error_result,
    create_retrieval_error_result,
    # Safe execution functions
    safely_execute_component_operation,
    create_safe_execution_factory,
    safely_execute_chain,
    safely_execute_model,
    safely_execute_rule,
    safely_execute_critic,
    safely_execute_classifier,
    safely_execute_retrieval,
)

# Export all the imported symbols
__all__ = [
    # Error result classes
    "ErrorResult",
    # Component-specific error handlers
    "handle_component_error",
    "create_error_handler",
    "handle_chain_error",
    "handle_model_error",
    "handle_rule_error",
    "handle_critic_error",
    "handle_classifier_error",
    "handle_retrieval_error",
    # Error result creation functions
    "create_error_result",
    "create_error_result_factory",
    "create_chain_error_result",
    "create_model_error_result",
    "create_rule_error_result",
    "create_critic_error_result",
    "create_classifier_error_result",
    "create_retrieval_error_result",
    # Safe execution functions
    "safely_execute_component_operation",
    "create_safe_execution_factory",
    "safely_execute_chain",
    "safely_execute_model",
    "safely_execute_rule",
    "safely_execute_critic",
    "safely_execute_classifier",
    "safely_execute_retrieval",
]
