from typing import Any, List
"""
Core Model Provider Implementation

This module provides the ModelProviderCore class which is the main interface
for model providers, delegating to specialized components for better separation of concerns.

## Overview
The ModelProviderCore class implements a component-based architecture for model providers,
delegating functionality to specialized managers and services. It serves as the foundation
for all model provider implementations in Sifaka, providing a consistent interface while
allowing for provider-specific customization.

## Components
- **ModelProviderCore**: Main provider class that delegates to specialized components
- **ClientManager**: Manages API client creation and lifecycle
- **TokenCounterManager**: Manages token counting functionality
- **TracingManager**: Manages tracing and logging
- **GenerationService**: Handles text generation and error handling

## Usage Examples
```python
# Create a custom provider that extends ModelProviderCore
class MyProvider(ModelProviderCore):
    def _create_default_client(self) -> APIClient:
        return MyAPIClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        return MyTokenCounter(model=self.model_name)

# Use the provider
provider = MyProvider(model_name="my-model")
response = provider.generate("Hello, world!") if provider else ""
```

## Error Handling
The module implements standardized error handling patterns:
- Input validation with clear error messages
- Structured error recording and propagation
- Consistent error types for different failure modes
- Detailed error metadata for debugging

## Configuration
The module uses the standardized configuration approach from utils/config.py,
with provider-specific extensions as needed.
"""
from .provider import ModelProviderCore
from .state import create_model_state
from .initialization import initialize_resources, release_resources
from .generation import process_input
from .token_counting import count_tokens_impl
from .error_handling import record_error
from .utils import update_statistics, update_token_count_statistics
__all__: List[Any] = ['ModelProviderCore', 'create_model_state',
    'initialize_resources', 'release_resources', 'process_input',
    'count_tokens_impl', 'record_error', 'update_statistics',
    'update_token_count_statistics']
