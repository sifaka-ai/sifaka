"""Central constants and configuration values for Sifaka.

This module defines all constants used throughout the Sifaka framework.
Constants are organized into logical groups for easy reference and
maintenance. All values here are immutable and should not be changed
at runtime.

## Constant Categories:

1. **Model Providers**: Supported LLM provider identifiers
2. **Default Values**: Sensible defaults for common parameters
3. **Limits**: Boundary values for validation
4. **Message Formats**: API communication constants
5. **Error Messages**: Standardized error message templates
6. **Plugin System**: Entry points for extensibility

## Usage:

    >>> from sifaka.core.constants import DEFAULT_MODEL, DEFAULT_CRITIC
    >>> config = Config(model=DEFAULT_MODEL)
    >>> critics = [DEFAULT_CRITIC]

## Design Principles:

- All constants are UPPER_CASE with underscores
- Related constants are grouped with common prefixes
- Error messages include placeholders for dynamic values
- Limits are defined in pairs (MIN/MAX) for validation
- Defaults are chosen for the most common use case

Note:
    These constants are used for internal consistency. Users typically
    don't need to import them directly unless building extensions.
"""

# Model providers
# These constants identify supported LLM providers for automatic detection
PROVIDER_OPENAI = "openai"  # OpenAI models (GPT-3.5, GPT-4, etc.)
PROVIDER_ANTHROPIC = "anthropic"  # Anthropic models (Claude family)
PROVIDER_GEMINI = "gemini"  # Google Gemini models
PROVIDER_XAI = "xai"  # xAI models (Grok)

# Default values
# These provide sensible defaults that work well for most use cases
DEFAULT_MODEL = "gpt-4o-mini"  # Fast, cost-effective model for general use
DEFAULT_TEMPERATURE = 0.7  # Balanced between creativity and consistency
DEFAULT_MAX_ITERATIONS = 3  # Usually sufficient for good improvements
DEFAULT_TIMEOUT = 300  # 5 minutes - reasonable for most operations
# DEFAULT_CRITIC removed - use CriticType.REFLEXION directly

# Limits
# These define acceptable ranges for configuration validation
MIN_ITERATIONS = 1  # At least one iteration required
MAX_ITERATIONS = 10  # Prevent runaway processing
MIN_TEMPERATURE = 0.0  # Deterministic generation
MAX_TEMPERATURE = 2.0  # Maximum creativity (may be unstable)
MIN_TIMEOUT = 30  # Minimum 30 seconds for API calls
MAX_TIMEOUT = 3600  # Maximum 1 hour to prevent hanging

# JSON response format
JSON_FORMAT = "json"

# Message roles
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Error messages
ERROR_UNKNOWN_CRITIC = "Unknown critic: '{name}'. Available: {available}"
ERROR_INVALID_ITERATIONS = (
    f"max_iterations must be between {MIN_ITERATIONS} and {MAX_ITERATIONS}"
)
ERROR_INVALID_TEMPERATURE = (
    f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
)
ERROR_NO_PROVIDER = "No provider specified and unable to determine from model name"
ERROR_TIMEOUT = "Operation timed out after {timeout} seconds"

# Success messages
MSG_CRITIQUE_COMPLETE = "Critique completed successfully"
MSG_IMPROVEMENT_COMPLETE = "Text improvement completed"

# Plugin entry points
ENTRY_POINT_CRITICS = "sifaka.critics"
ENTRY_POINT_VALIDATORS = "sifaka.validators"
ENTRY_POINT_STORAGE = "sifaka.storage"
