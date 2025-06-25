"""Constants for Sifaka."""

# Model providers
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"
PROVIDER_XAI = "xai"

# Default values
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_TIMEOUT = 300
DEFAULT_CRITIC = "reflexion"

# Limits
MIN_ITERATIONS = 1
MAX_ITERATIONS = 10
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TIMEOUT = 30
MAX_TIMEOUT = 3600

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
