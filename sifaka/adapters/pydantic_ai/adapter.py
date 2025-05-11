"""
PydanticAI Adapter

Core adapter implementation for PydanticAI integration with Sifaka.

## Overview
This module provides the core adapter class that bridges between PydanticAI agents
and Sifaka's validation and refinement capabilities. It enables PydanticAI agents
to benefit from Sifaka's rule-based validation and critic-based refinement to
improve the semantic quality of outputs beyond just structural validation.

## Components
1. **SifakaPydanticConfig**: Configuration for the adapter
2. **SifakaPydanticAdapter**: Main adapter class for PydanticAI integration

## Usage Examples
```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from sifaka.adapters.pydantic_ai import SifakaPydanticAdapter
from sifaka.rules.formatting.length import create_length_rule

# Define a Pydantic model
class Response(BaseModel):
    content: str

# Create rules and adapter
rules = [create_length_rule(min_chars=10, max_chars=100)]
adapter = SifakaPydanticAdapter(
    rules=rules,
    output_model=Response,
    max_refine=2
)

# Use as a PydanticAI output validator
@agent.output_validator
def validate_with_sifaka(ctx: RunContext, output: Response) -> Response:
    return adapter(ctx, output)
```

## Error Handling
- ImportError: Raised when PydanticAI is not installed
- ValueError: Raised when output model is invalid
- ModelRetry: Raised when validation fails and refinement is needed
- AdapterError: Raised for adapter-specific errors

## State Management
The module uses a standardized state management approach:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state object
- Clear separation of configuration and state
- Execution tracking for monitoring and debugging

## Configuration
- max_refine: Maximum number of refinement attempts
- prioritize_by_cost: Whether to prioritize rules by cost
- serialize_method: Method to use for serializing Pydantic models
- deserialize_method: Method to use for deserializing Pydantic models
"""

import time
import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

# Import PydanticAI types
try:
    from pydantic_ai import ModelRetry, RunContext
except ImportError:
    raise ImportError(
        "PydanticAI is not installed. Please install it with 'pip install pydantic-ai'"
    )

# Import Sifaka components
from sifaka.critics.base import BaseCritic
from sifaka.rules.base import Rule, RuleResult, ConfigurationError, ValidationError
from sifaka.validation.models import ValidationResult
from sifaka.validation.validator import Validator, ValidatorConfig
from sifaka.adapters.base import BaseAdapter, AdapterError
from sifaka.utils.state import StateManager, create_adapter_state
from sifaka.utils.errors import handle_error
from sifaka.utils.logging import get_logger

# Type variables
T = TypeVar("T", bound=BaseModel)

# Set up logging
logger = get_logger(__name__)


class SifakaPydanticConfig(BaseModel):
    """
    Configuration for the SifakaPydanticAdapter.

    ## Overview
    This class provides configuration options for the PydanticAI adapter,
    controlling aspects like refinement attempts and serialization methods.

    ## Architecture
    The configuration follows a standard pattern with:
    1. Refinement settings
    2. Rule prioritization
    3. Serialization methods

    ## Error Handling
    - ValueError: Raised when max_refine is invalid
    - TypeError: Raised when serialization methods are invalid

    ## Examples
    ```python
    from sifaka.adapters.pydantic_ai import SifakaPydanticConfig

    # Create a configuration
    config = SifakaPydanticConfig(
        max_refine=3,
        prioritize_by_cost=True,
        serialize_method="model_dump",
        deserialize_method="model_validate"
    )
    ```

    Attributes:
        max_refine (int): Maximum number of refinement attempts
        prioritize_by_cost (bool): Whether to prioritize rules by cost
        serialize_method (str): Method to use for serializing Pydantic models
        deserialize_method (str): Method to use for deserializing Pydantic models
    """

    max_refine: int = Field(default=2, description="Maximum number of refinement attempts")
    prioritize_by_cost: bool = Field(
        default=False, description="Whether to prioritize rules by cost"
    )
    serialize_method: str = Field(
        default="model_dump", description="Method to use for serializing Pydantic models"
    )
    deserialize_method: str = Field(
        default="model_validate", description="Method to use for deserializing Pydantic models"
    )

    model_config = ConfigDict(
        title="PydanticAI Adapter Configuration",
        description="Configuration for the SifakaPydanticAdapter",
    )


class SifakaPydanticAdapter(BaseModel):
    """
    Adapter for integrating Sifaka's validation and refinement with PydanticAI agents.

    ## Overview
    This adapter bridges between PydanticAI's output validation system and Sifaka's
    rule-based validation and critic-based refinement capabilities. It enables
    PydanticAI agents to benefit from Sifaka's semantic validation beyond just
    structural validation.

    ## Architecture
    The adapter follows a standard pattern:
    1. Initialization with rules and configuration
    2. Validation against Sifaka rules
    3. Refinement using critic if needed
    4. Result conversion to Pydantic model

    ## Lifecycle
    1. Initialization: Set up with rules, critic, and configuration
    2. Validation: Validate PydanticAI output against Sifaka rules
    3. Refinement: If validation fails, use critic to refine output
    4. Result: Return validated or refined output

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state object
    - Clear separation of configuration and state
    - State components:
      - rules: List of Sifaka rules to validate against
      - critic: Optional Sifaka critic for refinement
      - output_model: The Pydantic model type for the output
      - validator: The validator instance
      - execution_count: Number of validation executions
      - last_execution_time: Timestamp of last execution
      - avg_execution_time: Average execution time
      - error_count: Number of validation errors
      - cache: Temporary data storage

    ## Error Handling
    - ImportError: Raised when PydanticAI is not installed
    - ValueError: Raised when output model is invalid
    - ModelRetry: Raised when validation fails and refinement is needed
    - AdapterError: Raised for adapter-specific errors

    ## Examples
    ```python
    from pydantic import BaseModel
    from pydantic_ai import Agent, RunContext
    from sifaka.adapters.pydantic_ai import SifakaPydanticAdapter
    from sifaka.rules.formatting.length import create_length_rule

    # Define a Pydantic model
    class Response(BaseModel):
        content: str

    # Create rules and adapter
    rules = [create_length_rule(min_chars=10, max_chars=100)]
    adapter = SifakaPydanticAdapter(
        rules=rules,
        output_model=Response,
        max_refine=2
    )

    # Use as a PydanticAI output validator
    @agent.output_validator
    def validate_with_sifaka(ctx: RunContext, output: Response) -> Response:
        return adapter(ctx, output)
    ```

    Attributes:
        rules (List[Rule]): List of Sifaka rules to validate against
        critic (Optional[BaseCritic]): Optional Sifaka critic for refinement
        output_model (Type[BaseModel]): The Pydantic model type for the output
        config (SifakaPydanticConfig): Configuration for the adapter
    """

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using standardized state manager
    _state_manager = PrivateAttr(default_factory=create_adapter_state)

    # Required fields
    rules: List[Rule]
    output_model: Type[BaseModel]

    # Optional fields
    critic: Optional[BaseCritic] = None
    config: SifakaPydanticConfig = Field(default_factory=SifakaPydanticConfig)

    def __init__(
        self,
        rules: List[Rule],
        output_model: Type[BaseModel],
        critic: Optional[BaseCritic] = None,
        config: Optional[SifakaPydanticConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize the adapter.

        Args:
            rules: List of Sifaka rules to validate against
            output_model: The Pydantic model type for the output
            critic: Optional Sifaka critic for refinement
            config: Configuration for the adapter
            **kwargs: Additional keyword arguments

        Raises:
            ConfigurationError: If configuration is invalid
            AdapterError: If initialization fails
        """
        try:
            # Initialize Pydantic model
            super().__init__(
                rules=rules,
                output_model=output_model,
                critic=critic,
                config=config or SifakaPydanticConfig(),
                **kwargs,
            )

            # Initialize state
            state = self._state_manager.get_state()
            state.adaptee = None  # This adapter doesn't have a single adaptee
            state.initialized = True
            state.execution_count = 0
            state.error_count = 0
            state.last_execution_time = None
            state.avg_execution_time = 0
            state.cache = {}
            state.config_cache = {"rules": rules, "output_model": output_model, "critic": critic}

            # Create validator config
            validator_config = ValidatorConfig(
                prioritize_by_cost=self.config.prioritize_by_cost,
                fail_fast=False,  # Don't fail fast for PydanticAI integration
            )

            # Create validator with config
            state.validator = Validator(rules=self.rules, config=validator_config)

            # Set metadata
            self._state_manager.set_metadata("component_type", "adapter")
            self._state_manager.set_metadata("adapter_type", "pydantic_ai")
            self._state_manager.set_metadata("creation_time", time.time())
            self._state_manager.set_metadata("rule_count", len(self.rules))
            if self.critic:
                self._state_manager.set_metadata("critic_type", self.critic.__class__.__name__)

            logger.debug(f"Initialized SifakaPydanticAdapter with {len(self.rules)} rules")
        except Exception as e:
            error_info = handle_error(e, "SifakaPydanticAdapter:init")
            raise AdapterError(
                f"Failed to initialize SifakaPydanticAdapter: {str(e)}", metadata=error_info
            ) from e

    def warm_up(self) -> None:
        """
        Initialize the adapter if needed.

        This method ensures the adapter is properly initialized before use.
        It's safe to call multiple times.

        Raises:
            AdapterError: If initialization fails
        """
        try:
            # Check if already initialized
            if self._state_manager.get_state().initialized:
                return

            # Initialize state
            state = self._state_manager.get_state()
            state.initialized = True

            logger.debug(f"Adapter {self.__class__.__name__} initialized")
        except Exception as e:
            error_info = handle_error(e, f"Adapter:{self.__class__.__name__}")
            raise AdapterError(
                f"Failed to initialize adapter: {str(e)}", metadata=error_info
            ) from e

    def __call__(self, ctx: RunContext, output: T) -> T:
        """
        Validate and potentially refine the PydanticAI output.

        ## Overview
        This method is called by PydanticAI's output validation system. It validates
        the output against Sifaka rules and, if validation fails, uses the critic
        to refine the output.

        Args:
            ctx (RunContext): The PydanticAI run context
            output (T): The PydanticAI output to validate

        Returns:
            T: The validated or refined output

        Raises:
            ModelRetry: If validation fails and refinement is needed
            ValueError: If output model is invalid
            AdapterError: If adapter-specific error occurs
        """
        # Ensure initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Track execution
        state.execution_count += 1
        start_time = time.time()

        try:
            # Check if ctx has a state attribute with retries
            retries = 0
            if hasattr(ctx, "state") and hasattr(ctx.state, "retries"):
                retries = ctx.state.retries

            logger.info(f"PydanticAI adapter processing output (attempt {retries + 1})")

            # Convert Pydantic model to dict for validation using Pydantic 2 method
            serialize_method = getattr(output, self.config.serialize_method, None)
            if serialize_method is None:
                raise ValueError(
                    f"Cannot serialize {type(output).__name__}. The model must be a Pydantic v2 model with a {self.config.serialize_method} method."
                )

            # Call the serialization method to get the data
            output_data = serialize_method()
            logger.debug(f"Serialized output: {output_data}")
            issues = []

            # Check cache if enabled
            cache_key = self._get_cache_key(output_data)
            if cache_key and cache_key in state.cache:
                cached_result = state.cache[cache_key]
                logger.debug(f"Cache hit for output validation")

                # If cached result passed, return the original output
                if cached_result.get("passed", False):
                    logger.info("✅ Validation passed (cached) - returning original output")
                    return output

                # If cached result failed, use cached issues
                issues = cached_result.get("issues", [])
                logger.warning(f"❌ Validation failed (cached) with {len(issues)} issues")
            else:
                # Convert output_data to string for validation if needed
                # Most Sifaka rules expect string input
                if isinstance(output_data, dict):
                    # Try to convert dict to string for validation
                    try:
                        output_str = json.dumps(output_data)
                        logger.debug("Converted output to JSON string for validation")
                    except Exception as e:
                        output_str = str(output_data)
                        logger.debug(f"JSON conversion failed, using str(): {e}")
                else:
                    output_str = str(output_data)
                    logger.debug("Using string representation of output for validation")

                # Validate against Sifaka rules
                logger.info(f"Validating output against {len(self.rules)} Sifaka rules")
                validation_result = state.validator.validate(output_str)

                # If validation passes, return the original output
                if validation_result.all_passed:
                    # Cache result if enabled
                    if cache_key:
                        state.cache[cache_key] = {
                            "passed": True,
                            "issues": [],
                        }
                    logger.info("✅ Validation passed - returning original output")
                    return output

                # Get error messages for failed rules
                error_messages = state.validator.get_error_messages(validation_result)
                issues.extend(error_messages)

                # Cache result if enabled
                if cache_key:
                    state.cache[cache_key] = {
                        "passed": False,
                        "issues": issues,
                    }

                logger.warning(f"❌ Validation failed with {len(issues)} issues:")
                for i, issue in enumerate(issues):
                    logger.warning(f"  Issue {i+1}: {issue}")

            # If we have a critic and haven't exceeded max refinement attempts, retry
            # Check if ctx has a state attribute with retries
            retries = 0
            if hasattr(ctx, "state") and hasattr(ctx.state, "retries"):
                retries = ctx.state.retries

            if retries < self.config.max_refine:
                # Format issues for the model to understand
                formatted_issues = "\n".join([f"- {issue}" for issue in issues])
                error_message = f"Validation failed:\n{formatted_issues}\nPlease fix these issues and try again."

                logger.info(
                    f"Requesting refinement (attempt {retries + 1}/{self.config.max_refine})"
                )

                # If we have a critic, log that information
                if self.critic:
                    critic_name = getattr(self.critic, "name", type(self.critic).__name__)
                    logger.info(f"Using critic: {critic_name} for additional guidance")

                # Raise ModelRetry to trigger a retry with the error message
                raise ModelRetry(error_message)

            # If we've exceeded max refinement attempts, return the original output
            logger.warning(
                f"⚠️ Max refinement attempts ({self.config.max_refine}) reached - returning output as-is"
            )
            return output
        except Exception as e:
            # Track error
            state.error_count += 1

            # Don't wrap ModelRetry exceptions
            if isinstance(e, ModelRetry):
                raise

            # Handle different error types
            if isinstance(e, ValueError):
                raise
            elif isinstance(e, AdapterError):
                raise
            else:
                error_info = handle_error(e, "SifakaPydanticAdapter:validate")
                raise AdapterError(f"Validation failed: {str(e)}", metadata=error_info) from e
        finally:
            # Update execution stats
            execution_time = time.time() - start_time
            state.last_execution_time = execution_time

            # Update average execution time
            if state.execution_count > 1:
                state.avg_execution_time = (
                    state.avg_execution_time * (state.execution_count - 1) + execution_time
                ) / state.execution_count
            else:
                state.avg_execution_time = execution_time

    def _get_cache_key(self, output_data: Any) -> Optional[str]:
        """
        Generate a cache key for the output data.

        Args:
            output_data: The output data to generate a cache key for

        Returns:
            Optional[str]: Cache key or None if caching is disabled
        """
        try:
            # Convert output_data to a hashable string
            if isinstance(output_data, dict):
                return json.dumps(output_data, sort_keys=True)
            return str(output_data)
        except Exception:
            # If we can't generate a cache key, disable caching
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about adapter usage.

        Returns:
            Dict[str, Any]: Dictionary with usage statistics
        """
        state = self._state_manager.get_state()
        return {
            "execution_count": state.execution_count,
            "error_count": state.error_count,
            "avg_execution_time": state.avg_execution_time,
            "last_execution_time": state.last_execution_time,
            "cache_size": len(state.cache),
            "rule_count": len(self.rules),
            "has_critic": self.critic is not None,
            "max_refine": self.config.max_refine,
        }
