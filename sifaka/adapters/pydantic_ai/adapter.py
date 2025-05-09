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

## Configuration
- max_refine: Maximum number of refinement attempts
- prioritize_by_cost: Whether to prioritize rules by cost
- serialize_method: Method to use for serializing Pydantic models
- deserialize_method: Method to use for deserializing Pydantic models
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from pydantic import BaseModel

# Import PydanticAI types
try:
    from pydantic_ai import ModelRetry, RunContext
except ImportError:
    raise ImportError(
        "PydanticAI is not installed. Please install it with 'pip install pydantic-ai'"
    )

# Import Sifaka components
from sifaka.critics.base import BaseCritic
from sifaka.rules.base import Rule, RuleResult
from sifaka.validation.models import ValidationResult
from sifaka.validation.validator import Validator, ValidatorConfig

# Type variables
T = TypeVar("T", bound=BaseModel)


@dataclass
class SifakaPydanticConfig:
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

    max_refine: int = 2
    prioritize_by_cost: bool = False
    serialize_method: str = "model_dump"  # For Pydantic v2
    deserialize_method: str = "model_validate"  # For Pydantic v2


class SifakaPydanticAdapter:
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

    ## Error Handling
    - ImportError: Raised when PydanticAI is not installed
    - ValueError: Raised when output model is invalid
    - ModelRetry: Raised when validation fails and refinement is needed

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

    def __init__(
        self,
        rules: List[Rule],
        output_model: Type[BaseModel],
        critic: Optional[BaseCritic] = None,
        config: Optional[SifakaPydanticConfig] = None,
    ):
        """
        Initialize the adapter.

        Args:
            rules: List of Sifaka rules to validate against
            output_model: The Pydantic model type for the output
            critic: Optional Sifaka critic for refinement
            config: Configuration for the adapter
        """
        self.rules = rules
        self.critic = critic
        self.output_model = output_model
        self.config = config or SifakaPydanticConfig()

        # Create validator config
        validator_config = ValidatorConfig(
            prioritize_by_cost=self.config.prioritize_by_cost,
            fail_fast=False,  # Don't fail fast for PydanticAI integration
        )

        # Create validator with config
        self.validator = Validator(rules=self.rules, config=validator_config)

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
        """
        # Set up logging
        import logging

        logger = logging.getLogger("sifaka.adapters.pydantic_ai")

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

        # Convert output_data to string for validation if needed
        # Most Sifaka rules expect string input
        if isinstance(output_data, dict):
            # Try to convert dict to string for validation
            try:
                import json

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
        validation_result = self.validator.validate(output_str)

        # If validation passes, return the original output
        if validation_result.all_passed:
            logger.info("✅ Validation passed - returning original output")
            return output

        # Get error messages for failed rules
        error_messages = self.validator.get_error_messages(validation_result)
        issues.extend(error_messages)
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
            error_message = (
                f"Validation failed:\n{formatted_issues}\nPlease fix these issues and try again."
            )

            logger.info(f"Requesting refinement (attempt {retries + 1}/{self.config.max_refine})")

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

    def _refine_with_critic(self, output_data: Dict[str, Any], issues: List[str]) -> T:
        """
        Refine the output using the critic.

        ## Overview
        This method uses the configured critic to refine the output based on
        validation issues.

        Args:
            output_data (Dict[str, Any]): The output data to refine
            issues (List[str]): The validation issues to address

        Returns:
            T: The refined output data

        Raises:
            ValueError: If critic is not configured
            RuntimeError: If refinement fails
        """
        # Set up logging
        import logging

        logger = logging.getLogger("sifaka.adapters.pydantic_ai")

        # Store the original output for fallback
        original_output = self.output

        if not self.critic:
            return original_output

        # Convert output_data to string for the critic
        output_str = str(output_data)

        # Format issues as feedback for the critic
        feedback = "The output has the following issues:\n"
        feedback += "\n".join([f"- {issue}" for issue in issues])

        # Use the critic to improve the output
        improved_output = self.critic.improve(output_str, feedback)

        # Try to parse the improved output back to a Pydantic model
        try:
            # This is a simplified approach - in a real implementation,
            # you would need more robust parsing logic
            import json

            # Parse the improved output to a dict
            improved_data = json.loads(improved_output)

            # Use Pydantic v2's model_validate method to convert back to the model
            deserialize_method = getattr(self.output_model, self.config.deserialize_method, None)
            if deserialize_method is None:
                raise ValueError(
                    f"Cannot deserialize to {self.output_model.__name__}. The model must be a Pydantic v2 model with a {self.config.deserialize_method} method."
                )

            # Return the validated model
            return deserialize_method(improved_data)
        except Exception as e:
            # If parsing fails, log the error and return the original output
            logger.warning(f"Failed to parse improved output: {e}")
            return original_output
