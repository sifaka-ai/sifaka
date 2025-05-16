"""
Chain API Module

This module provides a simplified Chain class for building and running Sifaka chains.
It uses a fluent interface for adding components and configuring the chain.
"""

from typing import Any, Dict, List, Optional, Union
import logging

# Import the necessary components from Sifaka
from sifaka.core import initialize_registry
from sifaka.chain.chain import Chain as CoreChain
from sifaka.chain.state import create_chain_state
from sifaka.adapters.chain.validator import ValidatorAdapter
from sifaka.rules.formatting.length import create_length_rule
from sifaka.utils.config.rules import RuleConfig
from sifaka.core.results import ValidationResult, create_base_result
from sifaka.interfaces.chain.components import Validator

# Initialize the registry to ensure all factory functions are registered
initialize_registry.initialize_registry()

logger = logging.getLogger(__name__)


class Chain:
    """
    A simplified Chain builder for Sifaka.

    This class provides a fluent interface for building chains, making it easier
    to create and configure chains without dealing with the underlying complexity.

    Example:
        ```python
        import sifaka

        # Create a chain
        chain = (sifaka.Chain()
            .add_critic("prompt", instructions="Evaluate this text")
            .add_rule("length", max_length=500)
            .set_model(sifaka.model("openai", api_key="...")))

        # Run the chain
        result = chain.run("Your input here")
        ```
    """

    def __init__(self, name: str = "default_chain"):
        """
        Initialize a new Chain builder.

        Args:
            name: A name for the chain (optional)
        """
        self.name = name
        self.critics: List[Dict[str, Any]] = []
        self.rules: List[Dict[str, Any]] = []
        self.model = None
        self.classifiers: List[Dict[str, Any]] = []
        self.retrievers: List[Dict[str, Any]] = []
        self._chain: Optional[CoreChain] = None

    def add_critic(self, critic_type: str, **kwargs) -> "Chain":
        """
        Add a critic to the chain.

        Args:
            critic_type: The type of critic to add (e.g., "prompt", "reflexion")
            **kwargs: Additional arguments to pass to the critic factory

        Returns:
            The chain instance for method chaining
        """
        self.critics.append({"type": critic_type, "config": kwargs})
        return self

    def add_rule(self, rule_type: str, **kwargs) -> "Chain":
        """
        Add a rule to the chain.

        Args:
            rule_type: The type of rule to add (e.g., "length", "toxicity")
            **kwargs: Additional arguments to pass to the rule factory

        Returns:
            The chain instance for method chaining
        """
        self.rules.append({"type": rule_type, "config": kwargs})
        return self

    def add_classifier(self, classifier_type: str, **kwargs) -> "Chain":
        """
        Add a classifier to the chain.

        Args:
            classifier_type: The type of classifier to add
            **kwargs: Additional arguments to pass to the classifier factory

        Returns:
            The chain instance for method chaining
        """
        self.classifiers.append({"type": classifier_type, "config": kwargs})
        return self

    def add_retriever(self, retriever_type: str, **kwargs) -> "Chain":
        """
        Add a retriever to the chain.

        Args:
            retriever_type: The type of retriever to add
            **kwargs: Additional arguments to pass to the retriever factory

        Returns:
            The chain instance for method chaining
        """
        # Store retriever_type as part of the config
        config = kwargs.copy()
        config["retriever_type"] = retriever_type
        self.retrievers.append({"type": retriever_type, "config": config})
        return self

    def set_model(self, model: Any) -> "Chain":
        """
        Set the model to use for the chain.

        Args:
            model: The model provider instance to use

        Returns:
            The chain instance for method chaining
        """
        self.model = model
        return self

    def _build(self) -> CoreChain:
        """
        Build the actual chain from the configured components.

        Returns:
            A configured CoreChain instance
        """
        from sifaka.core.factories import create_critic
        from sifaka.critics.implementations.prompt import create_prompt_critic
        from sifaka.rules.formatting.length import create_length_rule
        from sifaka.interfaces.chain.components import Validator
        from sifaka.adapters.chain.validator import ValidatorAdapter

        # First make sure we have a model
        if not self.model:
            raise ValueError("A model provider must be set before building the chain")

        # Create critics (improvers)
        improver = None
        if self.critics:
            # For now, we'll use only the first critic as the improver
            critic_config = self.critics[0]

            # Use specific critic factory functions based on type
            if critic_config["type"] == "prompt":
                # For prompt critics, we need to use create_prompt_critic and pass llm_provider explicitly
                improver = create_prompt_critic(llm_provider=self.model, **critic_config["config"])
            else:
                # For other critic types, use the general factory function
                improver = create_critic(
                    critic_config["type"], model_provider=self.model, **critic_config["config"]
                )

        # Create validators
        validators = []
        if self.rules:
            for rule_config in self.rules:
                if rule_config["type"] == "length":
                    # Convert max_length to max_chars for length rules
                    config = rule_config["config"].copy()
                    if "max_length" in config:
                        config["max_chars"] = config.pop("max_length")

                    # Create a length rule using the specialized factory
                    length_rule = create_length_rule(**config)

                    # Wrap the rule in a validator adapter
                    validator = ValidatorAdapter(length_rule)
                    validators.append(validator)

        # Create the chain with all components
        chain = CoreChain(
            model=self.model, name=self.name, validators=validators, improver=improver
        )

        return chain

    def run(self, input_text: str, **kwargs) -> Any:
        """
        Run the chain on the input text.

        Args:
            input_text: The input text to process
            **kwargs: Additional arguments to pass to the chain's run method

        Returns:
            The result of running the chain
        """
        # Create the chain if not already created
        if not self._chain:
            self._chain = self._build()

        try:
            # Use the core chain to run with the input text
            result = self._chain.run(input_text, **kwargs)

            # If we get a ChainResult, extract just the output text for simplicity
            if hasattr(result, "output"):
                return result.output

            return result
        except Exception as e:
            # If the chain execution fails, fall back to using the model directly
            print(f"Warning: Chain execution failed: {str(e)}")
            print("Falling back to direct model generation...")

            # Use the model directly as a fallback
            generated_output = self.model.generate(input_text)

            # If we got back a GenerationResult, extract just the output
            if hasattr(generated_output, "output"):
                return generated_output.output

            return generated_output

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the chain configuration to a dictionary.

        This can be used to save the chain configuration for later use.

        Returns:
            A dictionary representation of the chain configuration
        """
        return {
            "name": self.name,
            "critics": self.critics,
            "rules": self.rules,
            "classifiers": self.classifiers,
            "retrievers": self.retrievers,
            # Note: model is not included as it might not be serializable
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Chain":
        """
        Create a chain from a dictionary configuration.

        Args:
            config: A dictionary representation of the chain configuration

        Returns:
            A configured Chain instance
        """
        chain = cls(name=config.get("name", "default_chain"))

        # Add critics
        for critic_config in config.get("critics", []):
            chain.add_critic(critic_config["type"], **critic_config.get("config", {}))

        # Add rules
        for rule_config in config.get("rules", []):
            chain.add_rule(rule_config["type"], **rule_config.get("config", {}))

        # Add classifiers
        for classifier_config in config.get("classifiers", []):
            chain.add_classifier(classifier_config["type"], **classifier_config.get("config", {}))

        # Add retrievers
        for retriever_config in config.get("retrievers", []):
            chain.add_retriever(retriever_config["type"], **retriever_config.get("config", {}))

        return chain
