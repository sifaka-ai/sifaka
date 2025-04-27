"""
Core Reflector class for Sifaka.
"""

from typing import List, Optional, Dict, Any, Callable, Union, ClassVar
from pydantic import BaseModel, Field
from .rules.base import Rule
from .critique.base import Critique
from .models.base import ModelProvider
from .utils.logging import get_logger

logger = get_logger(__name__)


class Reflector(BaseModel):
    """
    The main class for the Sifaka framework that orchestrates the reflection process.

    Reflector applies rules and critiques to LLM outputs to ensure they meet quality standards
    before being presented to users.

    Attributes:
        rules (List[Union[Rule, Callable]]): List of rules to apply to the LLM output
        critique_enabled (bool): Whether critique is enabled
        critique (Optional[Critique]): The critique to use if enabled
        trace (bool): Whether to trace the reflection process
        trace_data (List[Dict[str, Any]]): Trace data if tracing is enabled
    """

    rules: List[Union[Rule, Callable]] = Field(default_factory=list)
    critique_enabled: bool = False
    critique: Optional[Critique] = None
    trace: bool = False
    trace_data: List[Dict[str, Any]] = Field(default_factory=list)

    # Class variables
    _function_rule_class: ClassVar[type] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        rules: Optional[List[Union[Rule, Callable]]] = None,
        critique: Union[bool, Critique] = False,
        trace: bool = False,
        **data,
    ):
        """
        Initialize a reflector.

        Args:
            rules (Optional[List[Union[Rule, Callable]]]): List of rules to apply to the LLM output
            critique (Union[bool, Critique]): Whether to apply critique to improve the output
            trace (bool): Whether to trace the reflection process
            **data: Additional data for the reflector
        """
        # Set up initial values
        init_data = {
            "rules": rules or [],
            "critique_enabled": bool(critique),
            "critique": critique if isinstance(critique, Critique) else None,
            "trace": trace,
            "trace_data": [],
        }

        # Update with any additional data
        init_data.update(data)

        # Initialize the model
        super().__init__(**init_data)

        # Convert callable rules to Rule objects
        self._convert_callable_rules()

    def _convert_callable_rules(self) -> None:
        """
        Convert callable rules to Rule objects.
        """
        if self._function_rule_class is None:
            from .rules.base import FunctionRule

            self.__class__._function_rule_class = FunctionRule

        for i, rule in enumerate(self.rules):
            if callable(rule) and not isinstance(rule, Rule):
                self.rules[i] = self._function_rule_class(func=rule)

    def run(self, model: ModelProvider, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Run the reflection process on the given prompt using the provided model.

        Args:
            model (ModelProvider): The LLM provider to use
            prompt (str): The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the model

        Returns:
            Dict[str, Any]: The result of the reflection process, including:
                - original_output: The original output from the LLM
                - final_output: The final output after applying rules and critiques
                - rule_violations: Any rule violations that were detected
                - trace: Trace data if tracing is enabled
        """
        logger.info(f"Running reflection with {len(self.rules)} rules")

        # Get the initial output from the model
        original_output = model.generate(prompt, **kwargs)
        current_output = original_output

        if self.trace:
            self.trace_data.append({"stage": "initial_output", "output": original_output})

        # Apply rules
        rule_violations = []
        for rule in self.rules:
            result = rule.validate(current_output, prompt=prompt)
            if not result.passed:
                rule_violations.append(
                    {"rule": rule.name, "message": result.message, "metadata": result.metadata}
                )

                if self.trace:
                    self.trace_data.append(
                        {
                            "stage": f"rule_violation_{rule.name}",
                            "violation": result.message,
                            "metadata": result.metadata,
                        }
                    )

        # Apply critique if enabled and there are violations
        if self.critique_enabled and (rule_violations or isinstance(self.critique, Critique)):
            if self.critique is None:
                from .critique.prompt import PromptCritique

                self.critique = PromptCritique(model)

            improved_output = self.critique.improve(
                current_output, prompt=prompt, rule_violations=rule_violations
            )

            if self.trace:
                self.trace_data.append(
                    {"stage": "critique", "original": current_output, "improved": improved_output}
                )

            current_output = improved_output

        # Prepare the result
        result = {
            "original_output": original_output,
            "final_output": current_output,
            "rule_violations": rule_violations,
        }

        if self.trace:
            result["trace"] = self.trace_data

        return result
