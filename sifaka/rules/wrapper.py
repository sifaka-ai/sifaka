"""Wrapper functions for integrating rules with LangGraph."""

from typing import List, Dict, Union, Any
from langchain.schema import BaseMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from sifaka.rules.base import Rule, RuleResult


def wrap_rule(rule: Rule):
    """Wrap a rule to handle both string and message inputs.

    Args:
        rule: The rule to wrap.

    Returns:
        A function that takes either a string or message and returns a RuleResult.
    """

    def wrapped(input_text: Union[str, BaseMessage], **kwargs) -> RuleResult:
        if isinstance(input_text, BaseMessage):
            text = input_text.content
        else:
            text = input_text
        return rule.validate(text, **kwargs)

    return wrapped


def wrap_graph(rules: List[Rule]) -> StateGraph:
    """Create a LangGraph from a list of rules.

    Args:
        rules: List of rules to include in the graph.

    Returns:
        A LangGraph that can process inputs through all rules.
    """

    # Define the state type
    class GraphState(BaseModel):
        input: Union[str, BaseMessage]
        results: Dict[str, RuleResult] = Field(default_factory=dict)

    workflow = StateGraph(state_schema=GraphState)

    # Create nodes for each rule
    for rule in rules:

        def create_node_fn(r: Rule):
            def node_fn(state: GraphState) -> GraphState:
                result = wrap_rule(r)(state.input)
                state.results[r.name] = result
                return state

            return node_fn

        workflow.add_node(rule.name, create_node_fn(rule))

    # Connect rules sequentially
    for i in range(len(rules) - 1):
        workflow.add_edge(rules[i].name, rules[i + 1].name)

    # Configure the graph flow
    workflow.set_entry_point(rules[0].name)
    workflow.set_finish_point(rules[-1].name)

    return workflow
