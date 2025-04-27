from langgraph.graph import Graph
from langgraph.prebuilt import ToolNode
from anthropic import Anthropic
from pydantic import BaseModel, Field
from typing import Literal

from sifaka.integrations.langgraph import wrap_graph
from sifaka.rules.base import Rule, RuleResult
from sifaka.critique.base import Critique


class LengthRule(Rule):
    """Rule that checks if the output is too short."""

    def __init__(self):
        super().__init__(
            name="length_rule", description="Checks if the output is at least 50 characters long"
        )

    def validate(self, output: str) -> RuleResult:
        if len(output) < 50:
            return RuleResult(
                passed=False, message="Output is too short", metadata={"length": len(output)}
            )
        return RuleResult(passed=True)


class ProhibitedContentRule(Rule):
    """Rule that checks if the output contains prohibited terms."""

    def __init__(self, prohibited_terms: list[str]):
        super().__init__(
            name="prohibited_content_rule",
            description="Checks if the output contains any prohibited terms",
        )
        self.prohibited_terms = prohibited_terms

    def validate(self, output: str) -> RuleResult:
        found_terms = [term for term in self.prohibited_terms if term.lower() in output.lower()]
        if found_terms:
            return RuleResult(
                passed=False,
                message=f"Output contains prohibited terms: {', '.join(found_terms)}",
                metadata={"found_terms": found_terms},
            )
        return RuleResult(passed=True)


class SentimentRule(Rule):
    """Rule that checks if the output maintains a neutral or positive sentiment."""

    def __init__(self, min_sentiment: float = -0.5):
        super().__init__(
            name="sentiment_rule",
            description="Checks if the output maintains a neutral or positive sentiment",
        )
        self.min_sentiment = min_sentiment

    def validate(self, output: str) -> RuleResult:
        # This is a placeholder - in a real implementation, you'd use a sentiment analysis library
        sentiment_score = 0.0  # Placeholder value
        if sentiment_score < self.min_sentiment:
            return RuleResult(
                passed=False,
                message=f"Output sentiment score ({sentiment_score}) is below minimum threshold ({self.min_sentiment})",
                metadata={"sentiment_score": sentiment_score},
            )
        return RuleResult(passed=True)


class ToxicityRule(Rule):
    """Rule that checks if the output contains toxic content."""

    def __init__(self, max_toxicity: float = 0.5):
        super().__init__(
            name="toxicity_rule", description="Checks if the output contains toxic content"
        )
        self.max_toxicity = max_toxicity

    def validate(self, output: str) -> RuleResult:
        # This is a placeholder - in a real implementation, you'd use a toxicity detection library
        toxicity_score = 0.0  # Placeholder value
        if toxicity_score > self.max_toxicity:
            return RuleResult(
                passed=False,
                message=f"Output toxicity score ({toxicity_score}) exceeds maximum threshold ({self.max_toxicity})",
                metadata={"toxicity_score": toxicity_score},
            )
        return RuleResult(passed=True)


class FormatRule(Rule):
    """Rule that checks if the output follows specific formatting requirements."""

    def __init__(self, required_format: Literal["markdown", "plain_text", "json"] = "plain_text"):
        super().__init__(
            name="format_rule",
            description="Checks if the output follows specific formatting requirements",
        )
        self.required_format = required_format

    def validate(self, output: str) -> RuleResult:
        if self.required_format == "markdown":
            # Check for basic markdown syntax
            if not any(char in output for char in ["#", "*", "_", "`"]):
                return RuleResult(
                    passed=False,
                    message="Output does not contain any markdown formatting",
                    metadata={"format": self.required_format},
                )
        elif self.required_format == "json":
            # Check for valid JSON
            try:
                import json

                json.loads(output)
            except json.JSONDecodeError:
                return RuleResult(
                    passed=False,
                    message="Output is not valid JSON",
                    metadata={"format": self.required_format},
                )
        return RuleResult(passed=True)


class ClaudeCritique(Critique):
    """Critique that uses Claude to improve outputs."""

    client: Anthropic = Field(default_factory=Anthropic)

    def __init__(self):
        super().__init__(name="claude_critique", description="Uses Claude to improve outputs")

    def critique(self, prompt: str) -> dict:
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"output": response.content[0].text}


def main():
    # Create a simple graph that returns a short message
    graph = Graph()

    # Create a node that generates a short message
    def generate_message(inputs):
        return {"output": "Hello!"}

    # Add the node to the graph
    graph.add_node("generate", generate_message)
    graph.set_entry_point("generate")

    # Create rules and critique
    rules = [
        LengthRule(),
        ProhibitedContentRule(["badword1", "badword2"]),
        SentimentRule(min_sentiment=-0.5),
        ToxicityRule(max_toxicity=0.5),
        FormatRule(required_format="plain_text"),
    ]
    critic = ClaudeCritique()

    # Wrap the graph with Sifaka's features
    sifaka_graph = wrap_graph(graph=graph, rules=rules, critique=True, critic=critic)

    # Run the graph
    print("Running graph...")
    output = sifaka_graph({"input": "Hello"})
    print("\nFinal output:", output)


if __name__ == "__main__":
    main()
