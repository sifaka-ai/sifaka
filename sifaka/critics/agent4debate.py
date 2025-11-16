"""Agent4Debate critic implementation for competitive text improvement.

Based on: Agent4Debate: Multiagent Competitive Debate Dynamics
Paper: https://arxiv.org/abs/2408.04472
Authors: Chen et al. (2024)

Agent4Debate uses multi-agent competitive debate to improve text through
structured argumentation. This implementation adapts the approach for
iterative text refinement through adversarial critique.

## Similarity to Original Paper:

- PRESERVED: Multi-agent roles (Analyzer, Writer, Reviewer)
- PRESERVED: Competitive debate dynamics between positions
- PRESERVED: Judge evaluation mechanism
- SIMPLIFIED: 3 agents instead of 4 (merged Searcher into Analyzer)
- ADAPTED: Text improvement focus vs. general debate
- MODIFIED: Synchronous debate rounds vs. dynamic interaction

## Implementation Strategy:

1. **Agent Roles**:
   - ANALYZER: Evaluates text strengths/weaknesses, proposes improvements
   - WRITER: Champions specific text versions with arguments
   - REVIEWER: Critiques arguments and identifies flaws
   - JUDGE: Evaluates debate and determines winning position

2. **Debate Structure**:
   - Opening: Agents present initial positions on text improvements
   - Rebuttal: Agents challenge opposing views
   - Summary: Final arguments before judgment

3. **Competitive Dynamics**:
   - Multiple text versions compete through argumentation
   - Agents must defend their positions with evidence
   - Judge selects most compelling improvement

4. **Iterative Learning**:
   - Debate history informs future rounds
   - Winning arguments shape subsequent improvements

## Why This Approach:

- **Adversarial Testing**: Improvements must withstand scrutiny
- **Multiple Perspectives**: Different agents champion different qualities
- **Structured Reasoning**: Clear argumentation for each change
- **Quality Through Competition**: Best ideas emerge from debate
- **Transparent Decision-Making**: Judge explains winning rationale

## Best Use Cases:

This critic excels when you need to carefully weigh competing improvements,
especially for high-stakes content where trade-offs between different
qualities (clarity vs. completeness, brevity vs. detail) must be evaluated.
"""

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from .core.base import BaseCritic, CriticResponse


class DebateRole(str, Enum):
    """Roles that agents play in the debate."""

    ANALYZER = "analyzer"
    WRITER = "writer"
    REVIEWER = "reviewer"
    JUDGE = "judge"


class DebatePosition(BaseModel):
    """A position taken by an agent in the debate."""

    agent_role: DebateRole = Field(..., description="The role of the agent")
    position: str = Field(..., description="The agent's position on text improvement")
    proposed_text: str = Field(..., description="The text version this agent champions")
    arguments: List[str] = Field(
        default_factory=list, description="Supporting arguments for this position"
    )
    evidence: List[str] = Field(
        default_factory=list, description="Evidence supporting the arguments"
    )


class DebateRound(BaseModel):
    """A round of debate between agents."""

    round_type: str = Field(
        ..., description="Type of round: opening, rebuttal, summary"
    )
    positions: List[DebatePosition] = Field(
        default_factory=list, description="Positions taken in this round"
    )
    key_contentions: List[str] = Field(
        default_factory=list, description="Main points of contention"
    )


class JudgeDecision(BaseModel):
    """The judge's evaluation of the debate."""

    winning_position: int = Field(..., description="Index of winning position")
    reasoning: str = Field(..., description="Explanation for the decision")
    strengths_per_position: Dict[int, List[str]] = Field(
        default_factory=dict, description="Strengths of each position"
    )
    weaknesses_per_position: Dict[int, List[str]] = Field(
        default_factory=dict, description="Weaknesses of each position"
    )
    improvement_synthesis: str = Field(
        "", description="Synthesized improvement based on debate insights"
    )


class Agent4DebateResponse(CriticResponse):
    """Extended response with debate transcript and decision."""

    # Simplified fields that are easier for LLM to populate
    opening_positions: List[str] = Field(
        default_factory=list, description="Opening positions from each agent"
    )
    key_arguments: List[str] = Field(
        default_factory=list, description="Key arguments made during debate"
    )
    winning_approach: str = Field("", description="The winning approach and why it won")
    debate_insights: str = Field("", description="Key insights from the debate process")


class Agent4DebateCritic(BaseCritic):
    """Multi-agent competitive debate critic for text improvement.

    This critic simulates a debate between agents who champion different
    text improvements. Through structured argumentation and judge evaluation,
    the best improvements emerge from competition.
    """

    @property
    def name(self) -> str:
        return "agent4debate"

    def _get_system_prompt(self) -> str:
        return """You are a text improvement critic that uses debate between different perspectives.

Your role: Simulate agents debating how to improve text, then synthesize the best approach.

Always provide your critique in the form of a debate with competing viewpoints."""

    def _get_response_type(self) -> type[BaseModel]:
        # Use base response type for now to debug
        return CriticResponse

    def get_instructions(self, text: str, result: SifakaResult) -> str:
        iteration_context = ""
        if result.generations and len(result.generations) > 1:
            iteration_context = f"""
This is iteration {len(result.generations)}. Previous debates focused on:
{self._get_previous_debate_topics(result)}
Focus on NEW aspects and trade-offs not yet debated.
"""

        return f"""Analyze this text by simulating a debate between different improvement approaches:

{text}

Create a debate where:
- One perspective argues for MINIMAL changes (preserve original style)
- Another argues for MAJOR rewrites (transform completely)
- A third seeks BALANCE (selective improvements)

For each perspective, explain:
1. What specific changes they propose (or don't propose)
2. Why their approach is best
3. What trade-offs they accept

Then declare which approach wins and why, providing specific improvement suggestions based on the winning perspective.

{iteration_context}

Your response should demonstrate genuine disagreement between the perspectives, revealing the trade-offs in different improvement strategies."""

    def _get_previous_debate_topics(self, result: SifakaResult) -> str:
        """Extract topics from previous debates to avoid repetition."""
        topics = []
        for critique in result.critiques:
            if critique.critic == self.name and "debate_summary" in critique.metadata:
                summary = critique.metadata["debate_summary"]
                # Extract key topic from summary
                if summary:
                    topics.append(summary[:100] + "...")

        return (
            "\n".join(f"- {topic}" for topic in topics[-2:])
            if topics
            else "No previous debates"
        )

    async def critique(self, text: str, result: SifakaResult) -> Any:
        """Override to ensure debate insights are captured."""
        critique_result = await super().critique(text, result)

        # Add winning approach to suggestions if not already there
        if (
            hasattr(critique_result, "metadata")
            and "winning_approach" in critique_result.metadata
        ):
            approach = critique_result.metadata["winning_approach"]
            if approach and len(critique_result.suggestions) == 0:
                critique_result.suggestions.append(f"Based on debate: {approach}")

        return critique_result
