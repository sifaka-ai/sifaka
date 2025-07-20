"""Self-RAG (Retrieval-Augmented Generation) critic implementation.

Based on: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
Paper: https://arxiv.org/abs/2310.11511
Authors: Asai et al. (2023)

Self-RAG enhances text generation through selective retrieval and self-reflection,
using reflection tokens to evaluate content quality and determine when retrieval
would improve generation.

## Similarity to Original Paper:

- PRESERVED: Core reflection tokens (ISREL, ISSUP, ISUSE) for content evaluation
- PRESERVED: Adaptive retrieval decisions based on content needs
- PRESERVED: Self-critique mechanism for quality assessment
- SIMPLIFIED: Single-pass critique vs. iterative retrieval-generation cycles
- ENHANCED: Integration with external tools for fact-checking capabilities

## Implementation Strategy:

1. **Reflection Token Framework**: Uses three key tokens from the paper:
   - **ISREL (Relevance)**: Evaluates if content is relevant to its purpose
   - **ISSUP (Support)**: Checks if content is supported by evidence
   - **ISUSE (Usefulness)**: Assesses if content is valuable for readers

2. **Adaptive Retrieval**: Identifies retrieval opportunities where external
   information would enhance content quality, similar to paper's on-demand retrieval

3. **Tool Integration**: Optional fact-checking through search tools when
   factual claims need verification (extends paper's retrieval concept)

4. **Self-Reflection Loop**: Re-evaluates content after retrieval to update
   reflection tokens based on new evidence

## Why This Approach:

- **Efficiency**: Avoids unnecessary retrievals when content is already strong
- **Accuracy**: Fact-checks claims only when needed, reducing false positives
- **Adaptability**: Adjusts critique based on content type and quality needs
- **Transparency**: Reflection tokens provide clear quality signals
- **Practicality**: Integrates with existing tool ecosystem for retrieval

## Key Differences from Paper:

The original Self-RAG trains a model to generate reflection tokens during text
generation. Our implementation:
- Uses a critic model to evaluate existing text post-generation
- Applies reflection tokens as evaluation criteria rather than generation signals
- Focuses on identifying improvement opportunities rather than guiding generation
- Leverages external tools for retrieval rather than a trained retriever

This adaptation maintains the core insight of selective, self-reflective
retrieval while fitting into a critique-refinement workflow."""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.models import SifakaResult, ToolUsage
from ..tools import ToolRegistry
from .core.base import BaseCritic, CritiqueResult


class SelfRAGResponse(BaseModel):
    """Structured response from Self-RAG critic with reflection tokens."""

    # Reflection tokens
    isrel: str = Field(description="ISREL: Is content relevant? YES/NO")
    issup: str = Field(description="ISSUP: Is content supported? YES/PARTIAL/NO")
    isuse: str = Field(description="ISUSE: Is content useful? YES/PARTIAL/NO")

    # Analysis
    overall_assessment: str = Field(
        description="Overall quality assessment based on reflection tokens"
    )
    specific_issues: List[str] = Field(
        default_factory=list, description="Specific issues found in the text"
    )
    specific_corrections: List[str] = Field(
        default_factory=list,
        description="Specific factual corrections based on search results",
    )
    factual_claims: List[str] = Field(
        default_factory=list, description="Factual claims that need verification"
    )
    retrieval_opportunities: List[str] = Field(
        default_factory=list,
        description="Where retrieval would improve the content",
    )

    # Improvements
    improvement_suggestions: List[str] = Field(
        default_factory=list, description="Specific suggestions for improvement"
    )
    needs_improvement: bool = Field(
        default=True, description="Whether the text needs improvement"
    )
    confidence_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in the assessment"
    )


class SelfRAGCritic(BaseCritic):
    """Self-RAG critic that uses reflection tokens and optional retrieval.

    This critic implements the Self-RAG framework which:
    1. Evaluates content using reflection tokens (ISREL, ISSUP, ISUSE)
    2. Identifies retrieval opportunities where facts need verification
    3. Optionally calls search tools to verify factual claims
    4. Provides specific feedback based on both analysis and retrieval

    The critic adapts its behavior based on:
    - Content type (academic, factual, opinion, etc.)
    - Presence of factual claims
    - Availability of search tools
    - Previous critique history

    Example:
        >>> critic = SelfRAGCritic(model="gpt-4")
        >>> result = await critic.critique(
        ...     text="The Eiffel Tower is 500 meters tall.",
        ...     result=sifaka_result
        ... )
        >>> print(result.feedback)
        "ISREL: YES, ISSUP: NO, ISUSE: PARTIAL
         The height claim needs verification - actual height is 330 meters."
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Any] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        enable_tools: Optional[bool] = None,
    ):
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key, enable_tools)

    @property
    def name(self) -> str:
        return "self_rag"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom SelfRAGResponse for structured output."""
        return SelfRAGResponse

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Perform Self-RAG critique with optional tool usage."""
        # Get standard critique without tools first
        messages = await self._create_messages(text, result)
        base_critique = await self._perform_critique(messages)

        # Check if tools are enabled
        tools_enabled = self._are_tools_enabled()
        if not tools_enabled:
            return base_critique

        # Extract factual claims from the base critique
        if hasattr(base_critique, "metadata") and base_critique.metadata:
            factual_claims = base_critique.metadata.get("factual_claims", [])
        else:
            factual_claims = []

        if not factual_claims:
            return base_critique

        # Let the LLM decide which tool to use for verification
        tool_decision = await self._decide_tool_usage(text, factual_claims)
        if not tool_decision or not tool_decision.get("use_tools"):
            return base_critique

        # Use the selected tool
        tool_name = tool_decision.get("tool_name", "web_search")
        tool_class = ToolRegistry.get(tool_name)
        if not tool_class:
            return base_critique

        # Perform searches with the selected tool
        tool_instance = tool_class()
        tool_results = []

        for claim in factual_claims[:3]:  # Limit to 3 claims
            search_query = tool_decision.get("queries", {}).get(claim, claim)
            result_data = await self._search_with_tool(
                tool_instance, claim, search_query
            )
            if result_data:
                tool_results.append(result_data)

        # Re-critique with tool results
        if tool_results:
            messages_with_tools = await self._create_messages_with_tools(
                text, result, tool_results
            )
            final_critique = await self._perform_critique(messages_with_tools)

            # Add tool usage to the critique
            for tool_result in tool_results:
                if "tool_usage" in tool_result:
                    final_critique.tools_used.append(tool_result["tool_usage"])

            return final_critique

        return base_critique

    def _are_tools_enabled(self) -> bool:
        """Check if tools are enabled for this critic."""
        if self.config and hasattr(self.config.critic, "critic_settings"):
            return bool(
                self.config.critic.critic_settings.get(self.name, {}).get(
                    "enable_tools", False
                )
            )
        elif self.config and hasattr(self.config.critic, "enable_tools"):
            return bool(self.config.critic.enable_tools)
        return False

    async def _decide_tool_usage(
        self, text: str, factual_claims: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Let the LLM decide which tool to use and how."""
        if not factual_claims:
            return None

        # Get available tools
        available_tools = ToolRegistry.list_available()
        if not available_tools:
            return None

        # Create tool descriptions
        tool_descriptions = []
        for tool_name in available_tools:
            tool_class = ToolRegistry.get(tool_name)
            if tool_class:
                tool = tool_class()
                tool_descriptions.append(f"- {tool_name}: {tool.description}")

        prompt = f"""Given this text and its factual claims, decide which tool (if any) to use for verification.

Text: {text}

Factual claims to verify:
{chr(10).join(f"- {claim}" for claim in factual_claims)}

Available tools:
{chr(10).join(tool_descriptions)}

For each claim, provide:
1. Whether to use tools (true/false)
2. Which tool to use (tool name)
3. The search query to use for that claim

Respond in JSON format:
{{
    "use_tools": true/false,
    "tool_name": "selected_tool_name",
    "queries": {{
        "claim text": "search query",
        ...
    }}
}}"""

        try:

            class ToolDecision(BaseModel):
                tool: str
                rationale: str

            agent = self.client.create_agent(
                system_prompt="You are a tool selection expert. Choose the most appropriate tool for fact-checking.",
                result_type=ToolDecision,
            )

            result = await agent.run(prompt)
            output = result.output
            if hasattr(output, "model_dump"):
                return output.model_dump()  # type: ignore[no-any-return]
            elif isinstance(output, dict):  # type: ignore[unreachable]
                return output  # type: ignore[unreachable]
            else:
                return None
        except Exception:
            # Default to web search if LLM fails
            return {
                "use_tools": True,
                "tool_name": "web_search",
                "queries": {claim: claim for claim in factual_claims},
            }

    async def _search_with_tool(
        self, tool_instance: Any, claim: str, search_query: str
    ) -> Optional[Dict[str, Any]]:
        """Search using the provided tool."""
        start_time = time.time()

        try:
            results = await tool_instance(search_query)
            processing_time = time.time() - start_time

            tool_usage = ToolUsage(
                tool_name=tool_instance.name,
                status="success",
                input_data=search_query,
                result_count=len(results) if results else 0,
                processing_time=processing_time,
                metadata={
                    "search_results": results[:3] if results else [],
                    "original_claim": claim,
                },
            )

            return {
                "tool_usage": tool_usage,
                "search_results": results,
                "original_claim": claim,
                "search_query": search_query,
            }

        except Exception as e:
            processing_time = time.time() - start_time

            tool_usage = ToolUsage(
                tool_name=tool_instance.name,
                status="failure",
                input_data=search_query,
                error_message=str(e),
                processing_time=processing_time,
                metadata={
                    "original_claim": claim,
                    "error_type": type(e).__name__,
                },
            )

            return {
                "tool_usage": tool_usage,
                "search_results": [],
                "original_claim": claim,
                "search_query": search_query,
                "error": str(e),
            }

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for Self-RAG critique."""
        previous_context = self._get_previous_context(result)

        user_prompt = f"""You are a Self-RAG critic using reflection tokens to evaluate content quality.

Text to evaluate:
{text}
{previous_context}

Apply Self-RAG's reflection framework:

1. **Overall Assessment** (Reflection Tokens):
   - ISREL: Is the content relevant to its intended purpose?
   - ISSUP: Is the content supported by evidence or verifiable information?
   - ISUSE: Is the content useful and valuable for readers?

2. **Detailed Analysis**:
   - Identify ALL factual claims that could be verified
   - Find retrieval opportunities where external information would enhance quality
   - Evaluate the overall information sufficiency

3. **Adaptive Reflection**:
   - Where would retrieval most improve the content?
   - Which claims critically need verification?
   - How can the content better serve its purpose?

START YOUR RESPONSE WITH THE REFLECTION TOKENS:
- ISREL: [YES/NO]
- ISSUP: [YES/PARTIAL/NO]
- ISUSE: [YES/PARTIAL/NO]

Then provide detailed feedback. List all factual claims that should be verified."""

        return [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

    async def _create_messages_with_tools(
        self, text: str, result: SifakaResult, tool_results: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Create messages that include tool search results."""
        previous_context = self._get_previous_context(result)

        # Format tool results
        tool_context = "\n\n**Fact-Checking Results:**\n"
        for tool_result in tool_results:
            claim = tool_result.get("original_claim", "")
            tool_context += f"\nClaim: {claim}\n"

            search_results = tool_result.get("search_results", [])
            if search_results:
                tool_context += "Found information:\n"
                for i, sr in enumerate(search_results[:3], 1):
                    title = sr.get("title", "")
                    snippet = sr.get("snippet", "")
                    tool_context += f"  {i}. {title}\n"
                    tool_context += f"     {snippet}\n"
            else:
                tool_context += "  No relevant information found.\n"

        user_prompt = f"""You are a Self-RAG critic using reflection tokens to evaluate content quality.

Text to evaluate:
{text}
{previous_context}
{tool_context}

Apply Self-RAG's reflection framework:

1. **Overall Assessment** (Reflection Tokens):
   - ISREL: Is the content relevant to its intended purpose?
   - ISSUP: Is the content supported by the fact-checking results above?
   - ISUSE: Is the content useful and valuable for readers?

2. **Detailed Analysis**:
   - Compare EACH claim in the text against the fact-checking results
   - Identify specific inaccuracies based on the search results
   - Note which claims are supported and which contradict the evidence

3. **Specific Corrections** (REQUIRED):
   For EACH factual claim, you MUST:
   - State whether it's CORRECT or INCORRECT based on search results
   - If INCORRECT, quote the exact correction from the search results
   - Example: "The claim 'X is Y' is INCORRECT. According to [source]: 'X is actually Z'"

START YOUR RESPONSE WITH THE REFLECTION TOKENS:
- ISREL: [YES/NO]
- ISSUP: [YES/PARTIAL/NO] (NO if facts are wrong)
- ISUSE: [YES/PARTIAL/NO]

Then provide your analysis. You MUST include specific corrections for any incorrect claims, quoting directly from the search results provided above."""

        return [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

    async def _perform_critique(self, messages: List[Dict[str, str]]) -> CritiqueResult:
        """Perform the actual critique using the LLM."""
        try:
            agent = self.client.create_agent(
                system_prompt=self._get_system_prompt(),
                result_type=self._get_response_type(),
            )

            start_time = time.time()
            agent_result = await agent.run(messages[1]["content"])
            processing_time = time.time() - start_time

            response = agent_result.output

            # Get usage data
            tokens_used = 0
            try:
                if hasattr(agent_result, "usage"):
                    usage = agent_result.usage()
                    if usage and hasattr(usage, "total_tokens"):
                        tokens_used = getattr(usage, "total_tokens", 0)
            except Exception:
                tokens_used = 0

            # Handle case where response is a string
            if isinstance(response, str):
                # Create a basic response
                return CritiqueResult(
                    critic=self.name,
                    feedback=response,
                    suggestions=[],
                    needs_improvement=True,
                    confidence=0.7,
                    metadata={
                        "retrieval_opportunities": [],
                        "factual_claims": [],
                        "tokens_used": tokens_used,
                        "processing_time": processing_time,
                    },
                )

            # Build feedback for structured response
            feedback_parts = [  # type: ignore[unreachable]
                f"ISREL: {getattr(response, 'isrel', 'N/A')}",
                f"ISSUP: {getattr(response, 'issup', 'N/A')}",
                f"ISUSE: {getattr(response, 'isuse', 'N/A')}",
                "",
                getattr(response, "overall_assessment", ""),
            ]

            specific_issues = getattr(response, "specific_issues", None)
            if specific_issues:
                feedback_parts.append("\nSpecific issues:")
                for issue in specific_issues:
                    feedback_parts.append(f"- {issue}")

            specific_corrections = getattr(response, "specific_corrections", None)
            if specific_corrections:
                feedback_parts.append("\nFactual corrections needed:")
                for correction in specific_corrections:
                    feedback_parts.append(f"- {correction}")

            return CritiqueResult(
                critic=self.name,
                feedback="\n".join(feedback_parts),
                suggestions=getattr(response, "improvement_suggestions", []) or [],
                needs_improvement=getattr(response, "needs_improvement", True),
                confidence=getattr(response, "confidence_score", 0.7),
                metadata={
                    "retrieval_opportunities": getattr(
                        response, "retrieval_opportunities", []
                    )
                    or [],
                    "factual_claims": getattr(response, "factual_claims", []) or [],
                },
                timestamp=datetime.now(),
                model_used=self.model,
                temperature_used=self.temperature,
                prompt_sent=messages[1]["content"],
                tokens_used=tokens_used,
                processing_time=processing_time,
                tools_used=[],
            )

        except Exception as e:
            return CritiqueResult(
                critic=self.name,
                feedback=f"Error during critique: {e!s}",
                suggestions=["Review the text manually"],
                needs_improvement=True,
                confidence=0.0,
                timestamp=datetime.now(),
                model_used=self.model,
                temperature_used=self.temperature,
                prompt_sent=str(messages),
                tokens_used=0,
                processing_time=(
                    time.time() - start_time if "start_time" in locals() else 0
                ),
                tools_used=[],
            )

    def _get_system_prompt(self) -> str:
        """Get system prompt for Self-RAG critic."""
        return """You are an expert Self-RAG critic implementing reflection-augmented generation.
Your role is to evaluate text quality using reflection tokens and identify opportunities
for improvement through retrieval. Focus on factual accuracy, information completeness,
and overall utility for readers."""
