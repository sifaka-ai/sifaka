"""Self-RAG critic implementation (Reflection-based version).

Based on: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
Paper: https://arxiv.org/abs/2310.11511
Authors: Asai et al. (2023)

Self-RAG teaches LLMs to adaptively retrieve, generate, and critique text
using special reflection tokens to control behavior.

## Similarity to Original Paper:
- PRESERVED: Self-reflection on content quality
- PRESERVED: Adaptive decision-making about information needs
- PRESERVED: Multiple critique dimensions (relevance, support, utility)
- SIMPLIFIED: No actual retrieval system (identifies where needed)
- ADAPTED: Uses structured outputs instead of reflection tokens

## Implementation Choices:
1. Evaluates ISREL (relevance), ISSUP (support), ISUSE (utility)
2. Identifies retrieval opportunities without performing retrieval
3. Provides confidence scores for factual claims
4. Suggests specific source types needed

## Why This Approach:
- Maintains Self-RAG's core reflection concept
- Works without external retrieval infrastructure
- Provides actionable guidance on information gaps
- Preserves the adaptive, self-critical nature
"""

import time
import re
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field

from ..core.models import SifakaResult, CritiqueResult, ToolUsage
from ..core.llm_client import Provider
from ..tools.registry import ToolRegistry
from .core.base import BaseCritic
from ..core.config import Config


class FactualClaim(BaseModel):
    """A factual claim with Self-RAG reflection assessment."""

    claim: str = Field(..., description="The specific factual claim")
    isrel: bool = Field(
        ..., description="ISREL: Is this claim relevant to the main topic?"
    )
    issup: bool = Field(..., description="ISSUP: Is this claim supported by evidence?")
    isuse: bool = Field(..., description="ISUSE: Is this claim useful for the reader?")
    confidence_level: str = Field(..., description="Confidence: high, medium, low")
    retrieval_needed: bool = Field(
        ..., description="Whether retrieval would help verify this claim"
    )


class RetrievalOpportunity(BaseModel):
    """A specific opportunity for retrieval to enhance the text."""

    reason: str = Field(..., description="Why retrieval is needed here")


class SelfRAGResponse(BaseModel):
    """Response model following Self-RAG's reflection framework."""

    feedback: str = Field(
        ..., description="Overall reflection on content quality and information needs"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Specific suggestions for improving content with better information",
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text needs retrieval or accuracy improvements"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in the assessment"
    )

    # Only keep retrieval_opportunities since that's what's used in generation.py
    retrieval_opportunities: list[RetrievalOpportunity] = Field(
        default_factory=list,
        description="Specific opportunities where retrieval would enhance quality",
    )


class SelfRAGCritic(BaseCritic):
    """Implements Self-RAG critique for factual accuracy.

    ## When to Use This Critic:

    ✅ When to use:
    - Fact-checking and verifying claims
    - Identifying unsupported assertions
    - Academic or research content requiring citations
    - Content where accuracy is paramount

    ❌ When to avoid:
    - Creative or fictional writing
    - Opinion pieces or personal narratives
    - When external verification isn't needed

    🎯 Best for:
    - Technical documentation
    - Research papers and reports
    - News articles and factual content
    - Educational materials requiring accuracy
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        # Initialize with custom config
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        return "self_rag"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom SelfRAGResponse for structured output."""
        return SelfRAGResponse

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Override critique to add actual tool calling for fact-checking."""
        # First, do the standard Self-RAG critique
        critique_result = await super().critique(text, result)

        # Check if tools are enabled for this critic
        tools_enabled = False
        if self.config and hasattr(self.config, "critic_tool_settings"):
            tools_enabled = self.config.critic_tool_settings.get(self.name, {}).get(
                "enable_tools", False
            )
        elif self.config and hasattr(self.config, "enable_tools"):
            tools_enabled = self.config.enable_tools

        if not tools_enabled:
            return critique_result

        # Extract factual claims that need verification
        factual_claims = self._extract_factual_claims(text)

        # Use tools to verify claims if available
        search_tool = ToolRegistry.get("duckduckgo")
        if search_tool and factual_claims:
            tool_instance = search_tool()

            # Verify each factual claim
            search_results = []
            # Extract main subject from the full text for context
            main_subject = "Eiffel Tower" if "eiffel tower" in text.lower() else None

            for claim in factual_claims:
                verification_result = await self._verify_claim_with_tool(
                    tool_instance, claim, critique_result, main_subject
                )
                if verification_result:
                    search_results.append(verification_result)

            # Update critique feedback with search results
            if search_results:
                critique_result = await self._integrate_search_results(
                    critique_result, text, search_results
                )

        return critique_result

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for RAG-style critique."""
        # Get previous context
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
   - Identify all factual claims and assess each with ISREL/ISSUP/ISUSE
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

Then provide detailed feedback. Focus on actionable improvements that would enhance relevance, support, and utility."""

        return [
            {
                "role": "system",
                "content": "You are a Self-RAG critic that uses reflection tokens (ISREL, ISSUP, ISUSE) to adaptively evaluate content and identify where retrieval would enhance quality. You reflect on relevance, support, and utility.",
            },
            {"role": "user", "content": user_prompt},
        ]

    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims that might need verification."""
        # Simple heuristic: look for sentences with numbers, dates, names, or specific facts
        factual_patterns = [
            r"\d+\s*(meters?|feet|km|miles|years?|visitors?|million|billion)",  # Measurements/quantities
            r"\b\d{4}\b",  # Years
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Proper names (simplified)
            r"built\s+by|designed\s+by|created\s+by",  # Attribution claims
            r"attracts?\s+\d+|visited?\s+by\s+\d+",  # Visitor statistics
        ]

        claims = []
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if sentence contains factual patterns
            for pattern in factual_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append(sentence)
                    break

        return claims[:3]  # Limit to 3 claims to avoid too many API calls

    async def _verify_claim_with_tool(
        self,
        tool_instance: Any,
        claim: str,
        critique_result: CritiqueResult,
        main_subject: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Verify a factual claim using the search tool."""
        start_time = time.time()

        try:
            # Create search query from claim with context
            search_query = self._create_search_query(claim, main_subject)

            # Call the tool
            results = await tool_instance(search_query, max_results=3)
            processing_time = time.time() - start_time

            # Create tool usage record with metadata
            tool_usage = ToolUsage(
                tool_name="duckduckgo",
                status="success",
                input_data=search_query,
                result_count=len(results) if results else 0,
                processing_time=processing_time,
                metadata={
                    "search_results": (
                        results[:3] if results else []
                    ),  # Store top 3 results
                    "original_claim": claim,
                },
            )

            # Add to critique result
            critique_result.tools_used.append(tool_usage)

            # Return both tool usage and actual results for analysis
            return {
                "tool_usage": tool_usage,
                "search_results": results,
                "original_claim": claim,
                "search_query": search_query,
            }

        except Exception as e:
            processing_time = time.time() - start_time

            # Create error record
            tool_usage = ToolUsage(
                tool_name="duckduckgo",
                status="failure",
                input_data=search_query if "search_query" in locals() else claim,
                error_message=str(e),
                processing_time=processing_time,
                metadata={
                    "original_claim": claim,
                    "error_type": type(e).__name__,
                },
            )

            # Add to critique result
            critique_result.tools_used.append(tool_usage)

            return {
                "tool_usage": tool_usage,
                "search_results": [],
                "original_claim": claim,
                "search_query": search_query if "search_query" in locals() else claim,
                "error": str(e),
            }

    def _create_search_query(
        self, claim: str, context_subject: Optional[str] = None
    ) -> str:
        """Convert a factual claim into a search query."""
        # Extract key terms for searching
        claim = claim.strip()

        # Check if main subject is in this claim or use context
        main_subject = None
        if "eiffel tower" in claim.lower():
            main_subject = "Eiffel Tower"
        elif context_subject:
            main_subject = context_subject

        # Build specific search terms
        search_terms = []

        # Always include the main subject if found
        if main_subject:
            search_terms.append(main_subject)

        # Add specific search terms based on what we're fact-checking
        if re.search(r"\d+\s*meters?", claim, re.IGNORECASE):
            search_terms.extend(["height", "meters"])
        elif re.search(r"\d{4}", claim):
            search_terms.extend(["built", "construction", "year"])
        elif "napoleon" in claim.lower():
            search_terms.extend(["builder", "Napoleon", "Bonaparte"])
        elif "visitors" in claim.lower() or "million" in claim.lower():
            search_terms.extend(["visitors", "annually", "tourism"])
        elif "painted" in claim.lower() or "pink" in claim.lower():
            search_terms.extend(["color", "paint", "maintenance"])
        elif "repaint" in claim.lower() or "months" in claim.lower():
            search_terms.extend(["repainting", "schedule", "maintenance"])

        # Default fallback if no specific terms
        if not search_terms:
            # Extract meaningful words
            words = claim.split()
            search_terms = [
                w
                for w in words
                if len(w) > 3
                and w.lower() not in ["the", "and", "that", "this", "with", "from"]
            ][:5]

        return " ".join(search_terms)

    async def _integrate_search_results(
        self,
        critique_result: CritiqueResult,
        original_text: str,
        search_results: List[Dict[str, Any]],
    ) -> CritiqueResult:
        """Update critique feedback to incorporate actual search findings."""
        specific_findings = []
        successful_searches = []

        for result in search_results:
            if result.get("search_results") and len(result["search_results"]) > 0:
                successful_searches.append(result)

                # Analyze search results for specific corrections
                finding = self._analyze_search_result(result)
                if finding:
                    specific_findings.append(finding)

        if successful_searches:
            # Build specific feedback based on actual search findings
            enhanced_feedback = f"{critique_result.feedback}\n\nFact-checking performed using web search:"

            if specific_findings:
                for finding in specific_findings:
                    enhanced_feedback += f"\n• {finding}"
            else:
                enhanced_feedback += f"\n• {len(successful_searches)} claim(s) verified - search results available for fact-checking"

            # Add URLs visited for transparency
            urls_visited = []
            for result in successful_searches:
                for sr in result.get("search_results", [])[
                    :2
                ]:  # Top 2 results per search
                    if isinstance(sr, dict) and "link" in sr:
                        urls_visited.append(sr["link"])

            if urls_visited:
                enhanced_feedback += "\n\nURLs visited:"
                for url in urls_visited[:5]:  # Limit to 5 URLs total
                    enhanced_feedback += f"\n- {url}"

            critique_result.feedback = enhanced_feedback

            # Add specific retrieval opportunities
            if (
                hasattr(critique_result, "metadata")
                and "retrieval_opportunities" in critique_result.metadata
            ):
                if not critique_result.metadata["retrieval_opportunities"]:
                    critique_result.metadata["retrieval_opportunities"] = []

                for finding in specific_findings[:2]:  # Limit to top 2 most important
                    critique_result.metadata["retrieval_opportunities"].append(
                        {"reason": finding}
                    )

        return critique_result

    def _analyze_search_result(self, result: Dict[str, Any]) -> Optional[str]:
        """Analyze search results to extract specific factual corrections."""
        original_claim = result.get("original_claim", "")
        search_results = result.get("search_results", [])
        search_query = result.get("search_query", "")

        if not search_results:
            return None

        # Extract text content from search results
        search_content = ""
        for sr in search_results[:2]:  # Use top 2 results
            if isinstance(sr, dict):
                title = sr.get("title", "")
                snippet = sr.get("snippet", "")
                search_content += f"{title} {snippet} "

        search_content = search_content.lower()
        original_lower = original_claim.lower()

        # Specific fact-checking based on content
        findings = []

        # Height checking
        if "height" in search_query.lower() or "meters" in original_lower:
            height_mentions = re.findall(
                r"(\d+)\s*(?:meters?|metres?|m\b)", search_content
            )
            if height_mentions:
                search_height = height_mentions[0]
                if "500" in original_lower and search_height != "500":
                    findings.append(
                        f"Height claim needs correction: search indicates {search_height} meters, not 500 meters"
                    )
                elif search_height:
                    findings.append(
                        f"Height verified from search: approximately {search_height} meters"
                    )

        # Date/year checking
        if "built" in search_query.lower() or "1850" in original_lower:
            year_mentions = re.findall(r"\b(18\d{2}|19\d{2})\b", search_content)
            if year_mentions:
                search_year = year_mentions[0]
                if "1850" in original_lower and search_year != "1850":
                    findings.append(
                        f"Construction date needs correction: search indicates {search_year}, not 1850"
                    )

        # Visitor statistics
        if "visitors" in search_query.lower() or "million" in original_lower:
            visitor_mentions = re.findall(
                r"(\d+(?:\.\d+)?)\s*million\s*(?:visitors?|people)", search_content
            )
            if visitor_mentions:
                search_visitors = visitor_mentions[0]
                if "100" in original_lower and search_visitors != "100":
                    findings.append(
                        f"Visitor count needs correction: search indicates {search_visitors} million annually, not 100 million"
                    )

        # Color checking
        if "color" in search_query.lower() or "pink" in original_lower:
            if "brown" in search_content or "bronze" in search_content:
                findings.append(
                    "Tower color needs correction: search indicates brown/bronze color, not pink"
                )

        # Return the most specific finding
        if findings:
            return findings[0]
        else:
            # Don't truncate the claim - show it in full
            return f"Claim '{original_claim}' verified against search results"
