"""Self-RAG critic implementation.

Based on: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
Paper: https://arxiv.org/abs/2310.11511
Authors: Asai et al. (2023)

Self-RAG enables models to reflect on their outputs and determine when
additional retrieval would improve content quality.

## Similarity to Original Paper:
- ADAPTED: Retrieval decision without actual retrieval implementation
- PRESERVED: Self-reflection on content support and utility
- PRESERVED: Evaluation of when external information needed
- SIMPLIFIED: No actual RAG pipeline; focuses on identifying needs

## Implementation Choices:
1. 4 evaluation aspects: Retrieval need, Relevance, Support, Utility
2. Binary YES/NO decision for retrieval needs
3. Uses older Critic interface (not BaseCritic) - simpler structure
4. Confidence based on how many aspects need improvement
5. Flags content that would benefit from external sources

## Why This Approach:
- Identifies when text lacks factual support or references
- Useful for flagging content needing fact-checking
- Simpler than implementing full retrieval pipeline
- Can be used to guide human reviewers to add sources
- Preserves the "self-aware of knowledge gaps" concept

"""

from typing import List, Optional, Union

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .base import BaseCritic, CriticConfig, create_prompt_with_format, CriticResponse


class SelfRAGCritic(BaseCritic):
    """Implements Self-RAG approach for retrieval-augmented critique."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.4,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[CriticConfig] = None,
    ):
        # Initialize with custom config for self-rag
        if config is None:
            config = CriticConfig(
                response_format="structured",
                base_confidence=0.7,
                context_weight=0.1,
                depth_weight=0.2,
            )
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        return "self_rag"

    async def _generate_critique(self, text: str, result: SifakaResult) -> str:
        """Generate critique focusing on retrieval needs."""
        base_prompt = f"""Using the Self-RAG technique, evaluate this text for factual accuracy, completeness, and retrieval needs:

Text to assess:
{text}

Please evaluate:
1. RETRIEVAL NEED: Does this text need additional information retrieval?
2. RELEVANCE: Are the current facts relevant and well-supported?
3. SUPPORT: Is the content well-supported by evidence?
4. UTILITY: How useful and complete is the current information?

Consider these aspects:
- Factual accuracy and potential knowledge gaps
- Currency of information and need for updates  
- Completeness of coverage for the topic
- Evidence quality and source reliability"""

        prompt = create_prompt_with_format(
            base_prompt, self.config.response_format, include_examples=False
        )

        # Add format-specific instructions
        if self.config.response_format == "structured":
            prompt += """\n\nFormat your response as:
RETRIEVAL_NEEDED: [YES/NO and explanation]
FEEDBACK: [Detailed evaluation of factual accuracy and completeness]
SUGGESTIONS:
1. [Specific improvement or retrieval query]
2. [Another suggestion]"""
        
        # Get evaluation from model
        response = await self.client.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert text critic using the Self-RAG technique for retrieval-augmented evaluation.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return response.content

    def _parse_structured_response(self, response: str) -> CriticResponse:
        """Parse Self-RAG specific structured response."""
        # First try the base parsing
        base_response = super()._parse_structured_response(response)
        
        # Enhance with Self-RAG specific parsing
        retrieval_needed = False
        if "RETRIEVAL_NEEDED:" in response:
            retrieval_line = response.split("RETRIEVAL_NEEDED:")[1].split("\n")[0]
            retrieval_needed = "YES" in retrieval_line.upper()
        
        # Update metadata
        base_response.metadata.update({
            "retrieval_needed": retrieval_needed,
            "aspects_evaluated": ["retrieval_need", "relevance", "support", "utility"]
        })
        
        # Override needs_improvement based on retrieval decision
        base_response.needs_improvement = retrieval_needed
        
        # Adjust confidence based on retrieval need
        if retrieval_needed:
            base_response.confidence = min(0.9, base_response.confidence + 0.1)
        
        return base_response

    def _calculate_confidence(self, feedback: str, full_response: str) -> float:
        """Calculate confidence with Self-RAG specific logic."""
        # Start with base confidence calculation
        confidence = super()._calculate_confidence(feedback, full_response)
        
        # Boost confidence if retrieval assessment is clear
        retrieval_indicators = ["retrieval needed", "needs retrieval", "requires additional", "missing information"]
        if any(ind in feedback.lower() for ind in retrieval_indicators):
            confidence = min(1.0, confidence + 0.1)
        
        return confidence
