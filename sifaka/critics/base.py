"""Base classes and utilities for critics."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel, Field, ConfigDict
import re
import json
from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.core.interfaces import Critic
from sifaka.core.llm_client import LLMClient, LLMManager, Provider


class CriticResponse(BaseModel):
    """Standardized response format for all critics."""

    model_config = ConfigDict(extra="forbid")

    feedback: str = Field(..., description="Main feedback about the text")
    suggestions: List[str] = Field(
        default_factory=list, description="Specific improvement suggestions"
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text needs improvement"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the assessment"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional critic-specific data"
    )


class CriticConfig(BaseModel):
    """Configuration for critic behavior."""

    model_config = ConfigDict(extra="forbid")

    # Confidence calculation weights
    base_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    context_weight: float = Field(default=0.1, ge=0.0, le=0.5)
    depth_weight: float = Field(default=0.15, ge=0.0, le=0.5)
    domain_weight: float = Field(default=0.05, ge=0.0, le=0.5)

    # Common parameters
    feedback_truncation_length: int = Field(default=200, ge=50, le=1000)
    context_window_size: int = Field(default=3, ge=1, le=10)
    min_word_length: int = Field(default=3, ge=1, le=10)
    common_word_threshold: int = Field(default=5, ge=1, le=20)
    
    # Rating scales
    rating_scale_min: int = Field(default=1, ge=0, le=1)
    rating_scale_max: int = Field(default=5, ge=5, le=10)
    quality_score_scale_max: int = Field(default=5, ge=5, le=10)
    
    # Thresholds
    consensus_threshold: float = Field(default=0.75, ge=0.5, le=1.0)
    consistency_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    reliability_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    quality_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    severe_violation_threshold: int = Field(default=4, ge=3, le=5)
    low_score_threshold: int = Field(default=2, ge=1, le=3)
    suggestion_count_threshold: int = Field(default=3, ge=1, le=10)
    
    # Variance and normalization
    max_variance: float = Field(default=0.5, ge=0.1, le=1.0)
    variance_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    consensus_weight: float = Field(default=0.7, ge=0.5, le=1.0)
    agreement_weight: float = Field(default=0.3, ge=0.0, le=0.5)
    
    # Coverage and reliability multipliers
    coverage_multiplier_base: float = Field(default=0.8, ge=0.5, le=1.0)
    coverage_multiplier_range: float = Field(default=0.2, ge=0.1, le=0.5)
    confidence_boost: float = Field(default=0.1, ge=0.0, le=0.3)
    confidence_multiplier: float = Field(default=0.9, ge=0.5, le=1.0)

    # Parsing patterns
    feedback_markers: List[str] = Field(
        default=["REFLECTION:", "ASSESSMENT:", "FEEDBACK:", "ANALYSIS:"]
    )
    suggestion_markers: List[str] = Field(
        default=["SUGGESTIONS:", "IMPROVEMENTS:", "RECOMMENDATIONS:"]
    )

    # Response format preference
    response_format: str = Field(default="json", pattern="^(json|structured|text)$")
    
    # Temperature overrides for specific critics
    constitutional_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    self_consistency_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    n_critics_temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    meta_rewarding_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    
    # Default values for specific critics
    self_consistency_num_samples: int = Field(default=3, ge=2, le=10)
    
    # Default dimensions/principles/perspectives
    constitutional_principles: List[str] = Field(
        default=[
            "Clarity: Is the text clear and easy to understand?",
            "Accuracy: Are the claims factually correct and well-supported?", 
            "Completeness: Does the text fully address the topic?",
            "Objectivity: Is the text balanced and unbiased?",
            "Engagement: Is the text interesting and engaging?",
            "Structure: Is the text well-organized and coherent?",
            "Appropriateness: Is the tone and style suitable for the audience?"
        ]
    )
    
    self_refine_dimensions: List[str] = Field(
        default=[
            "clarity and coherence",
            "accuracy and factual correctness",
            "completeness and depth",
            "engagement and readability",
            "structure and organization",
            "grammar and style",
            "relevance and focus"
        ]
    )
    
    n_critics_perspectives: List[str] = Field(
        default=[
            "A technical expert focused on accuracy and precision",
            "A general reader focused on clarity and accessibility",
            "An editor focused on structure and flow",
            "A subject matter expert focused on completeness and depth"
        ]
    )
    
    self_rag_retrieval_indicators: List[str] = Field(
        default=[
            "specific data", "statistics", "research findings", "expert opinions"
        ]
    )
    
    # Prompt template paths (optional)
    prompt_templates: Dict[str, str] = Field(default_factory=dict)


class BaseCritic(Critic, ABC):
    """Enhanced base critic with standardized parsing."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        config: Optional[CriticConfig] = None,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
    ):
        self.config = config or CriticConfig()
        self.model = model
        
        # Use critic-specific temperature if available and not overridden
        if temperature is not None:
            self.temperature = temperature
        else:
            # Check for critic-specific temperature in config
            critic_temp_map = {
                "ConstitutionalCritic": self.config.constitutional_temperature,
                "SelfConsistencyCritic": self.config.self_consistency_temperature,
                "NCriticsCritic": self.config.n_critics_temperature,
                "MetaRewardingCritic": self.config.meta_rewarding_temperature,
            }
            self.temperature = critic_temp_map.get(self.__class__.__name__, 0.7)
        
        self.provider = provider
        self._api_key = api_key
        self._client: Optional[LLMClient] = None
    
    @property
    def client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._client is None:
            self._client = LLMManager.get_client(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                api_key=self._api_key
            )
        return self._client

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the critic's name."""
        pass

    @abstractmethod
    async def _generate_critique(self, text: str, result: SifakaResult) -> str:
        """Generate raw critique response from the model."""
        pass
    
    async def _call_llm(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Call the LLM and return the response with optional prompt logging."""
        # Log the full prompt if requested
        if hasattr(self, '_show_prompts') and self._show_prompts:
            print(f"\n{'='*80}")
            print(f"CRITIC PROMPT ({self.name})")
            print(f"{'='*80}")
            for msg in messages:
                print(f"\n[{msg['role'].upper()}]")
                print(msg['content'])
            print(f"{'='*80}\n")
        
        response = await self.client.complete(messages, **kwargs)
        return response.content
    
    def _get_previous_feedback_context(self, result: SifakaResult) -> str:
        """Get context from previous critiques to avoid repetition."""
        if not result.critiques:
            return ""
        
        previous_feedback = []
        previous_suggestions = set()
        
        # Get all previous feedback from this critic
        critiques_list = list(result.critiques)
        for critique in critiques_list:
            if critique.critic == self.name:
                previous_feedback.append(f"- {critique.feedback}")
                previous_suggestions.update(critique.suggestions)
        
        if not previous_feedback:
            return ""
        
        context = "\n\nPREVIOUS FEEDBACK ALREADY PROVIDED:\n"
        context += "\n".join(previous_feedback)
        
        if previous_suggestions:
            context += "\n\nSUGGESTIONS ALREADY MADE:\n"
            context += "\n".join(f"- {s}" for s in previous_suggestions)
            context += "\n\nPlease provide NEW insights and avoid repeating previous feedback."
        
        return context
    
    def get_system_prompt(self) -> str:
        """Get the system prompt used by this critic."""
        # Check various places where system prompt might be stored
        if hasattr(self, 'system_prompt'):
            return self.system_prompt
        elif hasattr(self, 'SYSTEM_PROMPT'):
            return self.SYSTEM_PROMPT
        return "You are an expert text critic."

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Standardized critique method."""
        try:
            # Generate critique
            raw_response = await self._generate_critique(text, result)

            # Parse response based on format
            if self.config.response_format == "json":
                critic_response = self._parse_json_response(raw_response)
            elif self.config.response_format == "structured":
                critic_response = self._parse_structured_response(raw_response)
            else:
                critic_response = self._parse_text_response(raw_response)

            # Convert to CritiqueResult
            return CritiqueResult(
                critic=self.name,
                feedback=critic_response.feedback,
                suggestions=critic_response.suggestions,
                needs_improvement=critic_response.needs_improvement,
                confidence=critic_response.confidence,
                metadata=critic_response.metadata,
            )

        except Exception as e:
            # Fallback for errors
            return CritiqueResult(
                critic=self.name,
                feedback=f"Error during critique: {str(e)}",
                suggestions=["Please review the text manually"],
                needs_improvement=True,
                confidence=0.0,
            )

    def _parse_json_response(self, response: str) -> CriticResponse:
        """Parse JSON-formatted response."""
        try:
            # Extract JSON from response if wrapped in markdown
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
            )
            if json_match:
                response = json_match.group(1)

            data = json.loads(response)
            return CriticResponse(**data)
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback to text parsing
            return self._parse_text_response(response)

    def _parse_structured_response(self, response: str) -> CriticResponse:
        """Parse structured text response with markers."""
        feedback = ""
        suggestions = []

        # Extract feedback
        for marker in self.config.feedback_markers:
            pattern = (
                rf"{marker}\s*(.+?)(?={('|'.join(self.config.suggestion_markers))}|$)"
            )
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                feedback = match.group(1).strip()
                break

        if not feedback:
            # Take first paragraph as feedback
            feedback = response.split("\n\n")[0].strip()

        # Extract suggestions
        for marker in self.config.suggestion_markers:
            pattern = rf"{marker}\s*(.+?)$"
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                suggestion_text = match.group(1).strip()
                # Parse numbered or bulleted list
                suggestions = self._extract_list_items(suggestion_text)
                break

        # Determine if improvement needed
        needs_improvement = self._assess_needs_improvement(feedback, suggestions)

        # Calculate confidence
        confidence = self._calculate_confidence(feedback, response)

        return CriticResponse(
            feedback=feedback,
            suggestions=suggestions,
            needs_improvement=needs_improvement,
            confidence=confidence,
        )

    def _parse_text_response(self, response: str) -> CriticResponse:
        """Parse unstructured text response."""
        # Use the entire response as feedback
        feedback = response.strip()

        # Try to extract any list-like suggestions
        suggestions = self._extract_list_items(response)

        # If no explicit suggestions, create a generic one
        if not suggestions:
            suggestions = ["Consider the feedback provided above"]

        needs_improvement = self._assess_needs_improvement(feedback, suggestions)
        confidence = self._calculate_confidence(feedback, response)

        return CriticResponse(
            feedback=feedback,
            suggestions=suggestions,
            needs_improvement=needs_improvement,
            confidence=confidence,
        )

    def _extract_list_items(self, text: str) -> List[str]:
        """Extract numbered or bulleted list items from text."""
        items = []

        # Match numbered lists (1. 2. etc)
        numbered_pattern = r"^\s*\d+[\.)]\s*(.+)$"
        # Match bullet points (-, *, •)
        bullet_pattern = r"^\s*[-*•]\s*(.+)$"

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            match = re.match(numbered_pattern, line) or re.match(bullet_pattern, line)
            if match:
                items.append(match.group(1).strip())
            elif items and not line.startswith((" ", "\t")):
                # Stop if we hit a non-list line after starting a list
                break
            elif items and line.startswith((" ", "\t")):
                # Continuation of previous item
                items[-1] += " " + line.strip()

        return items

    def _assess_needs_improvement(self, feedback: str, suggestions: List[str]) -> bool:
        """Assess whether the text needs improvement based on feedback."""
        feedback_lower = feedback.lower()

        # Negative indicators
        negative_indicators = [
            "needs improvement",
            "could be better",
            "should",
            "lacking",
            "insufficient",
            "weak",
            "poor",
            "missing",
            "unclear",
            "confusing",
            "needs more",
            "requires",
            "must",
            "fix",
            "revise",
            "rework",
        ]

        # Positive indicators
        positive_indicators = [
            "excellent",
            "perfect",
            "outstanding",
            "no issues",
            "well-written",
            "comprehensive",
            "clear",
            "effective",
            "strong",
            "solid",
        ]

        # Count indicators
        negative_count = sum(1 for ind in negative_indicators if ind in feedback_lower)
        positive_count = sum(1 for ind in positive_indicators if ind in feedback_lower)

        # More than one suggestion usually means improvement needed
        if len(suggestions) > 1:
            return True

        # If more negative than positive indicators
        if negative_count > positive_count:
            return True

        # If strongly positive with few suggestions
        if positive_count >= 2 and len(suggestions) <= 1:
            return False

        # Default to True if uncertain
        return True

    def _calculate_confidence(self, feedback: str, full_response: str) -> float:
        """Calculate confidence score based on response characteristics."""
        confidence = self.config.base_confidence

        # Adjust based on response length (longer = more thorough)
        if len(full_response) > 500:
            confidence += 0.1
        elif len(full_response) < 100:
            confidence -= 0.1

        # Adjust based on specificity indicators
        specificity_indicators = [
            "specifically",
            "particularly",
            "exactly",
            "precisely",
            "clearly",
            "definitely",
            "certainly",
        ]
        specificity_count = sum(
            1 for ind in specificity_indicators if ind in feedback.lower()
        )
        confidence += min(specificity_count * 0.05, 0.15)

        # Adjust based on uncertainty indicators
        uncertainty_indicators = [
            "might",
            "maybe",
            "perhaps",
            "possibly",
            "could be",
            "seems",
            "appears",
            "somewhat",
            "relatively",
        ]
        uncertainty_count = sum(
            1 for ind in uncertainty_indicators if ind in feedback.lower()
        )
        confidence -= min(uncertainty_count * 0.05, 0.15)

        # Ensure confidence stays in valid range
        return max(0.0, min(1.0, confidence))


def create_prompt_with_format(
    base_prompt: str, response_format: str = "json", include_examples: bool = True
) -> str:
    """Create a prompt with specific response format instructions."""
    format_instructions = {
        "json": """
Please provide your response in the following JSON format:
{
    "feedback": "Main feedback about the text",
    "suggestions": ["Suggestion 1", "Suggestion 2", ...],
    "needs_improvement": true/false,
    "confidence": 0.0-1.0,  // YOUR self-assessed confidence in this critique (consider clarity of issues, domain expertise, and certainty of suggestions)
    "metadata": {}
}
""",
        "structured": """
Please structure your response as follows:
FEEDBACK: [Your main feedback about the text]
SUGGESTIONS:
1. [First suggestion]
2. [Second suggestion]
...
""",
        "text": "",
    }

    prompt = base_prompt
    if response_format in format_instructions:
        prompt += "\n\n" + format_instructions[response_format]

    if include_examples and response_format == "json":
        prompt += """
Example response:
{
    "feedback": "The text provides a good overview but lacks specific examples",
    "suggestions": ["Add concrete examples", "Include data to support claims"],
    "needs_improvement": true,
    "confidence": 0.85
}
"""

    return prompt
