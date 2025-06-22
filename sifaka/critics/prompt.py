"""Prompt-based critic implementation.

A configurable critic that allows users to define custom evaluation
criteria through prompts for domain-specific text assessment.
"""

from typing import List, Optional, Union
import openai

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .base import BaseCritic, CriticConfig, create_prompt_with_format


class PromptCritic(BaseCritic):
    """Configurable prompt-based critic for custom evaluation criteria."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        custom_prompt: Optional[str] = None,
        criteria: Optional[List[str]] = None,
        name_suffix: str = "",
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[CriticConfig] = None,
    ):
        # Initialize with custom config for prompt critic
        if config is None:
            # Adjust base confidence based on criteria specificity
            base_conf = 0.7 if custom_prompt else (0.6 if criteria else 0.5)
            config = CriticConfig(
                response_format="json",
                base_confidence=base_conf,
                context_weight=0.05,
                depth_weight=0.1,
                domain_weight=0.15 if name_suffix else 0.05,
            )
        super().__init__(model, temperature, config, provider=provider, api_key=api_key)
        self.custom_prompt = custom_prompt
        self.criteria = criteria or []
        self.name_suffix = name_suffix

    @property
    def name(self) -> str:
        base_name = "prompt"
        return f"{base_name}_{self.name_suffix}" if self.name_suffix else base_name

    async def _generate_critique(self, text: str, result: SifakaResult) -> str:
        """Generate critique using custom prompt or criteria."""
        if self.custom_prompt:
            base_prompt = f"""{self.custom_prompt}

Text to evaluate:
{text}

Please provide your assessment following the specified criteria."""
        else:
            base_prompt = self._build_criteria_prompt(text)

        # Add specific instructions for JSON format if needed
        if self.config.response_format == "json":
            prompt = (
                base_prompt
                + """

Provide your response in this JSON format:
{
    "feedback": "Your detailed evaluation",
    "suggestions": ["Specific improvement 1", "Specific improvement 2", ...],
    "needs_improvement": true/false,
    "confidence": 0.0-1.0,
    "metadata": {
        "criteria_met": true/false,
        "criteria_scores": {"criterion1": "pass/fail", ...}
    }
}"""
            )
        else:
            prompt = create_prompt_with_format(base_prompt, self.config.response_format)

        response = await self.client.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a text critic applying custom evaluation criteria.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        return response.content

    def _build_criteria_prompt(self, text: str) -> str:
        """Build a prompt from the provided criteria."""
        if not self.criteria:
            # Default generic evaluation
            criteria_text = """1. Clarity and readability
2. Accuracy and factual correctness
3. Completeness and thoroughness
4. Structure and organization
5. Appropriateness for intended audience"""
        else:
            criteria_text = "\n".join(
                f"{i+1}. {criterion}" for i, criterion in enumerate(self.criteria)
            )

        return f"""Evaluate this text against the following custom criteria:

EVALUATION CRITERIA:
{criteria_text}

Text to evaluate:
{text}

For each criterion, assess whether the text meets the requirement.
Provide specific feedback and improvement suggestions."""

    def _parse_json_response(self, response: str) -> "CriticResponse":
        """Parse JSON response with prompt-specific handling."""
        try:
            critic_response = super()._parse_json_response(response)

            # Extract criteria met from metadata
            criteria_met = critic_response.metadata.get("criteria_met", None)
            if criteria_met is not None and not criteria_met:
                # Override needs_improvement if criteria not met
                critic_response.needs_improvement = True

            # Enhance confidence based on domain specificity
            if self.name_suffix:
                domain_indicators = self._get_domain_indicators()
                if domain_indicators:
                    # Check if feedback contains domain-specific terms
                    feedback_lower = critic_response.feedback.lower()
                    domain_matches = sum(
                        1 for ind in domain_indicators if ind in feedback_lower
                    )
                    if domain_matches > 0:
                        critic_response.confidence *= 1.0 + min(
                            0.2, domain_matches * 0.05
                        )

            return critic_response

        except Exception:
            # Fallback to standard parsing
            return super()._parse_json_response(response)

    def _get_domain_indicators(self) -> List[str]:
        """Get domain-specific indicators based on name suffix."""
        domain_map = {
            "academic": [
                "thesis",
                "research",
                "methodology",
                "citation",
                "peer-review",
                "hypothesis",
            ],
            "business": [
                "roi",
                "strategy",
                "stakeholder",
                "revenue",
                "growth",
                "market",
                "professional",
            ],
            "technical": [
                "implementation",
                "architecture",
                "performance",
                "scalability",
                "algorithm",
            ],
            "creative": ["narrative", "character", "plot", "imagery", "style", "voice"],
        }

        return domain_map.get(self.name_suffix.lower(), [])


def create_academic_critic(
    model: str = "gpt-4o-mini", temperature: float = 0.5
) -> PromptCritic:
    """Factory function to create an academic-focused critic."""
    criteria = [
        "Clear thesis statement and research objectives",
        "Proper academic tone and formal language",
        "Logical argument structure and flow",
        "Adequate evidence and citations",
        "Critical analysis and original insights",
        "Proper academic formatting and conventions",
    ]

    return PromptCritic(
        model=model, temperature=temperature, criteria=criteria, name_suffix="academic"
    )
