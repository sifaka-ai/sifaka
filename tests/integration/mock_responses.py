"""Mock responses for integration tests in CI environments."""

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock


class MockLLMProvider:
    """Mock LLM provider for CI testing."""

    def __init__(self, provider_name: str = "mock"):
        self.provider_name = provider_name
        self.call_count = 0
        self.responses = self._get_responses()

    def _get_responses(self) -> Dict[str, List[str]]:
        """Get predefined responses for different critics."""
        return {
            "reflexion": [
                "The text provides a basic overview of renewable energy benefits. However, it could be improved by including specific statistics and examples.",
                "The revised text now includes concrete data about CO2 reduction and job creation. It could further benefit from addressing potential challenges.",
                "The text is now comprehensive, covering benefits, statistics, and challenges while maintaining clarity and engagement.",
            ],
            "constitutional": [
                "The text should be more balanced and include diverse perspectives on renewable energy adoption.",
                "The improved version now addresses economic, social, and environmental aspects equally.",
                "The text successfully presents a nuanced view while remaining informative and accessible.",
            ],
            "self_refine": [
                "Initial draft covers main points but lacks depth in explaining technological advancements.",
                "Expanded coverage of solar, wind, and emerging technologies provides better context.",
                "The refined version effectively balances technical detail with accessibility.",
            ],
            "critic": [
                "Claim verification needed: 'renewable energy creates jobs' requires supporting evidence.",
                "Evidence added: '10 million jobs globally in renewable sector (IRENA, 2023)'.",
                "All claims are now properly supported with credible sources.",
            ],
            "chain_of_thought": [
                "Step 1: Identify key benefits. Step 2: Organize logically. Step 3: Add transitions.",
                "Reorganized with clear sections: Environmental, Economic, Social benefits.",
                "Structure is now logical and easy to follow with smooth transitions.",
            ],
            "step_by_step": [
                "✓ Introduction clear. ✗ Missing statistics. ✗ No future outlook.",
                "✓ Statistics added. ✓ Future trends included. ✗ Could use more examples.",
                "✓ All criteria met with comprehensive coverage and specific examples.",
            ],
            "debate": [
                "Optimist: Focus on benefits. Skeptic: What about intermittency and costs?",
                "Balanced view incorporating storage solutions and declining costs.",
                "Consensus reached on realistic assessment of benefits and challenges.",
            ],
            "expert_iteration": [
                "Domain expert suggests adding grid integration and policy aspects.",
                "Technical details on smart grids and policy frameworks incorporated.",
                "Expert-level content while maintaining general accessibility.",
            ],
            "default": [
                "The text has been improved with better structure and clarity.",
                "Further refinements made to enhance readability and impact.",
                "Final version is polished and ready for publication.",
            ],
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        self.call_count += 1

        # Simulate processing time
        time.sleep(0.01)

        # Determine which critic type based on prompt content
        critic_type = self._detect_critic_type(prompt)
        responses = self.responses.get(critic_type, self.responses["default"])

        # Cycle through responses based on call count
        response_index = (self.call_count - 1) % len(responses)

        # For improvements, generate improved text
        if "improve" in prompt.lower() or "revise" in prompt.lower():
            return self._generate_improved_text(prompt, response_index)

        # For critiques, return critique text
        return responses[response_index]

    def _detect_critic_type(self, prompt: str) -> str:
        """Detect critic type from prompt."""
        prompt_lower = prompt.lower()

        critic_keywords = {
            "reflexion": ["reflect", "self-reflection", "evaluate your"],
            "constitutional": ["constitutional", "principles", "ethical"],
            "self_refine": ["refine", "iteratively improve"],
            "critic": ["verify", "fact-check", "evidence"],
            "chain_of_thought": ["step-by-step reasoning", "chain of thought"],
            "step_by_step": ["checklist", "criteria", "verify each"],
            "debate": ["debate", "perspectives", "optimist", "skeptic"],
            "expert_iteration": ["expert", "domain knowledge", "specialized"],
        }

        for critic_type, keywords in critic_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return critic_type

        return "default"

    def _generate_improved_text(self, prompt: str, iteration: int) -> str:
        """Generate improved text based on iteration."""
        base_text = (
            "Renewable energy offers significant benefits for our planet and society."
        )

        improvements = [
            f"{base_text} It reduces greenhouse gas emissions by up to 80% compared to fossil fuels, creates millions of jobs globally, and provides energy security for nations.",
            f"{base_text} Studies show renewable energy reduces CO2 emissions by 2.8 gigatons annually. The sector employs over 10 million people worldwide and costs have decreased by 85% for solar and 70% for wind since 2010. However, challenges include intermittency and grid integration requirements.",
            f"{base_text} Renewable energy is transforming our world through three key impacts: Environmental (2.8Gt CO2 reduction annually, preserving ecosystems), Economic (10M jobs, $1.3T annual investment, energy independence), and Social (energy access for 1B people, reduced air pollution saving 7M lives yearly). While intermittency and initial costs remain challenges, advancing storage technology and supportive policies are rapidly addressing these concerns, making renewables the fastest-growing energy source globally.",
        ]

        if iteration < len(improvements):
            return improvements[iteration]
        return improvements[-1]

    def count_tokens(self, text: str) -> int:
        """Mock token counting."""
        # Approximate: 1 token per 4 characters
        return len(text) // 4


class MockModelConfig:
    """Mock model configuration."""

    def __init__(self, model_name: str = "mock-model", **kwargs):
        self.model_name = model_name
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1000)
        self.api_key = "mock-api-key"
        self.base_url = kwargs.get("base_url", "https://api.mock.com")


def create_mock_llm(provider: str = "openai") -> MagicMock:
    """Create a mock LLM instance."""
    mock_provider = MockLLMProvider(provider)

    mock_llm = MagicMock()
    mock_llm.generate = mock_provider.generate
    mock_llm.count_tokens = mock_provider.count_tokens
    mock_llm.model_config = MockModelConfig(f"mock-{provider}-model")
    mock_llm.supports_async = False

    return mock_llm


def create_mock_improvement_result(
    original_text: str, iterations: int = 1, critic_type: str = "reflexion"
) -> Dict[str, Any]:
    """Create a mock improvement result."""
    mock_provider = MockLLMProvider()

    improvement_history = []
    current_text = original_text
    total_tokens = 0

    for i in range(iterations):
        critique = mock_provider.responses[critic_type][
            i % len(mock_provider.responses[critic_type])
        ]
        improved_text = mock_provider._generate_improved_text(current_text, i)
        tokens_used = mock_provider.count_tokens(critique) + mock_provider.count_tokens(
            improved_text
        )

        improvement_history.append(
            {
                "iteration": i + 1,
                "critique": critique,
                "improved_text": improved_text,
                "tokens_used": tokens_used,
                "latency_ms": 10.5 + (i * 2),
            }
        )

        current_text = improved_text
        total_tokens += tokens_used

    return {
        "original_text": original_text,
        "final_text": current_text,
        "iterations": iterations,
        "total_tokens": total_tokens,
        "improvement_history": improvement_history,
        "metadata": {
            "critic": critic_type,
            "model": "mock-model",
            "provider": "mock",
            "validation_attempts": 1,
            "total_latency_ms": sum(h["latency_ms"] for h in improvement_history),
        },
    }


# Predefined mock responses for common test scenarios
MOCK_RESPONSES = {
    "simple_improvement": {
        "final_text": "Renewable energy offers significant benefits including reduced emissions, job creation, and energy independence.",
        "iterations": 1,
        "total_tokens": 150,
    },
    "multi_iteration": {
        "final_text": "Renewable energy transforms our world through environmental, economic, and social benefits, while addressing implementation challenges.",
        "iterations": 3,
        "total_tokens": 450,
    },
    "validation_failure": {
        "final_text": "Benefits of renewable energy",  # Too short, will fail length validation
        "iterations": 1,
        "total_tokens": 100,
    },
    "timeout_scenario": {
        "final_text": "Renewable energy benefits include...",  # Incomplete due to timeout
        "iterations": 1,
        "total_tokens": 50,
        "error": "Operation timed out",
    },
}
