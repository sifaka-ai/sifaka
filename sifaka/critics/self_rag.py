"""Self-RAG critic implementation.

Based on: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
Paper: https://arxiv.org/abs/2310.11511
Authors: Asai et al. (2023)

Self-RAG teaches LLMs to retrieve, generate, and critique text passages
to improve factual accuracy and quality.

## Similarity to Original Paper:
- PRESERVED: Self-critique of factual accuracy
- PRESERVED: Focus on verifiability and relevance
- SIMPLIFIED: No actual retrieval system integration
- ADAPTED: General critique vs retrieval-specific

## Implementation Choices:
1. Focuses on factual claims and verifiability
2. Evaluates need for supporting evidence
3. Critiques relevance and accuracy
4. Suggests where retrieval would help

## Why This Approach:
- Factual accuracy is crucial for quality text
- Self-critique improves reliability
- Works without external retrieval system
- Guides users on information gaps
"""

from typing import Optional, Union, List, Dict

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


class SelfRAGCritic(BaseCritic):
    """Implements Self-RAG critique for factual accuracy."""

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

    async def _create_messages(self, text: str, result: SifakaResult) -> List[Dict[str, str]]:
        """Create messages for RAG-style critique."""
        # Get previous context
        previous_context = self._get_previous_context(result)
        
        user_prompt = f"""You are a Self-RAG critic focused on factual accuracy and verifiability.

Text to evaluate:
{text}
{previous_context}

Analyze the text for:

1. **Factual Claims**: Identify all factual assertions
2. **Verifiability**: Which claims need supporting evidence?
3. **Accuracy Concerns**: Any potentially inaccurate statements?
4. **Information Gaps**: Where would retrieval/citations help?
5. **Relevance**: Is all information relevant to the topic?

For each issue found:
- Explain why it's problematic
- Suggest how to improve accuracy
- Identify where external sources would help

Focus on improving the text's reliability and factual grounding."""

        return [
            {
                "role": "system",
                "content": "You are a Self-RAG critic that evaluates text for factual accuracy, verifiability, and identifies where retrieval of external information would improve quality."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]