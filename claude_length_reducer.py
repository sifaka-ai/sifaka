"""
Example demonstrating usage of a Length Critic to reduce response length.

This example shows how to:
1. Configure a mock model provider
2. Set up a Length Rule to constrain response length
3. Create a Chain with the model and rule
4. Use a PromptCritic for guidance when length constraints aren't met
"""

import os
from typing import Any, Dict

from sifaka.models.mock import create_mock_provider
from sifaka.models.base import ModelConfig
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.chain import create_simple_chain


# Create a custom model with extended generate method that produces longer text
class VerboseMockProvider:
    """A wrapper around MockProvider that produces longer responses for testing."""

    def __init__(self):
        """Initialize with a mock provider from the sifaka library."""
        self.mock_model = create_mock_provider(
            model_name="verbose-mock",
            temperature=0.7,
            max_tokens=1500,
        )
        self.model_name = "verbose-mock"

    def generate(self, prompt: str) -> str:
        """Generate verbose responses that will exceed our word limit rule."""
        print(f"Generating response for prompt: {prompt[:50]}...")
        if "language models" in prompt.lower():
            return """
            Large Language Models (LLMs) are advanced AI systems that understand and generate human-like text based on the patterns they've learned from massive datasets. These models are built on neural network architectures, primarily transformers, which allow them to process and generate text with remarkable coherence and contextual understanding.

            The architecture of LLMs is centered around the transformer design, introduced in the 2017 paper "Attention is All You Need." This architecture uses self-attention mechanisms to weigh the importance of different words in relation to each other, allowing the model to understand context over long passages of text. Unlike earlier sequential models like RNNs (Recurrent Neural Networks) or LSTMs (Long Short-Term Memory networks), transformers process all words in parallel, making them more efficient to train on large datasets.

            The training process for LLMs occurs in multiple stages. Initially, models undergo pre-training on diverse internet text, books, and other written materials. During pre-training, models learn general language patterns, grammar, facts, and reasoning abilities through tasks like predicting the next word in a sentence. This unsupervised learning phase requires enormous computational resources - training a large model can consume millions of dollars in computing power. After pre-training, models often undergo fine-tuning on more specific datasets aligned with human preferences and intended use cases.

            Modern LLMs have displayed impressive capabilities, including:
            • Writing coherent and contextually appropriate text across various topics and styles
            • Answering questions and explaining complex concepts
            • Summarizing long documents
            • Translating between languages
            • Writing creative content like stories, poems, and scripts
            • Generating code in multiple programming languages
            • Reasoning through multi-step problems

            However, these models also have significant limitations:
            • They can generate plausible-sounding but factually incorrect information ("hallucinations")
            • They lack true understanding of the content they produce, instead predicting what text should come next based on patterns
            • They don't have access to up-to-date information beyond their training data cutoff
            • They can reflect or amplify biases present in their training data
            • They don't have personal experiences or consciousness despite sometimes appearing to
            • They can't perform actions in the physical world or access external systems unless specifically connected
            • They may struggle with complex reasoning, mathematical calculations, or tasks requiring a deep causal understanding

            LLMs have evolved rapidly in recent years. GPT (Generative Pre-trained Transformer) models from OpenAI have grown from 117 million parameters in GPT-1 to 175 billion in GPT-3 and even larger with GPT-4. Anthropic's Claude models, Google's PaLM and Gemini, Meta's LLaMA, and many other models have emerged with diverse architectures and training approaches. Open-source models have democratized access to this technology.

            The development of LLMs raises important ethical questions about misinformation, consent for training data, automation of creative work, security risks, and potential misuse. Researchers are actively working on techniques for alignment (ensuring models act according to human values), interpretability (understanding model decision-making), and safety (preventing harmful outputs).

            Future directions for LLM research include multimodal capabilities (working with images, audio, and video alongside text), better reasoning, more efficient architectures, reduced hallucination, and improved factuality. The rapid pace of advancement in this field suggests that LLMs will continue to improve and find new applications across many domains.
            """
        else:
            return "This is a mock response that would normally be generated by an actual language model."

    def invoke(self, prompt: str) -> Dict[str, Any]:
        """Implement invoke method to support the critic interface."""
        print(f"Invoking model with prompt: {prompt[:50]}...")
        return {"text": self.generate(prompt)}


# Create our verbose model
print("Creating mock model...")
model = VerboseMockProvider()

# Create a length rule with constraints
print("Creating length rule...")
length_rule = create_length_rule(
    min_words=100,  # Setting a reasonable minimum
    max_words=500,  # Maximum that our verbose model will exceed
    rule_id="word_limit_rule",
)

# Create a critic to help improve responses that don't meet the length rule
print("Creating prompt critic...")
critic = PromptCritic(
    name="length_critic",
    description="Helps adjust text to meet length requirements",
    llm_provider=model.mock_model,  # Use the underlying mock model for the critic
    config=PromptCriticConfig(
        name="length_critic",
        description="Helps adjust text to meet length requirements",
        system_prompt=(
            "You are a skilled editor who specializes in adjusting text length while "
            "preserving the core content and meaning. Your job is to make text more "
            "concise by removing unnecessary details, redundancies, and filler content.\n\n"
            "IMPORTANT LENGTH REQUIREMENTS:\n"
            "- Minimum: 100 words\n"
            "- Maximum: 500 words\n"
            "- Target: Aim for 300-400 words\n\n"
            "The text must be within these bounds. Be ruthless in cutting content if over the limit, "
            "and add relevant details if under the minimum. Count your words carefully."
        ),
        temperature=0.4,  # Slightly lower temperature for more consistent results
    ),
)

# Create a chain with the model, rule, and critic
print("Creating chain...")
chain = create_simple_chain(model=model, rules=[length_rule], critic=critic, max_attempts=3)

# Prompt designed to generate a verbose response (around 500 words)
prompt = """
Please explain how large language models work, including details about their architecture,
training process, capabilities, and limitations. Include information about transformer
architecture, attention mechanisms, pre-training and fine-tuning, and examples of what
they can and cannot do well. Make your explanation comprehensive yet accessible.
"""

# Run the chain - it will generate text, check the length rule, and if needed,
# use the critic to improve the output by reducing the length
print("Running chain...")
try:
    print("Chain execution starting...")
    result = chain.run(prompt)
    print(f"✅ Final output ({len(result.output.split())} words):")
    print(result.output)

    if result.critique_details:
        print("\n🔍 Critique details:")
        for key, value in result.critique_details.items():
            print(f"  {key}: {value}")

except ValueError as e:
    print(f"❌ Error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback

    traceback.print_exc()
