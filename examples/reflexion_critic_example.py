"""
Reflexion Critic Example

This example demonstrates how to use the ReflexionCritic to improve text
by maintaining a memory of past improvements and using these reflections
to guide future improvements.
"""

import os
import logging

from sifaka.critics.reflexion import ReflexionCritic, ReflexionCriticConfig
from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.formatting.length import create_length_rule
from sifaka.chain import Chain

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("reflexion_example")

# Set the OpenAI API key explicitly
# Replace this with your actual API key or set the OPENAI_API_KEY environment variable
api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

# Create OpenAI provider
openai_model = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    config=ModelConfig(api_key=api_key, temperature=0.7, max_tokens=2048),
)


# Create a reflexion critic with a custom method to make it compatible with Chain
class CompatibleReflexionCritic(ReflexionCritic):
    """A ReflexionCritic that's compatible with the Chain class."""

    def critique(self, text: str) -> dict:
        """Analyze text and provide detailed feedback as a dict instead of CriticMetadata."""
        logger.info("Critiquing text...")
        critic_metadata = super().critique(text)
        # Convert CriticMetadata to a plain dictionary
        result = {
            "score": critic_metadata.score,
            "feedback": critic_metadata.feedback,
            "issues": critic_metadata.issues,
            "suggestions": critic_metadata.suggestions,
        }
        logger.info(f"Critique result: {result}")
        return result

    def improve(self, text: str, feedback: str = None) -> str:
        """Override improve to add logging."""
        logger.info(f"Improving text with feedback: {feedback}")
        if feedback is None:
            feedback = "Please improve this text."

        # Log current memories if any
        reflections = self._get_relevant_reflections()
        if reflections:
            logger.info(f"Using {len(reflections)} reflections from memory:")
            for i, reflection in enumerate(reflections):
                logger.info(f"  Reflection {i+1}: {reflection}")
        else:
            logger.info("No reflections in memory yet.")

        # Call parent method
        improved = super().improve(text, feedback)

        # Generate reflection (this happens in the parent improve method but we'll log it)
        logger.info(f"Generating reflection based on improvement...")
        self._generate_reflection(text, feedback, improved)

        # Log updated memories
        if self._memory_buffer:
            logger.info(f"Updated memory buffer now has {len(self._memory_buffer)} reflections")
            for i, reflection in enumerate(self._memory_buffer):
                logger.info(f"  Reflection {i+1}: {reflection}")

        return improved

    def _generate_reflection(self, original_text: str, feedback: str, improved_text: str) -> None:
        """Override to log the reflection generation process."""
        logger.info("Generating reflection...")
        reflection_prompt = self._prompt_factory.create_reflection_prompt(
            original_text, feedback, improved_text
        )
        logger.info(f"Reflection prompt: {reflection_prompt}")

        try:
            response = self._model.invoke(reflection_prompt)
            logger.info(f"Reflection response: {response}")

            # Extract reflection from response
            reflection = ""
            if isinstance(response, dict) and "reflection" in response:
                reflection = response["reflection"]
            elif isinstance(response, str):
                if "REFLECTION:" in response:
                    parts = response.split("REFLECTION:")
                    if len(parts) > 1:
                        reflection = parts[1].strip()
                else:
                    reflection = response.strip()

            # Add reflection to memory buffer
            if reflection:
                logger.info(f"Adding reflection to memory: {reflection}")
                self._add_to_memory(reflection)
            else:
                logger.info("No reflection extracted from response.")
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")

    def manually_create_reflection(self, prompt, issue, suggestion):
        """Manually create and add a reflection to demonstrate the process."""
        logger.info(f"Manually creating a reflection for prompt: {prompt}")

        # Generate a response
        original_text = self.llm_provider.generate(prompt)
        feedback = f"Issue: {issue}. Suggestion: {suggestion}"

        # Create a short/condensed version of the text
        short_prompt = (
            f"Create a very concise version (under 50 words) of this text: {original_text}"
        )
        improved_text = self.llm_provider.generate(short_prompt)

        logger.info(f"Original text: {original_text[:100]}... (truncated)")
        logger.info(f"Feedback: {feedback}")
        logger.info(f"Improved text: {improved_text}")

        # Generate reflection
        self._generate_reflection(original_text, feedback, improved_text)

        return original_text, improved_text


# Create a reflexion critic
reflexion_critic = CompatibleReflexionCritic(
    llm_provider=openai_model,
    config=ReflexionCriticConfig(
        name="length_reflexion_critic",
        description="A critic that helps adjust text length while learning from past attempts",
        system_prompt=(
            "You are an expert editor who specializes in adjusting text length. "
            "You maintain a memory of past improvements and use these reflections "
            "to guide future improvements. Focus on learning patterns from past "
            "feedback and applying them to new situations."
        ),
        memory_buffer_size=5,  # Store up to 5 reflections
        reflection_depth=1,  # Perform 1 level of reflection
    ),
)

print("\n\n=== GENERATING MANUAL REFLECTIONS ===\n")
print("First, we'll manually create reflections to populate the memory buffer\n")

# Manually create reflections for the first two prompts
original1, improved1 = reflexion_critic.manually_create_reflection(
    "Explain the concept of machine learning in detail.",
    "Text is too long for the required format",
    "Create a more concise explanation focusing on the key points only",
)

original2, improved2 = reflexion_critic.manually_create_reflection(
    "Describe the process of photosynthesis in plants.",
    "Text exceeds the required word count",
    "Condense the explanation to focus on the main steps only",
)

# Create two length rules - one very restrictive that most responses will fail
# This will force the reflexion critic to improve the text
short_rule = create_length_rule(
    min_words=10,
    max_words=50,
    rule_id="short_text_rule",
    description="Ensures text is between 10-50 words (very short)",
)

# Regular length rule
regular_rule = create_length_rule(
    min_words=100,
    max_words=500,
    rule_id="regular_length_rule",
    description="Ensures text is between 100-500 words",
)

# Create two chains - first with the short rule to force iterations and generate reflections
short_chain = Chain(
    model=openai_model,
    rules=[short_rule],
    critic=reflexion_critic,
    max_attempts=3,
)

# And another with the regular rule
regular_chain = Chain(
    model=openai_model,
    rules=[regular_rule],
    critic=reflexion_critic,
    max_attempts=3,
)

# Define a series of prompts that will benefit from reflexion
prompts = [
    "Explain how the internet works.",
    "Describe the water cycle.",
    "Explain the theory of relativity.",
]

print("\n\n=== TESTING WITH REGULAR LENGTH RULE (100-500 words) ===\n")
print("This should use the reflections we manually generated\n")

# Process all prompts with regular rule to use reflections
for i, prompt in enumerate(prompts):
    print(f"\n\n=== Processing Prompt {i+1}: {prompt} ===\n")

    # First show the memory buffer being used
    print("Current Reflexion Critic Memory Buffer:")
    for j, reflection in enumerate(reflexion_critic._memory_buffer):
        print(f"Reflection {j+1}: {reflection[:100]}... (truncated)")
    print()

    try:
        result = regular_chain.run(prompt)
        print(f"Output (Word Count: {len(result.output.split())}): {result.output}\n")
        print(f"Rule Results: {[r.passed for r in result.rule_results]}")

        if result.critique_details:
            print("\nCritique Details:")
            for key, value in result.critique_details.items():
                if key == "issues" or key == "suggestions":
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")

    except ValueError as e:
        print(f"Error: {e}")

# Print the final reflexion critic's memory buffer to see what it has learned
print("\n\n=== Final Reflexion Critic Memory Buffer ===\n")
for i, reflection in enumerate(reflexion_critic._memory_buffer):
    print(f"Reflection {i+1}:\n{reflection}\n")

print("\nNotice how the critic uses these reflections to guide text generation and improvement.")
