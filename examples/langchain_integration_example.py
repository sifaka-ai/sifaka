"""
Example demonstrating integration of LangChain with Sifaka.

This example shows how to:
1. Create a LangChain chain
2. Use the LangChain adapter to wrap the chain with Sifaka features
3. Add validation rules to the chain
4. Use a PromptCritic to improve outputs that fail validation
"""

import os
from typing import Any, Dict, List, Optional, Union, cast
from dotenv import load_dotenv

# Load environment variables from .env file (containing API keys)
load_dotenv()

# Import LangChain components
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Import Sifaka components
from sifaka.models.base import ModelConfig
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.integrations.langchain import (
    wrap_chain,
    ChainConfig,
    RuleBasedValidator,
    SifakaChain,
    ChainType,
    OutputType,
)


# Create a custom SifakaChain that implements critique
class CustomSifakaChain(SifakaChain[OutputType]):
    """Custom SifakaChain that implements critique functionality."""

    def __init__(
        self,
        chain: ChainType,
        config: ChainConfig[OutputType],
        critic: Optional[PromptCritic] = None,
        max_attempts: int = 3,
    ) -> None:
        """Initialize with a critic."""
        super().__init__(chain=chain, config=config)
        self._critic = critic
        self._max_attempts = max_attempts

    def run(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> OutputType:
        """Run with critique support."""
        # Run the chain
        if isinstance(self._chain, LLMChain):
            output = self._chain.run(inputs, **kwargs)
        else:
            # For RunnableSequence
            if isinstance(inputs, str):
                inputs = {"human_input": inputs}
            output = self._chain.invoke(inputs)
            if isinstance(output, dict):
                output = output.get("text", output.get("output", output))

        # Parse the output if configured
        if self.has_output_parser:
            output = cast(OutputType, self._config.output_parser.parse(output))

        # Process the output
        if self.has_processors:
            output = self._process_output(output)

        # Validate the output
        passed, violations = self._validate_output(output)
        if not passed:
            if self._config.critique and self._critic:
                # If critique enabled and we have a critic, try to improve
                attempts = 1
                original_inputs = inputs

                while attempts < self._max_attempts:
                    # Create feedback from violations
                    feedback = "\n".join([v.get("message", "") for v in violations])

                    # Critique the output
                    critique = self._critic.critique(str(output))

                    # Try again with LLM using the critic feedback
                    if isinstance(self._chain, LLMChain):
                        # For LLMChain, append feedback to prompt
                        if isinstance(original_inputs, dict):
                            # Add feedback for specific key if it exists
                            topic = original_inputs.get("topic", "")
                            input_with_feedback = {
                                "topic": f"{topic}\n\nPrevious output had issues:\n{feedback}",
                                **{k: v for k, v in original_inputs.items() if k != "topic"},
                            }
                        else:
                            # For string input
                            input_with_feedback = (
                                f"{original_inputs}\n\nPrevious output had issues:\n{feedback}"
                            )

                        # Generate new output
                        output = self._chain.run(input_with_feedback, **kwargs)
                    else:
                        # For RunnableSequence, try to improve with the critique
                        # Note: This is simplified and might need adjustments for specific Runnable implementations
                        inputs_copy = dict(original_inputs)
                        inputs_copy["feedback"] = feedback
                        output = self._chain.invoke(inputs_copy)
                        if isinstance(output, dict):
                            output = output.get("text", output.get("output", output))

                    # Validate again
                    passed, violations = self._validate_output(output)
                    if passed:
                        break

                    attempts += 1

                # If still not passing, but we've tried our best
                if not passed:
                    print(f"Warning: Output still fails validation after {attempts} attempts")
            else:
                # If critique disabled or no critic, raise error
                raise ValueError(f"Output validation failed: {violations}")

        return output


# Function to create a custom SifakaChain with critique
def create_chain_with_critic(
    chain: ChainType,
    config: ChainConfig,
    critic: PromptCritic,
    max_attempts: int = 3,
) -> CustomSifakaChain:
    """Create a custom SifakaChain with critique functionality."""
    return CustomSifakaChain(
        chain=chain,
        config=config,
        critic=critic,
        max_attempts=max_attempts,
    )


# Configure OpenAI model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Create a prompt template for a longer response to demonstrate critique
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a paragraph about {topic}. Be informative and thorough.",
)

# Create a LangChain chain
chain = LLMChain(llm=llm, prompt=prompt)

# Create Sifaka rules for validation
length_rule = create_length_rule(max_words=150)

# Create a validator using the rules
validator = RuleBasedValidator(rules=[length_rule])

# Create a critic for improving outputs that fail validation
critic = PromptCritic(
    name="length_critic",
    description="Helps adjust text length to meet requirements",
    llm_provider=llm,
    config=PromptCriticConfig(
        name="length_critic",
        description="Helps adjust text length to meet requirements",
        system_prompt=(
            "You are a strict editor who specializes in making text concise. "
            "The following text is too long and must be shortened.\n\n"
            "{text}\n\n"
            "Issues found:\n{feedback}\n\n"
            "Please rewrite the text to be more concise, using FEWER THAN 150 WORDS TOTAL. "
            "Focus on the most essential information. Be ruthless in cutting unnecessary content. "
            "Do not exceed the 150-word limit under any circumstances."
        ),
        temperature=0.3,  # Lower temperature for more precise following of instructions
    ),
)

# Create a chain configuration
config = ChainConfig(
    validators=[validator],
    processors=[],  # Optional: Add output processors if needed
    callbacks=[],  # Optional: Add callback handlers if needed
    critique=True,  # Enable critique
)

# Create a custom SifakaChain with critique
sifaka_chain = create_chain_with_critic(
    chain=chain,
    config=config,
    critic=critic,
    max_attempts=3,
)


# Run the chain
def run_example():
    print("Running LangChain integration example with critique...")

    # Example topics
    topics = [
        "artificial intelligence",
        "climate change",
        "space exploration",
    ]

    for topic in topics:
        print(f"\n--- Topic: {topic} ---")
        try:
            result = sifaka_chain.run(inputs={"topic": topic})
            word_count = len(str(result).split())
            print(f"Result ({word_count} words): {result}")
            if word_count > 150:
                print("⚠️ Note: Even with critique, response exceeds 150-word limit")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_example()
