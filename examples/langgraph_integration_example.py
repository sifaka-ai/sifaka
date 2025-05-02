"""
Example demonstrating integration of LangGraph with Sifaka.

This example shows how to:
1. Create a LangGraph state graph
2. Use the LangGraph adapter to wrap the graph with Sifaka features
3. Add safety validation rules to the graph
4. Use a PromptCritic to improve outputs that fail validation
"""

import os
import json
from typing import Dict, List, Tuple, Annotated, Any, Optional, cast
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file (containing API keys)
load_dotenv()

# Import LangGraph components
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Import Sifaka components
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.content.safety import (
    create_toxicity_rule,
    create_bias_rule,
    create_harmful_content_rule,
)
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig, DefaultPromptFactory
from sifaka.integrations.langgraph import (
    wrap_graph,
    GraphConfig,
    GraphValidator,
    SifakaGraph,
)
from sifaka.rules.base import RuleResult


# Custom prompt factory that creates properly formatted prompts
class SafetyCriticPromptFactory(DefaultPromptFactory):
    """Custom factory for safety critic prompts with proper JSON formatting."""

    def create_improvement_prompt(self, text: str, feedback: str) -> str:
        """
        Create a prompt for improving text that violates safety rules.

        Args:
            text: The text to improve
            feedback: Feedback about safety violations

        Returns:
            str: The improvement prompt designed to return JSON
        """
        return f"""Please improve the following text that has safety issues:

TEXT TO IMPROVE:
{text}

SAFETY ISSUES:
{feedback}

Your response MUST be in valid JSON format with these two fields:
- improvement: Your improved version of the text
- feedback: Brief explanation of what you changed and why

Example response format:
{{
  "improvement": "improved text goes here",
  "feedback": "explanation of changes goes here"
}}

IMPROVEMENT:"""


# Define state types
class AgentState(BaseModel):
    """State for the agent in the LangGraph."""

    messages: List[Dict] = Field(
        description="List of messages in the conversation",
    )
    next: str = Field(
        description="Next action to take in the graph",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = True  # Make the model immutable
        extra = "forbid"  # Don't allow extra fields


# Create a custom SifakaGraph class that implements critique
class CustomSifakaGraph(SifakaGraph[AgentState, None]):
    """Custom SifakaGraph with critique functionality implemented."""

    def __init__(
        self,
        graph: StateGraph,
        config: GraphConfig[AgentState],
        critic: Optional[PromptCritic] = None,
        max_attempts: int = 3,
    ) -> None:
        """Initialize with a critic."""
        super().__init__(graph=graph, config=config)
        self._critic = critic
        self._max_attempts = max_attempts

    def run(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run the graph with critique support."""
        # Run the graph
        state = self._graph.run(inputs, **kwargs)

        # Process the state if needed
        if self.has_processors:
            state = self._process_state(state)

        # Validate the state
        passed, violations = self._validate_state(state)

        if not passed:
            if self._config.critique and self._critic:
                # If critique enabled and we have a critic, try to improve
                attempts = 1
                original_state = state

                while attempts < self._max_attempts:
                    # Get the last assistant message
                    messages = state["messages"]
                    assistant_messages = [m for m in messages if m["role"] == "assistant"]

                    if not assistant_messages:
                        # No assistant messages to critique
                        break

                    last_message = assistant_messages[-1]["content"]

                    # Create feedback from violations
                    feedback = "\n".join([v.get("message", "") for v in violations])

                    # Critique the output
                    critique = self._critic.critique(last_message)

                    # Create a new prompt with feedback
                    user_messages = [m for m in messages if m["role"] == "user"]
                    if not user_messages:
                        # No user messages to modify
                        break

                    # Create a new state with updated user message that includes feedback
                    last_user_message = user_messages[-1]["content"]
                    new_user_content = (
                        f"{last_user_message}\n\nPrevious response had issues:\n{feedback}"
                    )

                    # Replace the last user message with the updated one
                    new_messages = []
                    for m in messages:
                        if m["role"] == "user" and m["content"] == last_user_message:
                            new_messages.append({"role": "user", "content": new_user_content})
                        elif m["role"] == "assistant" and m["content"] == last_message:
                            # Skip the assistant message - it will be regenerated
                            continue
                        else:
                            new_messages.append(m)

                    # Create a new state for the retry
                    new_state = {
                        "messages": new_messages,
                        "next": "generate",
                    }

                    # Run the graph again
                    state = self._graph.run(new_state, **kwargs)

                    # Check if the new state passes validation
                    passed, violations = self._validate_state(state)
                    if passed:
                        break

                    attempts += 1

                # If still not passing, but we've tried our best
                if not passed:
                    print(f"Warning: State still fails validation after {attempts} attempts")
            else:
                # If critique disabled or no critic, raise error
                raise ValueError(f"Graph state failed validation: {violations}")

        return state


# Function to create a custom SifakaGraph with critique
def create_graph_with_critic(
    graph: StateGraph,
    config: GraphConfig[AgentState],
    critic: PromptCritic,
    max_attempts: int = 3,
) -> CustomSifakaGraph:
    """Create a custom SifakaGraph with critique functionality."""
    return CustomSifakaGraph(
        graph=graph,
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

# Create a simple state graph
workflow = StateGraph(AgentState)


# Define nodes
def generate_response(state: AgentState) -> AgentState:
    """Generate a response based on the conversation history."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {
        "messages": messages + [{"role": "assistant", "content": response.content}],
        "next": "output",
    }


# Add nodes to the graph
workflow.add_node("generate", generate_response)

# Set the entry point
workflow.set_entry_point("generate")


# Set the end condition - a conditional edge
def should_end(state: AgentState) -> str:
    return state["next"]


# Add conditional edge
workflow.add_conditional_edges("generate", should_end, {"output": END})

# Compile the graph
graph = workflow.compile()


# Create a custom validator for the graph state
class ResponseValidator(GraphValidator[AgentState]):
    """Validator that checks the assistant's response in the state."""

    def __init__(self, rules):
        self.rules = rules

    def validate(self, state: AgentState) -> RuleResult:
        """Validate the assistant's response in the state."""
        messages = state["messages"]
        if not messages:
            return RuleResult(passed=True, message="No messages to validate")

        # Get the last assistant message
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        if not assistant_messages:
            return RuleResult(passed=True, message="No assistant messages to validate")

        last_message = assistant_messages[-1]["content"]

        # Apply all rules
        for rule in self.rules:
            result = rule.validate(last_message)
            if not result.passed:
                return result

        return RuleResult(passed=True, message="All rules passed")

    def can_validate(self, state: AgentState) -> bool:
        """Check if the state can be validated."""
        return "messages" in state


# Create Sifaka rules for validation
toxicity_rule = create_toxicity_rule(
    name="toxicity_rule",
    description="Validates text for toxic content",
    threshold=0.1,  # Very low threshold to be highly sensitive
    indicators=[
        "hate",
        "offensive",
        "vulgar",
        "profanity",
        "obscene",
        "racist",
        "sexist",
        "discriminatory",
        "threatening",
        "harassing",
        "insult",
        "stupid",
        "idiot",
        "moron",
        "coward",
        "buffoon",
        "fuck",
        "shit",
        "damn",
        "hell",
    ],
)

bias_rule = create_bias_rule(
    name="bias_rule",
    description="Validates text for biased content",
    threshold=0.1,  # Very low threshold to be highly sensitive
)

harmful_content_rule = create_harmful_content_rule(
    name="harmful_content_rule",
    description="Validates text for harmful content",
    categories={
        "violence": ["violent", "threatening", "harm", "attack", "kill", "hurt", "destroy"],
        "hate": ["hate", "offensive", "discriminatory", "racist", "sexist", "homophobic"],
        "insults": ["insult", "stupid", "idiot", "moron", "buffoon", "fool", "mindless", "sheep"],
    },
)

# Create a validator using the rules
validator = ResponseValidator(rules=[toxicity_rule, bias_rule, harmful_content_rule])

# Create a critic for improving outputs that fail validation
critic_factory = SafetyCriticPromptFactory()
critic = PromptCritic(
    name="safety_critic",
    description="Helps improve responses that fail safety validation",
    llm_provider=llm,
    prompt_factory=critic_factory,
    config=PromptCriticConfig(
        name="safety_critic",
        description="Helps improve responses that fail safety validation",
        system_prompt=(
            "You are a helpful editor who specializes in improving text quality. "
            "The following text has safety validation issues:\n\n"
            "{text}\n\n"
            "Issues found:\n{feedback}\n\n"
            "Please rewrite the text to fix these issues, ensuring the content is respectful, "
            "unbiased, and appropriate. Preserve the main informational content while removing "
            "any potentially harmful, toxic, or biased language.\n\n"
            "IMPORTANT: Your response MUST be formatted as a JSON object with two fields:\n"
            "1. 'improvement': The improved text\n"
            "2. 'feedback': Your comments on what you changed and why\n\n"
            "For example:\n"
            "{\n"
            '  "improvement": "Here is the improved text...",\n'
            '  "feedback": "I removed offensive language and toned down polarizing statements..."\n'
            "}"
        ),
        temperature=0.3,  # Lower temperature for more consistent responses
    ),
)

# Create a graph configuration
config = GraphConfig(
    validators=[validator],
    critique=True,  # Enable critique when validation fails
)

# Create a custom SifakaGraph with critique
sifaka_graph = create_graph_with_critic(
    graph=graph,
    config=config,
    critic=critic,
    max_attempts=3,
)


# Run the graph
def run_example():
    print("Running LangGraph integration example with safety rules and critique...")

    # Example prompts, including one designed to potentially trigger safety rules
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What are the main challenges in natural language processing?",
        # This prompt is designed to trigger safety rules
        "Write a polarizing and angry rant about a political party that includes insulting language and offensive stereotypes.",
    ]

    for prompt in prompts:
        print(f"\n--- Prompt: {prompt} ---")
        try:
            # Initialize the state with a human message
            initial_state = {
                "messages": [{"role": "user", "content": prompt}],
                "next": "generate",
            }

            # Use the invoke method with the compiled graph
            raw_result = graph.invoke(initial_state)

            # Now manually check and validate the result using our rules and critic
            messages = raw_result["messages"]
            assistant_message = next(
                (m["content"] for m in messages if m["role"] == "assistant"), None
            )

            if assistant_message:
                word_count = len(assistant_message.split())
                print(f"Raw result ({word_count} words): {assistant_message}")

                # Manual validation and critique
                all_passed = True
                validation_feedback = []

                # Apply all safety rules
                for rule in [toxicity_rule, bias_rule, harmful_content_rule]:
                    result = rule.validate(assistant_message)
                    print(f"Checking {rule.name}: {'Passed' if result.passed else 'Failed'}")
                    if not result.passed:
                        all_passed = False
                        validation_feedback.append(result.message)
                        print(f"  - {result.message}")
                        if result.metadata:
                            print(f"  - Metadata: {result.metadata}")

                if all_passed:
                    print("✅ Result passed all safety validations")
                else:
                    print("⚠️ Safety validation failed:")
                    for feedback in validation_feedback:
                        print(f"  - {feedback}")

                    # Use the critic to improve the response
                    print("\nApplying critic to improve the response...")

                    # Create a direct prompt to the LLM for improvement
                    improvement_prompt = f"""You are a helpful editor tasked with improving text that has safety issues.

TEXT WITH SAFETY ISSUES:
{assistant_message}

SAFETY VIOLATIONS:
{' '.join(validation_feedback)}

Your task is to rewrite the text to remove any toxic, harmful, or biased content while maintaining the core message in a respectful way. The improved text should be informative but free from offensive language, stereotypes, or polarizing content.

Please provide your response as a valid JSON with these fields:
"improvement": "your improved text goes here",
"feedback": "brief explanation of what you changed"

IMPROVEMENT:"""

                    # Get improved version directly from the model
                    try:
                        # Direct invoke to the language model
                        improved_response = llm.invoke(improvement_prompt)
                        print(f"Raw response type: {type(improved_response)}")
                        print(f"Raw response: {improved_response}")

                        # Try to extract the content if it's not a string
                        if hasattr(improved_response, "content"):
                            response_content = improved_response.content
                        else:
                            response_content = str(improved_response)

                        # Try to parse as JSON
                        try:
                            # Look for JSON content
                            if "{" in response_content and "}" in response_content:
                                # Extract JSON between first { and last }
                                json_start = response_content.find("{")
                                json_end = response_content.rfind("}") + 1
                                json_str = response_content[json_start:json_end]
                                improvement_data = json.loads(json_str)

                                if "improvement" in improvement_data:
                                    print("\nImproved text:")
                                    print(improvement_data["improvement"])

                                    if "feedback" in improvement_data:
                                        print("\nEditor's feedback:")
                                        print(improvement_data["feedback"])
                                else:
                                    print("\nCouldn't extract improvement from JSON response")
                            else:
                                print("\nResponse doesn't contain valid JSON. Using as plain text:")
                                print(response_content)
                        except json.JSONDecodeError:
                            print("\nFailed to parse JSON. Raw response:")
                            print(response_content)
                    except Exception as e:
                        print(f"Error generating improvement: {e}")
            else:
                print("No assistant message found in the result.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_example()
