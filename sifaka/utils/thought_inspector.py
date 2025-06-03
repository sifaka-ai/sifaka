"""Thought inspection utilities for debugging and analysis.

This module provides utilities to inspect SifakaThought objects and extract
detailed information about model prompts, critic feedback, and validation results.
"""

from typing import List, Optional
from sifaka.core.thought import SifakaThought


def get_conversation_messages_for_iteration(thought: SifakaThought, iteration: int) -> List[str]:
    """Get all conversation messages (requests and responses) from a specific iteration.

    Args:
        thought: The SifakaThought object
        iteration: The iteration number to get messages for

    Returns:
        List of conversation messages from that iteration (includes both requests TO model and responses FROM model)
    """
    messages = []
    iteration_generations = [g for g in thought.generations if g.iteration == iteration]

    for generation in iteration_generations:
        if generation.conversation_history:
            for msg in generation.conversation_history:
                if isinstance(msg, dict):
                    # Extract content from structured message
                    content = msg.get("content", str(msg))
                    messages.append(content)
                else:
                    # Handle string message
                    messages.append(str(msg))

    return messages


def get_latest_conversation_messages(thought: SifakaThought) -> List[str]:
    """Get conversation messages from the most recent iteration.

    Args:
        thought: The SifakaThought object

    Returns:
        List of conversation messages from the current iteration (includes both requests and responses)
    """
    return get_conversation_messages_for_iteration(thought, thought.iteration)


# Backward compatibility functions (deprecated but maintained for existing code)
def get_model_prompts_for_iteration(thought: SifakaThought, iteration: int) -> List[str]:
    """DEPRECATED: Use get_conversation_messages_for_iteration() instead.

    This function name was misleading since it returns both requests TO the model
    and responses FROM the model, not just prompts.
    """
    return get_conversation_messages_for_iteration(thought, iteration)


def get_latest_model_prompts(thought: SifakaThought) -> List[str]:
    """DEPRECATED: Use get_latest_conversation_messages() instead.

    This function name was misleading since it returns both requests TO the model
    and responses FROM the model, not just prompts.
    """
    return get_latest_conversation_messages(thought)


def print_iteration_details(thought: SifakaThought, iteration: int = None) -> None:
    """Print detailed information for a specific iteration or current iteration.

    Args:
        thought: The SifakaThought object
        iteration: Iteration to print details for. If None, uses current iteration.
    """
    if iteration is None:
        iteration = thought.iteration

    print(f"🔄 ITERATION {iteration} DETAILS")
    print("-" * 50)

    # Show generations
    iteration_generations = [g for g in thought.generations if g.iteration == iteration]
    for gen in iteration_generations:
        print(f"🤖 Model: {gen.model}")
        print(f"📄 Generated text ({len(gen.text)} chars): {gen.text[:100]}...")

        # Show conversation messages (requests and responses)
        if gen.conversation_history:
            print("💬 Conversation messages:")
            for i, msg in enumerate(gen.conversation_history):
                content = msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg)
                truncated = content[:150] + "..." if len(content) > 150 else content
                print(f"   {i+1}. {truncated}")

    # Show validations
    iteration_validations = [v for v in thought.validations if v.iteration == iteration]
    if iteration_validations:
        print("\n✅ Validations:")
        for validation in iteration_validations:
            status = "✅ PASSED" if validation.passed else "❌ FAILED"
            print(f"   {validation.validator}: {status}")
            if not validation.passed and "error" in validation.details:
                print(f"      Error: {validation.details['error']}")

    # Show critiques
    iteration_critiques = [c for c in thought.critiques if c.iteration == iteration]
    if iteration_critiques:
        print("\n🔍 Critics:")
        for critique in iteration_critiques:
            print(f"   🎯 {critique.critic}:")
            print(f"      Needs Improvement: {critique.needs_improvement}")
            print(f"      Confidence: {critique.confidence}")
            print(f"      Feedback: {critique.feedback[:100]}...")
            if critique.suggestions:
                print(f"      Suggestions ({len(critique.suggestions)}):")
                for i, suggestion in enumerate(critique.suggestions[:2]):
                    print(f"         {i+1}. {suggestion[:80]}...")

    print()


def print_all_iterations(thought: SifakaThought) -> None:
    """Print details for all iterations in the thought.

    Args:
        thought: The SifakaThought object
    """
    print("🧠 COMPLETE THOUGHT ANALYSIS")
    print("=" * 60)

    for i in range(thought.iteration + 1):
        print_iteration_details(thought, iteration=i)


def print_critic_summary(thought: SifakaThought) -> None:
    """Print a summary of all critic feedback across iterations.

    Args:
        thought: The SifakaThought object
    """
    print("🔍 CRITIC FEEDBACK SUMMARY")
    print("=" * 50)

    if not thought.critiques:
        print("No critic feedback available.")
        return

    # Group by critic type
    critics_by_type = {}
    for critique in thought.critiques:
        if critique.critic not in critics_by_type:
            critics_by_type[critique.critic] = []
        critics_by_type[critique.critic].append(critique)

    for critic_name, critiques in critics_by_type.items():
        print(f"\n🎯 {critic_name} ({len(critiques)} evaluations):")

        for critique in critiques:
            print(f"   Iteration {critique.iteration}:")
            print(f"      Needs Improvement: {critique.needs_improvement}")
            print(f"      Confidence: {critique.confidence}")
            print(f"      Feedback: {critique.feedback[:100]}...")
            if critique.suggestions:
                print(f"      Suggestions: {len(critique.suggestions)} provided")


def print_validation_summary(thought: SifakaThought) -> None:
    """Print a summary of all validation results across iterations.

    Args:
        thought: The SifakaThought object
    """
    print("✅ VALIDATION SUMMARY")
    print("=" * 50)

    if not thought.validations:
        print("No validation results available.")
        return

    # Group by validator type
    validators_by_type = {}
    for validation in thought.validations:
        if validation.validator not in validators_by_type:
            validators_by_type[validation.validator] = []
        validators_by_type[validation.validator].append(validation)

    for validator_name, validations in validators_by_type.items():
        print(f"\n📋 {validator_name} ({len(validations)} checks):")

        for validation in validations:
            status = "✅ PASSED" if validation.passed else "❌ FAILED"
            print(f"   Iteration {validation.iteration}: {status}")
            if not validation.passed and "error" in validation.details:
                print(f"      Error: {validation.details['error']}")


def print_conversation_messages(
    thought: SifakaThought, iteration: int = None, full_messages: bool = False
) -> None:
    """Print conversation messages for a specific iteration or all iterations.

    Args:
        thought: The SifakaThought object
        iteration: Specific iteration to show messages for. If None, shows all iterations.
        full_messages: If True, shows full messages. If False, truncates long messages.
    """
    if iteration is not None:
        iterations_to_show = [iteration]
        print(f"💬 CONVERSATION MESSAGES - ITERATION {iteration}")
    else:
        iterations_to_show = list(range(thought.iteration + 1))
        print("💬 CONVERSATION MESSAGES - ALL ITERATIONS")

    print("=" * 60)

    for iter_num in iterations_to_show:
        if len(iterations_to_show) > 1:
            print(f"\n🔄 Iteration {iter_num}:")
            print("-" * 40)

        messages = get_conversation_messages_for_iteration(thought, iter_num)

        if not messages:
            print("   No conversation messages found for this iteration.")
            continue

        for i, message in enumerate(messages):
            print(f"\nMessage {i+1}:")
            if full_messages:
                print(message)
            else:
                # Truncate long messages for readability
                if len(message) > 300:
                    print(message[:300] + "\n... [truncated] ...")
                else:
                    print(message)
            print("-" * 30)


# Backward compatibility function (deprecated but maintained for existing code)
def print_model_prompts(
    thought: SifakaThought, iteration: int = None, full_prompts: bool = False
) -> None:
    """DEPRECATED: Use print_conversation_messages() instead.

    This function name was misleading since it prints both requests TO the model
    and responses FROM the model, not just prompts.
    """
    print_conversation_messages(thought, iteration, full_prompts)


def get_thought_overview(thought: SifakaThought) -> dict:
    """Get a comprehensive overview of the thought's state and history.

    Args:
        thought: The SifakaThought object

    Returns:
        Dictionary containing overview information
    """
    return {
        "id": thought.id,
        "prompt": thought.prompt,
        "current_iteration": thought.iteration,
        "max_iterations": thought.max_iterations,
        "final_text_length": len(thought.final_text) if thought.final_text else 0,
        "total_generations": len(thought.generations),
        "total_validations": len(thought.validations),
        "total_critiques": len(thought.critiques),
        "total_tool_calls": len(thought.tool_calls),
        "techniques_applied": thought.techniques_applied,
        "is_finalized": thought.final_text is not None,
        "validation_passed": all(v.passed for v in thought.get_current_iteration_validations()),
        "critics_applied": list(set(c.critic for c in thought.critiques)),
        "validators_used": list(set(v.validator for v in thought.validations)),
    }
