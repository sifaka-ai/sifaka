"""
LAC (LLM-Based Actor-Critic) Critic Comparison Demo

This example demonstrates how to use the different critic types in the LAC module:
- FeedbackCritic: Provides natural language feedback
- ValueCritic: Estimates numeric values
- LACCritic: Combines both feedback and value scoring

Based on: Language Feedback Improves Language Model-based Decision Making
https://arxiv.org/abs/2403.03692
"""

import os
import time

from sifaka.critics import (
    create_feedback_critic,
    create_value_critic,
    create_lac_critic,
)
from sifaka.models.openai import create_openai_provider

def main():
    # Get API key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    
    # Create a language model provider
    provider = create_openai_provider(
        model_name="gpt-3.5-turbo",
        api_key=openai_api_key,
        temperature=0.7,
        max_tokens=1000
    )
    
    # Create the critics
    feedback_critic = create_feedback_critic(
        llm_provider=provider,
        name="feedback_critic",
        description="A critic that provides natural language feedback",
        system_prompt="You are an expert at providing constructive feedback.",
        temperature=0.7,
        max_tokens=1000
    )
    
    value_critic = create_value_critic(
        llm_provider=provider,
        name="value_critic",
        description="A critic that estimates numeric values",
        system_prompt="You are an expert at estimating the quality of responses.",
        temperature=0.3,
        max_tokens=100
    )
    
    lac_critic = create_lac_critic(
        llm_provider=provider,
        name="lac_critic",
        description="A critic that combines feedback and value scoring",
        system_prompt="You are an expert at evaluating and improving text.",
        temperature=0.7,
        max_tokens=1000
    )
    
    # Define tasks and responses
    tasks = [
        "Explain the concept of quantum computing in simple terms.",
        "Write a short poem about the changing seasons.",
        "Describe the process of photosynthesis in 3 steps."
    ]
    
    # Process each task
    for i, task in enumerate(tasks):
        print(f"\n\n{'='*80}")
        print(f"TASK {i+1}: {task}")
        print(f"{'='*80}\n")
        
        # Generate an initial response
        initial_response = provider.generate(f"Task: {task}")
        print(f"INITIAL RESPONSE:\n{initial_response}\n")
        
        # Use the feedback critic
        print(f"\n{'*'*40}")
        print("FEEDBACK CRITIC")
        print(f"{'*'*40}\n")
        
        start_time = time.time()
        feedback = feedback_critic.run(task, initial_response)
        feedback_time = time.time() - start_time
        
        print(f"Feedback:\n{feedback}")
        print(f"Time: {feedback_time:.2f} seconds\n")
        
        # Use the value critic
        print(f"\n{'*'*40}")
        print("VALUE CRITIC")
        print(f"{'*'*40}\n")
        
        start_time = time.time()
        value = value_critic.run(task, initial_response)
        value_time = time.time() - start_time
        
        print(f"Value: {value:.2f}")
        print(f"Time: {value_time:.2f} seconds\n")
        
        # Use the LAC critic
        print(f"\n{'*'*40}")
        print("LAC CRITIC")
        print(f"{'*'*40}\n")
        
        start_time = time.time()
        lac_result = lac_critic.run(task, initial_response)
        lac_time = time.time() - start_time
        
        print(f"Feedback:\n{lac_result['feedback']}")
        print(f"Value: {lac_result['value']:.2f}")
        print(f"Time: {lac_time:.2f} seconds\n")
        
        # Improve the response using each critic
        print(f"\n{'*'*40}")
        print("IMPROVED RESPONSES")
        print(f"{'*'*40}\n")
        
        # Improve with feedback critic
        start_time = time.time()
        feedback_improved = feedback_critic.improve(initial_response, {"task": task})
        feedback_improve_time = time.time() - start_time
        
        print(f"Feedback Critic Improved Response:\n{feedback_improved}")
        print(f"Time: {feedback_improve_time:.2f} seconds\n")
        
        # Improve with value critic
        start_time = time.time()
        value_improved = value_critic.improve(initial_response, {"task": task})
        value_improve_time = time.time() - start_time
        
        print(f"Value Critic Improved Response:\n{value_improved}")
        print(f"Time: {value_improve_time:.2f} seconds\n")
        
        # Improve with LAC critic
        start_time = time.time()
        lac_improved = lac_critic.improve(initial_response, {"task": task})
        lac_improve_time = time.time() - start_time
        
        print(f"LAC Critic Improved Response:\n{lac_improved}")
        print(f"Time: {lac_improve_time:.2f} seconds\n")
        
        # Compare the quality of the improved responses
        print(f"\n{'*'*40}")
        print("QUALITY COMPARISON")
        print(f"{'*'*40}\n")
        
        # Evaluate the improved responses with the value critic
        original_value = value_critic.run(task, initial_response)
        feedback_value = value_critic.run(task, feedback_improved)
        value_value = value_critic.run(task, value_improved)
        lac_value = value_critic.run(task, lac_improved)
        
        print(f"Original Response Value: {original_value:.2f}")
        print(f"Feedback Critic Improved Value: {feedback_value:.2f}")
        print(f"Value Critic Improved Value: {value_value:.2f}")
        print(f"LAC Critic Improved Value: {lac_value:.2f}\n")
        
        # Print a summary
        print(f"\n{'*'*40}")
        print("SUMMARY")
        print(f"{'*'*40}\n")
        
        print(f"Task: {task}")
        print(f"Original Response Value: {original_value:.2f}")
        print(f"Feedback Critic: {feedback_value:.2f} (Improvement: {feedback_value - original_value:.2f})")
        print(f"Value Critic: {value_value:.2f} (Improvement: {value_value - original_value:.2f})")
        print(f"LAC Critic: {lac_value:.2f} (Improvement: {lac_value - original_value:.2f})")
        
        # Determine the best critic
        best_value = max(feedback_value, value_value, lac_value)
        if best_value == feedback_value:
            best_critic = "Feedback Critic"
        elif best_value == value_value:
            best_critic = "Value Critic"
        else:
            best_critic = "LAC Critic"
        
        print(f"Best Critic: {best_critic} with value {best_value:.2f}")

if __name__ == "__main__":
    main()
