#!/usr/bin/env python3
"""PydanticAI Reflexion Critic Self-Evaluation Improvement Example.

This example demonstrates:
- Using PydanticAI agent with Gemini model for generation
- Using a smaller model for the ReflexionCritic
- Generating a software engineer self-evaluation with specific characteristics
- Using ReflexionCritic to improve the self-evaluation through reflection
- Two rounds of improvement (max_improvement_iterations=2)
- No validation - just generation and improvement
- File storage for thought persistence
- PydanticAI chain architecture for better critic feedback storage

The chain will generate a self-evaluation for a software engineer highlighting:
- Above average technical performance
- Exceptional knowledge of AI
- Need for improvement in communication skills

The ReflexionCritic will then improve the self-evaluation through self-reflection.
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic_ai import Agent

from sifaka.agents import create_pydantic_chain
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.models.base import create_model
from sifaka.storage import FileStorage
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


async def main():
    """Run the Reflexion Critic Self-Evaluation Improvement example."""

    # Ensure API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    logger.info("Creating Reflexion Critic Self-Evaluation Improvement example")

    # Create PydanticAI agent for generation (Gemini Flash)
    agent = Agent(
        "google-gla:gemini-1.5-flash",
        system_prompt="You are an expert software engineer writing a professional self-evaluation for an annual performance review. You excel at highlighting technical achievements while maintaining professional humility.",
    )
    logger.info("Created PydanticAI agent with Gemini Flash")

    # Create regular model for critic (Gemini Flash) - avoid PydanticAI to prevent event loop conflicts
    critic_model = create_model("google-gla:gemini-1.5-flash", temperature=0.8, max_tokens=600)
    logger.info("Created critic model: Gemini Flash")

    # Set up file storage for thoughts
    storage = FileStorage(
        file_path="thoughts/reflexion_self_evaluation_thoughts.json", overwrite=True
    )
    logger.info("Using file storage for thoughts")

    # Create ReflexionCritic with the model
    critic = ReflexionCritic(
        model=critic_model,
        # Custom prompts for self-evaluation improvement
        critique_prompt_template="""
You are an expert HR professional and performance review specialist.
Analyze this software engineer's self-evaluation for clarity, professionalism, specificity, and impact.

Original Prompt: {prompt}
Self-Evaluation Text: {text}
Context: {context}

Provide a detailed critique focusing on:
1. Professional tone and language
2. Specific examples and quantifiable achievements
3. Balance between strengths and areas for improvement
4. Clarity and structure of the evaluation
5. How well it demonstrates technical expertise and AI knowledge
6. Whether communication improvement areas are appropriately addressed

Critique:""",
        reflection_prompt_template="""
You are reflecting on your critique of a software engineer's self-evaluation.

Original Prompt: {prompt}
Self-Evaluation: {text}
Your Critique: {critique}

Trial Context:
- Trial Number: {trial_number}
- Previous Attempts: {previous_attempts}
- Success Patterns: {success_patterns}
- Failure Patterns: {failure_patterns}
- External Feedback: {external_feedback}
- Episodic Memory: {episodic_memory}

Reflect deeply on your critique:
1. What specific improvements would make this self-evaluation more compelling?
2. How can the technical achievements be better highlighted?
3. What examples or metrics could strengthen the AI expertise claims?
4. How should the communication improvement area be better framed?
5. What would make this evaluation stand out to managers and peers?

Based on this reflection, what are the most important changes needed?

Reflection:""",
        improve_prompt_template="""
You are an expert career coach helping a software engineer improve their self-evaluation.

Original Prompt: {prompt}
Current Self-Evaluation: {text}
Context: {context}
Critique: {critique}
Reflection: {reflection}

Based on the critique and reflection, rewrite this self-evaluation to be more:
- Professional and polished
- Specific with concrete examples and metrics
- Balanced in highlighting strengths while acknowledging growth areas
- Clear in demonstrating technical excellence and AI expertise
- Thoughtful in addressing communication improvement needs

Keep the core message about above-average technical performance, exceptional AI knowledge, and communication improvement needs, but make it more compelling and professional.

Improved Self-Evaluation:""",
    )

    # Create the PydanticAI chain with 2 improvement iterations
    chain = create_pydantic_chain(
        agent=agent,
        critics=[critic],  # ReflexionCritic
        max_improvement_iterations=2,  # Two rounds of improvement
        always_apply_critics=True,  # Always apply the critic
        analytics_storage=storage,  # File storage for thoughts
    )

    # Run the chain with the prompt
    prompt = """Write a professional self-evaluation for a software engineer's annual performance review.

The evaluation should highlight:
- Above average technical performance with specific examples
- Exceptional knowledge and expertise in AI/machine learning
- Recognition that communication skills need improvement
- Professional tone appropriate for a corporate performance review
- Specific achievements and contributions from the past year

The self-evaluation should be honest, professional, and demonstrate strong technical capabilities while acknowledging areas for growth."""

    logger.info("Running PydanticAI chain with ReflexionCritic for self-evaluation improvement...")
    result = await chain.run(prompt)

    # Display results
    print("\n" + "=" * 80)
    print("REFLEXION CRITIC SELF-EVALUATION IMPROVEMENT EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Self-Evaluation ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nIterations: {result.iteration}")
    print(f"Max Iterations: 2 (as specified)")
    print(f"Chain ID: {result.chain_id}")

    # Display basic result information
    print(f"\n📊 Result Information:")
    print(f"  - Text length: {len(result.text or '')} characters")
    print(f"  - Validation results: {len(result.validation_results or {})}")
    print(f"  - Tool calls: {len(result.tool_calls or [])}")
    if result.metadata:
        print(f"  - Metadata keys: {list(result.metadata.keys())}")

    # Show critic feedback and improvements
    if result.critic_feedback:
        print(f"\nReflexionCritic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  Iteration {i}:")
            print(f"    Critic: {feedback.critic_name}")
            print(f"    Needs Improvement: {feedback.needs_improvement}")
            if feedback.feedback:
                print(f"    Critique: {feedback.feedback[:200]}...")
            if hasattr(feedback, "metadata") and feedback.metadata.get("reflection"):
                print(f"    Reflection: {feedback.metadata['reflection'][:200]}...")
    else:
        print(f"\nNo critic feedback found in result object")

    # Check if thoughts were saved with critic feedback
    print(f"\nChecking saved thoughts for critic feedback...")
    try:
        import json

        with open("thoughts/reflexion_self_evaluation_thoughts.json", "r") as f:
            saved_thoughts = json.load(f)

        feedback_count = 0
        for _, thought_data in saved_thoughts.items():
            if thought_data.get("critic_feedback"):
                feedback_count += 1
                print(f"  Thought {thought_data['iteration']}: Has critic feedback")
            else:
                print(f"  Thought {thought_data['iteration']}: No critic feedback stored")

        print(f"Total thoughts with critic feedback: {feedback_count}/{len(saved_thoughts)}")
    except Exception as e:
        print(f"Error reading saved thoughts: {e}")

    print(f"\nModel Architecture:")
    print(f"  Generator: Gemini Flash")
    print(f"  Critic: Gemini Flash")
    print(f"\nValidation: BYPASSED (no validators used)")
    print(f"\nStorage: File storage (thoughts/)")
    print("\n" + "=" * 80)
    logger.info("Reflexion Critic self-evaluation improvement example completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
