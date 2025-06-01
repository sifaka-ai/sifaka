#!/usr/bin/env python3
"""Claude 4 Generator with Gemini Flash Critics Example.

This example demonstrates:
- PydanticAI agent with Anthropic Claude 4 (claude-3-5-sonnet-latest) for generation
- Constitutional critic using Gemini Flash for principled evaluation
- Reflexion critic using Gemini Flash for self-reflection improvement
- Length validator to ensure appropriate response length
- Sentiment classifier validator to ensure positive sentiment
- Hybrid storage: File storage for thoughts directory + MCP Redis for cross-process sharing
- Chain termination after validation passes
- Complete observability with thought persistence

The chain will generate a motivational blog post about overcoming challenges,
with critics ensuring constitutional principles and self-reflection improvements,
and validators ensuring proper length and positive sentiment.

Prerequisites:
1. Install mcp-redis server: cd mcp/mcp-redis && uv venv && source .venv/bin/activate && uv sync
2. Start Redis server: redis-server (optional, for MCP Redis storage)
3. Set environment variables: ANTHROPIC_API_KEY, GOOGLE_API_KEY (or GEMINI_API_KEY)
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent

from sifaka.agents import create_pydantic_chain
from sifaka.classifiers.sentiment import create_sentiment_validator
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.models import create_model
from sifaka.storage import CachedStorage, FileStorage, MemoryStorage, RedisStorage
from sifaka.utils.logging import get_logger
from sifaka.validators import LengthValidator

# Import PydanticAI MCP for Redis
from pydantic_ai.mcp import MCPServerStdio

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def setup_storage():
    """Set up layered storage: Memory → Redis → File for optimal performance and persistence."""

    # Create thoughts directory if it doesn't exist
    thoughts_dir = Path("thoughts")
    thoughts_dir.mkdir(exist_ok=True)

    # Layer 1: Memory storage (fastest)
    memory_storage = MemoryStorage()

    # Layer 2: Redis storage using the MCP Redis server in the project
    redis_mcp_server = MCPServerStdio(
        "uv",
        args=[
            "run",
            "--directory",
            "/Users/evanvolgas/Documents/not_beam/sifaka/mcp/mcp-redis",
            "src/main.py",
        ],
        tool_prefix="redis",
    )
    redis_storage = RedisStorage(redis_mcp_server=redis_mcp_server, key_prefix="sifaka:test")

    # Layer 3: File storage (local debugging and backup)
    file_storage = FileStorage(
        "thoughts/claude_gemini_example_thoughts.json",
        overwrite=True,  # Overwrite existing file for clean runs
    )

    # Create two-tier cached storage: Redis and Filesystem
    cached_storage_persistent = CachedStorage(
        cache=redis_storage,  # L1: Redis (persistent)
        persistence=file_storage,  # L2: Disk (persistent)
    )

    # Create three-tier cached storage: Memory → Redis → File
    cached_storage = CachedStorage(
        cache=memory_storage,  # L1: Memory (fastest)
        persistence=cached_storage_persistent,  # L2+L3: Redis + File (persistent)
    )

    return cached_storage, file_storage


async def main():
    """Run the Claude 4 Generator with Gemini Flash Critics example."""

    # Ensure API keys are available
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required")

    logger.info("Creating Claude 4 Generator with Gemini Flash Critics example")

    # Set up layered storage: Memory → Redis → File
    cached_storage, file_storage = setup_storage()

    # Create PydanticAI agent with Anthropic Claude 4
    logger.info("Creating PydanticAI agent with Claude 4 (claude-3-5-sonnet-latest)")
    agent = Agent(
        "anthropic:claude-3-5-sonnet-latest",
        system_prompt="""You are an expert motivational writer and life coach.
        Your task is to write inspiring, uplifting blog posts that help people overcome challenges and achieve their goals.

        Write with:
        - Authentic personal stories and examples
        - Practical, actionable advice
        - Positive, encouraging tone
        - Clear structure with engaging headings
        - Specific strategies and techniques
        - Inspirational but realistic messaging

        Your writing should be professional yet warm, evidence-based yet accessible.""",
    )

    # Create Gemini Flash model for critics
    logger.info("Creating Gemini Flash model for critics")
    critic_model = create_model("google-gla:gemini-1.5-flash")

    # Create Constitutional critic with Gemini Flash
    logger.info("Creating Constitutional critic with Gemini Flash")
    constitutional_critic = ConstitutionalCritic(
        model=critic_model,
        principles=[
            "Content should be truthful, accurate, and evidence-based",
            "Writing should be inclusive and respectful of diverse perspectives",
            "Advice should be practical and actionable, not just theoretical",
            "Tone should be encouraging and supportive, never condescending",
            "Content should promote healthy coping strategies and realistic expectations",
        ],
        strict_mode=False,  # Allow some flexibility in interpretation
    )

    # Create Reflexion critic with Gemini Flash
    logger.info("Creating Reflexion critic with Gemini Flash")
    reflexion_critic = ReflexionCritic(
        model=critic_model,
        max_memory_size=15,  # Keep more reflections in memory for learning
    )

    # Create Length validator to ensure comprehensive content
    logger.info("Creating Length validator")
    length_validator = LengthValidator(
        min_length=800,  # Minimum 800 characters for substantial content
        max_length=2000,  # Maximum 2000 characters to keep it focused
        name="BlogPostLengthValidator",
    )

    # Create Sentiment validator to ensure positive tone
    logger.info("Creating Sentiment validator")
    sentiment_validator = create_sentiment_validator(
        required_sentiment="positive",  # Must be positive sentiment
        min_confidence=0.7,  # High confidence threshold
        name="PositiveSentimentValidator",
    )

    # Create PydanticAI chain with both critics and validators
    logger.info("Creating PydanticAI chain with Claude 4 generator and Gemini Flash critics")
    chain = create_pydantic_chain(
        agent=agent,
        validators=[length_validator, sentiment_validator],  # Both validators must pass
        critics=[constitutional_critic, reflexion_critic],  # Both critics provide feedback
        max_improvement_iterations=3,  # Allow up to 3 improvement iterations
        always_apply_critics=False,  # Only apply critics when validation fails (terminate after validation passes)
        analytics_storage=cached_storage,  # Use layered storage for optimal performance
    )

    print(
        f"DEBUG: Created PydanticAI chain with {len([constitutional_critic, reflexion_critic])} critics and {len([length_validator, sentiment_validator])} validators"
    )

    # Define the prompt for motivational blog post
    prompt = """Write a motivational blog post titled "Turning Setbacks into Comebacks: 5 Strategies for Resilient Success"

    The blog post should:
    - Share a compelling personal story or example of overcoming adversity
    - Provide 5 specific, actionable strategies for building resilience
    - Include practical exercises or techniques readers can implement immediately
    - Maintain an encouraging, positive tone throughout
    - Be well-structured with clear headings and flow
    - Offer hope and inspiration while being realistic about challenges

    Target audience: Young professionals and entrepreneurs facing career or business challenges."""

    # Store initial thought in layered storage (Memory → Redis)
    logger.info("Storing initial prompt in layered storage")
    try:
        current_timestamp = datetime.now().isoformat()
        await cached_storage.set(
            "initial_prompt",
            {
                "prompt": prompt,
                "timestamp": current_timestamp,
                "example_type": "claude_gemini_example",
                "generator": "claude-3-5-sonnet-latest",
                "critics": ["constitutional", "reflexion"],
                "validators": ["length", "sentiment"],
            },
        )
        logger.info("Successfully stored initial prompt in layered storage")
    except Exception as e:
        logger.warning(f"Failed to store in layered storage: {e}")

    # Run the chain
    logger.info("Running PydanticAI chain with Claude 4 generator and Gemini Flash critics...")
    result = await chain.run(prompt)

    # Store final result in layered storage
    try:
        final_timestamp = datetime.now().isoformat()
        await cached_storage.set(
            "final_result",
            {
                "text": result.text,
                "iteration": result.iteration,
                "chain_id": result.chain_id,
                "timestamp": final_timestamp,
                "success": True,
            },
        )
        logger.info("Successfully stored final result in layered storage")
    except Exception as e:
        logger.warning(f"Failed to store final result in layered storage: {e}")

    # Display results
    print("\n" + "=" * 80)
    print("CLAUDE 4 GENERATOR WITH GEMINI FLASH CRITICS EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Blog Post ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nProcessing Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Generator: Claude 4 (claude-3-5-sonnet-latest)")
    print(f"  Critics: Constitutional + Reflexion (Gemini Flash)")
    print(f"  Validators: Length + Sentiment")
    print(f"  Storage: Layered (Memory → Redis → File)")
    print(f"  Termination: After validation passes")

    # Show validation results
    if hasattr(result, "validation_results") and result.validation_results:
        print(f"\nValidation Results:")
        for i, validation_result in enumerate(result.validation_results, 1):
            # Handle both ValidationResult objects and string results
            if hasattr(validation_result, "is_valid"):
                status = "✅ PASSED" if validation_result.is_valid else "❌ FAILED"
                validator_name = getattr(validation_result, "validator_name", f"Validator {i}")
                print(f"  {i}. {validator_name}: {status}")
                if (
                    not validation_result.is_valid
                    and hasattr(validation_result, "issues")
                    and validation_result.issues
                ):
                    for issue in validation_result.issues[:2]:  # Show first 2 issues
                        print(f"     - {issue}")
            else:
                # Handle string results
                print(f"  {i}. Validator {i}: {validation_result}")

    # Show critic feedback summary
    if hasattr(result, "critic_results") and result.critic_results:
        print(f"\nCritic Feedback Summary:")
        for i, critic_result in enumerate(result.critic_results, 1):
            improvement_needed = "Yes" if critic_result.get("needs_improvement", False) else "No"
            print(
                f"  {i}. {critic_result.get('critic_name', 'Unknown')}: Improvement needed: {improvement_needed}"
            )

    print(f"\nThoughts saved to: Layered storage (Memory → Redis → File)")
    print(f"File backup: thoughts/claude_gemini_example_thoughts.json")
    print(f"Redis prefix: sifaka:test")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
