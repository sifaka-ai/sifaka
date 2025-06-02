"""Simple test for the new PydanticAI-based critics."""

import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from sifaka.core.thought import Thought
from sifaka.critics.constitutional import ConstitutionalCritic

# Load environment variables
load_dotenv()


async def test_simple():
    """Simple test of the ConstitutionalCritic."""
    print("Testing ConstitutionalCritic...")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Create critic
    critic = ConstitutionalCritic(
        model_name="openai:gpt-4o-mini"
    )
    
    # Create test thought
    thought = Thought(
        prompt="Write about AI safety",
        text="AI is completely safe and will never cause any problems.",
        timestamp=datetime.now(),
        id="test_001"
    )
    
    print(f"Original text: {thought.text}")
    
    try:
        # Test critique
        print("Running critique...")
        result = await critic.critique_async(thought)
        
        print(f"Success: {result.success}")
        print(f"Needs improvement: {result.feedback.needs_improvement}")
        print(f"Message: {result.feedback.message}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple())
