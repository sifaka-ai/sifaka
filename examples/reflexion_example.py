"""Reflexion example using OpenAI generator and Anthropic critic."""

import asyncio
from sifaka import Runner

async def main():
    runner = Runner(
        critic_name="reflexion",
        description="Self-reflection based improvement",
        prompt="Write a brief introduction to machine learning",
        min_length=200,
        max_length=1200,
        iterations=3
    )
    
    await runner.run(
        model="gpt-4o-mini",  # OpenAI for generation
        critic_model="claude-3-5-sonnet-20241022",  # Anthropic for critique
        
        # Other available options (with defaults):
        # temperature=0.7,  # Generation temperature (0.0-2.0)
        # critic_temperature=0.3,  # Critic temperature (usually lower for consistency)
        # force_improvements=True,  # Always run critics even if validation passes
        # show_improvement_prompt=True,  # Display the prompts sent to improve text
        # timeout_seconds=300,  # Maximum time for the entire process
        # storage=None,  # Custom storage backend (default: MemoryStorage)
        # validators=None,  # Custom validators (default: LengthValidator)
    )

if __name__ == "__main__":
    asyncio.run(main())