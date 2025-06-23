"""Self-Refine example using Anthropic generator and OpenAI critic."""

import asyncio
from sifaka import Runner

async def main():
    runner = Runner(
        critic_name="self_refine",
        description="Iterative refinement across multiple dimensions",
        prompt="Describe the impact of climate change on global agriculture",
        min_length=200,
        max_length=1200,
        iterations=3
    )
    
    await runner.run(
        model="claude-3-5-sonnet-20241022",  # Anthropic for generation
        critic_model="gpt-4-turbo",  # OpenAI for critique
        temperature=0.8,  # Higher temp for creative generation
        critic_temperature=0.2,  # Lower temp for consistent critique
        
        # Other available options (with defaults):
        # force_improvements=True,  # Always run critics even if validation passes
        # show_improvement_prompt=True,  # Display the prompts sent to improve text
        # timeout_seconds=300,  # Maximum time for the entire process
        # storage=None,  # Custom storage backend (default: MemoryStorage)
        # validators=None,  # Custom validators (default: LengthValidator)
    )

if __name__ == "__main__":
    asyncio.run(main())