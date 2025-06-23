"""Self-Consistency example using Anthropic generator and Gemini critic."""

import asyncio
from sifaka import Runner

async def main():
    runner = Runner(
        critic_name="self_consistency",
        description="Multiple evaluations with consensus",
        prompt="Write about the ethics of genetic engineering",
        min_length=200,
        max_length=1200,
        iterations=4  # Extra iteration for consistency checking
    )
    
    await runner.run(
        model="claude-3-haiku-20240307",  # Anthropic for generation
        critic_model="gemini-1.5-pro-latest",  # Gemini for critique
        critic_temperature=0.1,  # Very low temp for consistent evaluations
        
        # Other available options (with defaults):
        # temperature=0.7,  # Generation temperature (0.0-2.0)
        # force_improvements=True,  # Always run critics even if validation passes
        # show_improvement_prompt=True,  # Display the prompts sent to improve text
        # timeout_seconds=300,  # Maximum time for the entire process
        # storage=None,  # Custom storage backend (default: MemoryStorage)
        # validators=None,  # Custom validators (default: LengthValidator)
    )

if __name__ == "__main__":
    asyncio.run(main())