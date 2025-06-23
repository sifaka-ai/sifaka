"""Meta-Rewarding example using Gemini generator and Anthropic critic."""

import asyncio
from sifaka import Runner
from sifaka.validators import LengthValidator, ReadabilityValidator

async def main():
    runner = Runner(
        critic_name="meta_rewarding",
        description="Two-stage meta-evaluation critique",
        prompt="Describe the future of work in the age of AI",
        min_length=200,
        max_length=1200,
        iterations=3
    )
    
    # Custom validators for meta-evaluation
    validators = [
        LengthValidator(min_length=200, max_length=1200),
        ReadabilityValidator(min_score=50)  # Ensure readability
    ]
    
    await runner.run(
        model="gemini-1.5-flash-latest",  # Gemini for generation
        critic_model="claude-3-5-sonnet-20241022",  # Anthropic for critique
        validators=validators,  # Custom validation criteria
        force_improvements=True,  # Always apply meta-evaluation
        
        # Other available options (with defaults):
        # temperature=0.7,  # Generation temperature (0.0-2.0)
        # critic_temperature=0.3,  # Critic temperature (usually lower for consistency)
        # show_improvement_prompt=True,  # Display the prompts sent to improve text
        # timeout_seconds=300,  # Maximum time for the entire process
        # storage=None,  # Custom storage backend (default: MemoryStorage)
    )

if __name__ == "__main__":
    asyncio.run(main())