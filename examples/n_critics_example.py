"""N-Critics example using Groq generator and Gemini critic."""

import asyncio
from sifaka import Runner

async def main():
    runner = Runner(
        critic_name="n_critics",
        description="Multi-perspective evaluation",
        prompt="How can renewable energy address climate change?",
        min_length=200,
        max_length=1200,
        iterations=3
    )
    
    await runner.run(
        model="llama-3.3-70b-versatile",  # Groq for generation
        critic_model="gemini-1.5-pro",  # Gemini for critique
        show_improvement_prompt=True,  # See how multiple perspectives combine
        
        # Other available options (with defaults):
        # temperature=0.7,  # Generation temperature (0.0-2.0)
        # critic_temperature=0.3,  # Critic temperature (usually lower for consistency)
        # force_improvements=True,  # Always run critics even if validation passes
        # timeout_seconds=300,  # Maximum time for the entire process
        # storage=None,  # Custom storage backend (default: MemoryStorage)
        # validators=None,  # Custom validators (default: LengthValidator)
    )

if __name__ == "__main__":
    asyncio.run(main())