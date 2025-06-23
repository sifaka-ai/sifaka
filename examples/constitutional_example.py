"""Constitutional example using Gemini generator and Groq critic."""

import asyncio
from sifaka import Runner

async def main():
    runner = Runner(
        critic_name="constitutional",
        description="Principled critique against constitutional principles",
        prompt="Explain the benefits of artificial intelligence in healthcare",
        min_length=150,
        max_length=1200,
        iterations=3
    )
    
    await runner.run(
        model="gemini-1.5-flash",  # Gemini for generation
        critic_model="llama-3.3-70b-versatile",  # Groq for critique
        force_improvements=True,  # Always apply constitutional principles
        
        # Other available options (with defaults):
        # temperature=0.7,  # Generation temperature (0.0-2.0)
        # critic_temperature=0.3,  # Critic temperature (usually lower for consistency)
        # show_improvement_prompt=True,  # Display the prompts sent to improve text
        # timeout_seconds=300,  # Maximum time for the entire process
        # storage=None,  # Custom storage backend (default: MemoryStorage)
        # validators=None,  # Custom validators (default: LengthValidator)
    )

if __name__ == "__main__":
    asyncio.run(main())