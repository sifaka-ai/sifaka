"""Self-RAG example using OpenAI generator and Groq critic."""

import asyncio
from sifaka import Runner
from sifaka.storage import FileStorage

async def main():
    runner = Runner(
        critic_name="self_rag",
        description="Retrieval-aware self-critique",
        prompt="Explain quantum computing and its applications in cryptography",
        min_length=200,
        max_length=1200,
        iterations=3
    )
    
    await runner.run(
        model="gpt-4o",  # OpenAI for generation
        critic_model="mixtral-8x7b-32768",  # Groq for critique
        storage=FileStorage("./rag_results"),  # Save results to disk
        timeout_seconds=600,  # Longer timeout for fact-checking
        
        # Other available options (with defaults):
        # temperature=0.7,  # Generation temperature (0.0-2.0)
        # critic_temperature=0.3,  # Critic temperature (usually lower for consistency)
        # force_improvements=True,  # Always run critics even if validation passes
        # show_improvement_prompt=True,  # Display the prompts sent to improve text
        # validators=None,  # Custom validators (default: LengthValidator)
    )

if __name__ == "__main__":
    asyncio.run(main())