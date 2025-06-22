"""Example of using the Reflexion critic."""

import asyncio
from sifaka import improve


async def main():
    text = "Python is used for web development."
    
    result = await improve(
        text,
        critics=["reflexion"],
        max_iterations=2,
        show_improvement_prompt=True
    )
    
    print(f"Original: {text}")
    print(f"Improved: {result.final_text}")


if __name__ == "__main__":
    asyncio.run(main())