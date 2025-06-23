"""Run all critic examples to demonstrate the workflow."""

import asyncio
import subprocess
import sys
from pathlib import Path


async def run_example(example_path: Path):
    """Run a single example."""
    print(f"\n{'='*80}")
    print(f"Running: {example_path.name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout per example
        )
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"ERROR in {example_path.name}:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {example_path.name} took too long")
        return False
    except Exception as e:
        print(f"ERROR running {example_path.name}: {e}")
        return False


async def main():
    """Run all critic examples."""
    examples_dir = Path("examples")
    
    # Get all critic example files
    example_files = sorted([
        f for f in examples_dir.glob("*_example.py")
        if f.name != "run_all_examples.py"
    ])
    
    print(f"Found {len(example_files)} critic examples to run:")
    for f in example_files:
        print(f"  - {f.name}")
    
    # Run each example
    successful = 0
    failed = 0
    
    for example in example_files:
        if await run_example(example):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(example_files)}")
    
    # Check thoughts directory
    thoughts_dir = Path("thoughts")
    if thoughts_dir.exists():
        thought_files = list(thoughts_dir.glob("*.json"))
        print(f"\n📝 Thought files created: {len(thought_files)}")
        for f in sorted(thought_files)[-5:]:  # Show last 5
            print(f"   - {f.name}")


if __name__ == "__main__":
    asyncio.run(main())