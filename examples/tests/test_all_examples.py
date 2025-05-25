#!/usr/bin/env python3
"""Test runner for all Sifaka examples.

This script tests all examples to ensure they work correctly and demonstrate
the intended functionality. It provides a comprehensive validation of the
example suite.
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sifaka.utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)


def discover_examples() -> Dict[str, List[Path]]:
    """Discover all example files organized by category."""
    
    examples_dir = Path(__file__).parent.parent
    categories = {}
    
    for category_dir in examples_dir.iterdir():
        if category_dir.is_dir() and category_dir.name != "tests":
            category_name = category_dir.name
            example_files = []
            
            for example_file in category_dir.glob("*.py"):
                if not example_file.name.startswith("__"):
                    example_files.append(example_file)
            
            if example_files:
                categories[category_name] = sorted(example_files)
    
    return categories


def run_example(example_path: Path) -> Dict[str, Any]:
    """Run a single example and return results."""
    
    result = {
        "path": example_path,
        "name": example_path.stem,
        "category": example_path.parent.name,
        "success": False,
        "error": None,
        "output": "",
        "execution_time": 0
    }
    
    try:
        # Import and run the example
        spec = importlib.util.spec_from_file_location(example_path.stem, example_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            
            # Capture output
            import io
            import contextlib
            
            output_buffer = io.StringIO()
            
            import time
            start_time = time.time()
            
            with contextlib.redirect_stdout(output_buffer):
                with contextlib.redirect_stderr(output_buffer):
                    spec.loader.exec_module(module)
                    
                    # Run main function if it exists
                    if hasattr(module, 'main'):
                        module.main()
            
            result["execution_time"] = time.time() - start_time
            result["output"] = output_buffer.getvalue()
            result["success"] = True
            
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    return result


def print_results_summary(results: List[Dict[str, Any]]):
    """Print a summary of test results."""
    
    total_examples = len(results)
    successful_examples = sum(1 for r in results if r["success"])
    failed_examples = total_examples - successful_examples
    
    print("\n" + "="*80)
    print("SIFAKA EXAMPLES TEST SUMMARY")
    print("="*80)
    print(f"\nTotal Examples: {total_examples}")
    print(f"Successful: {successful_examples}")
    print(f"Failed: {failed_examples}")
    print(f"Success Rate: {(successful_examples/total_examples)*100:.1f}%")
    
    # Group by category
    categories = {}
    for result in results:
        category = result["category"]
        if category not in categories:
            categories[category] = {"total": 0, "success": 0}
        categories[category]["total"] += 1
        if result["success"]:
            categories[category]["success"] += 1
    
    print(f"\nResults by Category:")
    for category, stats in categories.items():
        success_rate = (stats["success"] / stats["total"]) * 100
        print(f"  {category}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    # Show failed examples
    if failed_examples > 0:
        print(f"\nFailed Examples:")
        for result in results:
            if not result["success"]:
                print(f"  ✗ {result['category']}/{result['name']}")
                print(f"    Error: {result['error']}")
    
    # Show execution times
    print(f"\nExecution Times:")
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"  {status} {result['category']}/{result['name']}: {result['execution_time']:.2f}s")


def main():
    """Run all examples and report results."""
    
    print("Discovering Sifaka examples...")
    categories = discover_examples()
    
    total_examples = sum(len(files) for files in categories.values())
    print(f"Found {total_examples} examples in {len(categories)} categories")
    
    # List all examples
    print(f"\nExample Categories:")
    for category, files in categories.items():
        print(f"  {category}: {len(files)} examples")
        for example_file in files:
            print(f"    - {example_file.stem}")
    
    print(f"\nRunning examples...")
    print("="*80)
    
    all_results = []
    
    for category, example_files in categories.items():
        print(f"\nTesting {category} examples:")
        
        for example_file in example_files:
            print(f"  Running {example_file.stem}...", end=" ")
            
            result = run_example(example_file)
            all_results.append(result)
            
            if result["success"]:
                print(f"✓ ({result['execution_time']:.2f}s)")
            else:
                print(f"✗ FAILED")
                print(f"    Error: {result['error']}")
    
    # Print summary
    print_results_summary(all_results)
    
    # Return exit code based on results
    failed_count = sum(1 for r in all_results if not r["success"])
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
