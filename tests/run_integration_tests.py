#!/usr/bin/env python3
"""Runner script for Sifaka integration tests with multiple providers."""

import argparse
import os
import subprocess
import sys
from typing import Dict, Optional


def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are available."""
    providers = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "xAI": "XAI_API_KEY",
    }

    available = {}
    for name, env_var in providers.items():
        available[name] = bool(os.getenv(env_var))

    return available


def print_status(available_keys: Dict[str, bool]):
    """Print status of available API keys."""
    print("API Key Status:")
    print("=" * 40)

    for provider, is_available in available_keys.items():
        status = "✓ Available" if is_available else "✗ Not found"
        print(f"{provider:12} {status}")

    available_count = sum(available_keys.values())
    print(f"\nTotal: {available_count}/{len(available_keys)} providers available")

    if available_count == 0:
        print("\n⚠️  No API keys found! Set environment variables:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export ANTHROPIC_API_KEY='your-key'")
        print("   export GOOGLE_API_KEY='your-key'")
        print("   export XAI_API_KEY='your-key'")
        return False

    return True


def run_tests(test_pattern: Optional[str] = None, verbose: bool = True):
    """Run the integration tests."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/test_multi_provider_critics.py",
        "tests/integration/test_new_features.py",
        "-v" if verbose else "",
        "-s",  # Show print statements
        "--tb=short",  # Shorter traceback
    ]

    if test_pattern:
        cmd.extend(["-k", test_pattern])

    # Remove empty strings
    cmd = [c for c in cmd if c]

    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(
        cmd, cwd=os.path.dirname(os.path.dirname(__file__)), check=False
    )
    return result.returncode


def run_specific_test_sets():
    """Run specific sets of tests."""
    test_sets = {
        "1": ("Basic Multi-Provider Tests", "test_all_critics_all_providers"),
        "2": ("Provider Consistency", "test_critic_consistency_across_providers"),
        "3": ("Enhanced N-Critics", "TestNCriticsEnhancements"),
        "4": ("Advanced Confidence", "TestAdvancedConfidenceCalculation"),
        "5": ("SelfRAG with Retrieval", "TestSelfRAGEnhancements"),
        "6": ("Composable Validators", "TestComposableValidators"),
        "7": ("Performance Tests", "test_provider_latency"),
        "8": ("All Integration Tests", None),
    }

    print("\nAvailable Test Sets:")
    print("=" * 60)
    for key, (name, _) in test_sets.items():
        print(f"{key}. {name}")

    choice = input("\nSelect test set (1-8) or press Enter for all: ").strip()

    if choice in test_sets:
        name, pattern = test_sets[choice]
        print(f"\nRunning: {name}")
        return run_tests(pattern)
    else:
        print("\nRunning all integration tests")
        return run_tests()


def run_quick_smoke_test():
    """Run a quick smoke test with one provider."""
    print("\nRunning quick smoke test...")

    # Find first available provider
    available = check_api_keys()
    provider = None

    for p, is_available in available.items():
        if is_available:
            provider = p
            break

    if not provider:
        print("No providers available for smoke test")
        return 1

    print(f"Using {provider} for smoke test")

    # Run a simple test
    test_code = """
import asyncio
from sifaka import improve

async def test():
    result = await improve(
        "AI is important",
        critics=["reflexion"],
        max_iterations=1
    )
    print(f"Success! Improved text: {result.final_text[:50]}...")
    return 0

asyncio.run(test())
"""

    result = subprocess.run([sys.executable, "-c", test_code], check=False)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Sifaka integration tests with multiple providers"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test with first available provider",
    )
    parser.add_argument("--pattern", "-k", help="Run only tests matching this pattern")
    parser.add_argument(
        "--no-verbose", "-q", action="store_true", help="Less verbose output"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode - choose which tests to run",
    )

    args = parser.parse_args()

    print("Sifaka Integration Test Runner")
    print("=" * 60)

    # Check API keys
    available = check_api_keys()
    if not print_status(available):
        return 1

    print("")

    # Run appropriate tests
    if args.quick:
        return run_quick_smoke_test()
    elif args.interactive:
        return run_specific_test_sets()
    else:
        return run_tests(args.pattern, not args.no_verbose)


if __name__ == "__main__":
    sys.exit(main())
