"""
Script to run Sifaka benchmarks.
"""

import os
from typing import Optional

from benchmark_sifaka import SifakaBenchmark, print_benchmark_results


def get_api_key(env_var: str) -> Optional[str]:
    """Get API key from environment variable."""
    return os.getenv(env_var)


def main():
    # Get API keys from environment variables
    anthropic_api_key = get_api_key("ANTHROPIC_API_KEY")
    guardrails_api_key = get_api_key("GUARDRAILS_API_KEY")

    # Configure Guardrails
    guardrails_config = {
        "api_key": guardrails_api_key,
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
    } if guardrails_api_key else None

    # Initialize benchmark
    benchmark = SifakaBenchmark(
        num_samples=100,  # Reduced for initial testing
        warm_up_rounds=2,
        text_length=200,
        api_key=anthropic_api_key,
        guardrails_config=guardrails_config,
    )

    # Run benchmarks
    results = benchmark.run_all_benchmarks(sample_size=50)  # Reduced sample size for initial testing

    # Print results
    print_benchmark_results(results)


if __name__ == "__main__":
    main()
