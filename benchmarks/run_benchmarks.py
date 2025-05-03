"""
Script to run Sifaka benchmarks.
"""

import os
from typing import Optional

from sifaka.benchmarks.benchmark_guardrails import SifakaBenchmark, print_benchmark_results
from sifaka.benchmarks.benchmark_config import GUARDRAILS_CONFIG


def get_api_key(env_var: str) -> Optional[str]:
    """Get API key from environment variable."""
    return os.getenv(env_var)


def main():
    # Get API keys from environment variables
    anthropic_api_key = get_api_key("ANTHROPIC_API_KEY")
    guardrails_api_key = get_api_key("GUARDRAILS_API_KEY")

    # Configure Guardrails
    guardrails_config = GUARDRAILS_CONFIG.copy()
    if guardrails_api_key:
        guardrails_config["api_key"] = guardrails_api_key
    else:
        guardrails_config = None

    # Initialize benchmark
    benchmark = SifakaBenchmark(
        num_samples=5,  # Reduced for faster testing
        warm_up_rounds=2,
        text_length=200,
        api_key=anthropic_api_key,
        guardrails_config=guardrails_config,
    )

    # Run benchmarks
    results = benchmark.run_all_benchmarks(sample_size=5)  # Reduced sample size for faster testing

    # Print results
    print_benchmark_results(results)


if __name__ == "__main__":
    main()
