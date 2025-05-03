"""
Benchmarking suite for Sifaka with Guardrails integration.

This module provides tools to measure:
1. Validation performance
2. Rule execution performance
3. Model provider performance
4. Prompt critic performance with Guardrails
"""

import gc
import statistics
import time
from typing import Any, Dict, List, Optional
import numpy as np
import psutil
from tqdm import tqdm

from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
from sifaka.rules.base import Rule, RuleResult
from sifaka.rules.formatting.length import create_length_rule
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

class SifakaBenchmark:
    """Benchmark suite for Sifaka with Guardrails integration."""

    def __init__(
        self,
        num_samples: int = 100,
        warm_up_rounds: int = 2,
        text_length: int = 200,
        api_key: Optional[str] = None,
        guardrails_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the benchmark suite.

        Args:
            num_samples: Number of text samples to use
            warm_up_rounds: Number of warm-up rounds before measurement
            text_length: Length of text samples to generate
            api_key: API key for model provider
            guardrails_config: Configuration for Guardrails integration
        """
        self.num_samples = num_samples
        self.warm_up_rounds = warm_up_rounds
        self.text_length = text_length
        self.api_key = api_key
        self.guardrails_config = guardrails_config

        # Initialize components
        self._init_model()
        self._init_rules()
        self._init_critic()
        self._generate_test_data()

    def _init_model(self):
        """Initialize the model provider."""
        if not self.api_key:
            raise ValueError("API key is required for benchmarking")

        self.model = AnthropicProvider(
            model_name="claude-3-sonnet-20240229",
            config=ModelConfig(
                api_key=self.api_key,
                temperature=0.7,
                max_tokens=1000,
            ),
        )

    def _init_rules(self):
        """Initialize benchmark rules."""
        self.rules = {
            "length": create_length_rule(
                min_words=50,
                max_words=500,
                rule_id="length_rule",
                description="Validates text length",
            )
        }

    def _init_critic(self):
        """Initialize the prompt critic with Guardrails if configured."""
        if self.guardrails_config:
            self.critic = PromptCritic(
                name="benchmark_critic",
                description="Critic for benchmark testing",
                llm_provider=self.model,
                config=PromptCriticConfig(
                    name="benchmark_critic",
                    description="Helps ensure text meets benchmark requirements",
                    system_prompt=(
                        "You are a helpful editor who ensures text meets the following requirements:\n"
                        "1. Length between 50-500 words\n"
                        "2. Clear and coherent content\n"
                        "3. Appropriate formatting"
                    ),
                    temperature=0.5,
                ),
            )
        else:
            self.critic = None

    def _generate_test_data(self):
        """Generate test data for benchmarking."""
        # Simple test data generation
        self.test_data = [
            f"This is test sample {i} " * (self.text_length // 4)
            for i in range(self.num_samples)
        ]

    def _measure_memory(self, func, *args):
        """Measure memory usage of a function."""
        gc.collect()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()
        result = func(*args)
        elapsed = time.time() - start_time

        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        return {
            "result": result,
            "elapsed": elapsed,
            "memory_diff_mb": mem_after - mem_before
        }

    def run_all_benchmarks(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Run all benchmarks and return results.

        Args:
            sample_size: Optional override for number of samples to test

        Returns:
            Dictionary containing benchmark results
        """
        n_samples = sample_size or self.num_samples
        results = {
            "validation": self._benchmark_validation(n_samples),
            "rule_execution": self._benchmark_rules(n_samples),
            "model_provider": self._benchmark_model(n_samples),
        }

        if self.critic and self.guardrails_config:
            results["prompt_critic"] = self._benchmark_critic(n_samples)

        return results

    def _benchmark_validation(self, n_samples: int) -> Dict[str, Any]:
        """Benchmark validation performance."""
        latencies = []
        memory_usage = []

        for text in tqdm(self.test_data[:n_samples], desc="Validation"):
            stats = self._measure_memory(lambda: isinstance(text, str))
            latencies.append(stats["elapsed"])
            memory_usage.append(stats["memory_diff_mb"])

        return {
            "mean_latency": statistics.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "memory_diff_mb": statistics.mean(memory_usage),
            "throughput": n_samples / sum(latencies)
        }

    def _benchmark_rules(self, n_samples: int) -> Dict[str, Any]:
        """Benchmark rule execution performance."""
        latencies = []
        memory_usage = []

        for text in tqdm(self.test_data[:n_samples], desc="Rules"):
            stats = self._measure_memory(
                lambda: [rule.validate(text) for rule in self.rules.values()]
            )
            latencies.append(stats["elapsed"])
            memory_usage.append(stats["memory_diff_mb"])

        return {
            "mean_latency": statistics.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "memory_diff_mb": statistics.mean(memory_usage),
            "throughput": n_samples / sum(latencies)
        }

    def _benchmark_model(self, n_samples: int) -> Dict[str, Any]:
        """Benchmark model provider performance."""
        latencies = []
        memory_usage = []

        for text in tqdm(self.test_data[:n_samples], desc="Model"):
            prompt = f"Summarize this text: {text[:100]}..."
            stats = self._measure_memory(
                lambda: self.model.generate(prompt=prompt)
            )
            latencies.append(stats["elapsed"])
            memory_usage.append(stats["memory_diff_mb"])

        return {
            "mean_latency": statistics.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "memory_diff_mb": statistics.mean(memory_usage),
            "throughput": n_samples / sum(latencies)
        }

    def _benchmark_critic(self, n_samples: int) -> Dict[str, Any]:
        """Benchmark prompt critic performance with Guardrails."""
        if not self.critic or not self.guardrails_config:
            logger.warning("No Guardrails config provided, skipping prompt critic benchmark")
            return {
                "mean_latency": 0,
                "p95_latency": 0,
                "p99_latency": 0,
                "memory_diff_mb": 0,
                "throughput": 0,
                "violations": 0
            }

        latencies = []
        memory_usage = []
        violations = []

        for text in tqdm(self.test_data[:n_samples], desc="Critic"):
            stats = self._measure_memory(
                lambda: self.critic.critique(text)
            )
            latencies.append(stats["elapsed"])
            memory_usage.append(stats["memory_diff_mb"])
            violations.append(isinstance(stats["result"], RuleResult))

        return {
            "mean_latency": statistics.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "memory_diff_mb": statistics.mean(memory_usage),
            "throughput": n_samples / sum(latencies),
            "violations": sum(violations) / len(violations)
        }


def print_benchmark_results(results: Dict[str, Any]) -> None:
    """Print benchmark results in a readable format."""
    print("\nBenchmark Results:")
    print("=" * 80)

    sections = {
        "VALIDATION": "validation",
        "RULE_EXECUTION": "rule_execution",
        "MODEL_PROVIDER": "model_provider",
        "PROMPT_CRITIC": "prompt_critic"
    }

    for title, key in sections.items():
        if key not in results:
            continue

        stats = results[key]
        print(f"\n{title} Performance:")
        print("-" * 40)

        print("Latency (ms):")
        print(f"  Mean: {stats['mean_latency']*1000:.2f}")
        print(f"  P95: {stats['p95_latency']*1000:.2f}")
        print(f"  P99: {stats['p99_latency']*1000:.2f}")

        print("\nMemory Usage (MB):")
        print(f"  Mean: {stats['memory_diff_mb']:.2f}")

        if "violations" in stats:
            print("\nViolations:")
            print(f"  Rate: {stats['violations']*100:.1f}%")