"""
Benchmarking suite for Sifaka's core components.

This module provides tools to measure:
1. Validation performance
2. Rule execution performance
3. Resource usage and memory efficiency
4. Reflector performance with Anthropic
5. Prompt critic performance with Guardrails
"""

import gc
import statistics
import time
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
from tqdm import tqdm

from sifaka.validation import Validator, ValidationResult
from sifaka.rules import Rule, RuleResult
from sifaka.rules.base import BaseValidator
from sifaka.utils.logging import get_logger
from sifaka.models.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig, create_prompt_critic
from sifaka.rules.adapters.guardrails_adapter import create_guardrails_rule

# Import Guardrails components
try:
    from guardrails.validator_base import Validator as GuardrailsValidator, register_validator
    from guardrails.classes import ValidationResult as GuardrailsValidationResult, PassResult, FailResult

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    print("⚠️ Guardrails is not installed. Please install it with 'pip install guardrails-ai'")

logger = get_logger(__name__)


class LengthValidator(BaseValidator[str]):
    """Validator for text length."""

    def __init__(self, min_length: int = 0, max_length: int = 1000):
        """Initialize the validator."""
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length

    @property
    def validation_type(self) -> type:
        """Get the type of input this validator can validate."""
        return str

    def validate(self, text: str) -> RuleResult:
        """Validate the text length."""
        length = len(text)
        passed = self.min_length <= length <= self.max_length
        message = (
            f"Text length {length} is outside allowed range "
            f"[{self.min_length}, {self.max_length}]"
        )
        return RuleResult(passed=passed, message=message)


class WordCountValidator(BaseValidator[str]):
    """Validator for word count."""

    def __init__(self, min_words: int = 0, max_words: int = 100):
        """Initialize the validator."""
        super().__init__()
        self.min_words = min_words
        self.max_words = max_words

    @property
    def validation_type(self) -> type:
        """Get the type of input this validator can validate."""
        return str

    def validate(self, text: str) -> RuleResult:
        """Validate the word count."""
        word_count = len(text.split())
        passed = self.min_words <= word_count <= self.max_words
        message = (
            f"Word count {word_count} is outside allowed range "
            f"[{self.min_words}, {self.max_words}]"
        )
        return RuleResult(passed=passed, message=message)


class LengthRule(Rule):
    """Rule to validate text length."""

    def __init__(self, min_length: int = 0, max_length: int = 1000):
        """Initialize the rule."""
        self.min_length = min_length
        self.max_length = max_length
        super().__init__(name="length_rule", description="Validates text length")

    def _create_default_validator(self) -> BaseValidator[str]:
        """Create the default validator for this rule."""
        return LengthValidator(min_length=self.min_length, max_length=self.max_length)


class WordCountRule(Rule):
    """Rule to validate word count."""

    def __init__(self, min_words: int = 0, max_words: int = 100):
        """Initialize the rule."""
        self.min_words = min_words
        self.max_words = max_words
        super().__init__(name="word_count_rule", description="Validates word count")

    def _create_default_validator(self) -> BaseValidator[str]:
        """Create the default validator for this rule."""
        return WordCountValidator(min_words=self.min_words, max_words=self.max_words)


@register_validator(name="benchmark_validator", data_type="string")
class BenchmarkGuardrailsValidator(GuardrailsValidator):
    """Validator for benchmarking Guardrails."""

    rail_alias = "benchmark_validator"

    def __init__(self, on_fail="exception"):
        """Initialize the validator."""
        super().__init__(on_fail=on_fail)

    def _validate(self, value, metadata):
        """Simple validation for benchmarking."""
        if len(value) > 0:
            return PassResult(actual_value=value, validated_value=value)
        else:
            return FailResult(
                actual_value=value,
                error_message="Value must not be empty",
            )


class SifakaBenchmark:
    """Benchmark suite for Sifaka's core components."""

    def __init__(
        self,
        num_samples: int = 1000,
        warm_up_rounds: int = 3,
        text_length: int = 100,
        api_key: Optional[str] = None,
        guardrails_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the benchmark suite.

        Args:
            num_samples: Number of text samples to use
            warm_up_rounds: Number of warm-up rounds before measurement
            text_length: Length of generated text samples
            api_key: Optional API key for Anthropic integration
            guardrails_config: Optional configuration for Guardrails provider
        """
        self.num_samples = num_samples
        self.warm_up_rounds = warm_up_rounds
        self.text_length = text_length
        self.samples = self._generate_test_data()
        self.validator = self._init_validator()
        self.model_provider = self._init_model_provider(api_key)
        self.prompt_critic = self._init_prompt_critic(guardrails_config)
        self._warm_up()

    def _generate_test_data(self) -> List[str]:
        """Generate test data samples."""
        samples = []
        for _ in range(self.num_samples):
            # Generate sample text of specified length
            sample = " ".join(["test"] * (self.text_length // 5))
            samples.append(sample)
        return samples

    def _init_validator(self) -> Validator[str]:
        """Initialize a validator with test rules."""
        rules: List[Rule] = [
            LengthRule(min_length=10, max_length=1000),
            WordCountRule(min_words=2, max_words=200),
        ]
        return Validator(rules)

    def _init_model_provider(self, api_key: Optional[str]) -> Optional[AnthropicProvider]:
        """Initialize the Anthropic model provider if API key is provided."""
        if api_key:
            return AnthropicProvider(
                model_name="claude-3-sonnet-20240229",
                config=ModelConfig(
                    api_key=api_key,
                    temperature=0.7,
                    max_tokens=1000,
                ),
            )
        return None

    def _init_prompt_critic(self, guardrails_config: Optional[Dict[str, Any]]) -> Optional[PromptCritic]:
        """Initialize the Prompt critic with Guardrails if config is provided."""
        if guardrails_config and GUARDRAILS_AVAILABLE and self.model_provider:
            # Create a Guardrails validator
            guardrails_validator = BenchmarkGuardrailsValidator(on_fail="exception")

            # Create a Sifaka rule using the Guardrails validator
            guardrails_rule = create_guardrails_rule(
                guardrails_validator=guardrails_validator,
                rule_id="benchmark_rule",
            )

            # Create a critic with the model provider
            return create_prompt_critic(
                model=self.model_provider,
                name="benchmark_critic",
                description="Critic for benchmarking",
                system_prompt="You are a helpful editor who ensures text meets the validation requirements.",
                temperature=0.7,
                max_tokens=1000,
                min_confidence=0.7,
            )
        return None

    def _warm_up(self) -> None:
        """Run warm-up rounds to stabilize performance."""
        logger.info("Running warm-up rounds...")
        for _ in range(self.warm_up_rounds):
            for sample in self.samples[:10]:
                self.validator.validate(sample)
                if self.model_provider:
                    self.model_provider.generate(sample)
                if self.prompt_critic:
                    self.prompt_critic.critique(sample)
        gc.collect()

    def _measure_memory(self, func: callable, *args, **kwargs) -> Dict[str, float]:
        """Measure memory usage of a function call."""
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = end_memory - start_memory

        return {
            "start_memory_mb": start_memory,
            "end_memory_mb": end_memory,
            "memory_used_mb": memory_used,
        }

    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
            }
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": np.percentile(values, 95),
        }

    def benchmark_validation(self, sample_size: int = 100) -> Dict[str, Any]:
        """Benchmark validation performance."""
        logger.info("Benchmarking validation performance...")
        latencies = []
        memory_usage = []

        for sample in tqdm(self.samples[:sample_size]):
            # Measure latency
            start_time = time.time()
            self.validator.validate(sample)
            end_time = time.time()
            latencies.append(end_time - start_time)

            # Measure memory
            memory = self._measure_memory(self.validator.validate, sample)
            memory_usage.append(memory["memory_used_mb"])

        return {
            "latency_ms": {
                k: v * 1000 for k, v in self._calculate_statistics(latencies).items()
            },
            "memory_mb": self._calculate_statistics(memory_usage),
        }

    def benchmark_rule_execution(self, sample_size: int = 100) -> Dict[str, Any]:
        """Benchmark individual rule execution performance."""
        logger.info("Benchmarking rule execution performance...")
        latencies = []
        memory_usage = []

        for sample in tqdm(self.samples[:sample_size]):
            for rule in self.validator.rules:
                # Measure latency
                start_time = time.time()
                rule.validate(sample)
                end_time = time.time()
                latencies.append(end_time - start_time)

                # Measure memory
                memory = self._measure_memory(rule.validate, sample)
                memory_usage.append(memory["memory_used_mb"])

        return {
            "latency_ms": {
                k: v * 1000 for k, v in self._calculate_statistics(latencies).items()
            },
            "memory_mb": self._calculate_statistics(memory_usage),
        }

    def benchmark_model_provider(self, sample_size: int = 100) -> Dict[str, Any]:
        """Benchmark Anthropic model provider performance."""
        if not self.model_provider:
            logger.warning("No API key provided, skipping model provider benchmark")
            return {
                "latency_ms": {"mean": 0.0, "median": 0.0, "p95": 0.0},
                "memory_mb": {"mean": 0.0, "median": 0.0, "p95": 0.0},
            }

        logger.info("Benchmarking model provider performance...")
        latencies = []
        memory_usage = []

        for sample in tqdm(self.samples[:sample_size]):
            # Measure latency
            start_time = time.time()
            self.model_provider.generate(sample)
            end_time = time.time()
            latencies.append(end_time - start_time)

            # Measure memory
            memory = self._measure_memory(self.model_provider.generate, sample)
            memory_usage.append(memory["memory_used_mb"])

        return {
            "latency_ms": {
                k: v * 1000 for k, v in self._calculate_statistics(latencies).items()
            },
            "memory_mb": self._calculate_statistics(memory_usage),
        }

    def benchmark_prompt_critic(self, sample_size: int = 100) -> Dict[str, Any]:
        """Benchmark Prompt critic performance with Guardrails."""
        if not self.prompt_critic:
            logger.warning("No Guardrails config provided, skipping prompt critic benchmark")
            return {
                "latency_ms": {"mean": 0.0, "median": 0.0, "p95": 0.0},
                "memory_mb": {"mean": 0.0, "median": 0.0, "p95": 0.0},
                "violations": {"mean": 0.0, "median": 0.0, "p95": 0.0},
            }

        logger.info("Benchmarking prompt critic performance...")
        latencies = []
        memory_usage = []
        violations = []

        for sample in tqdm(self.samples[:sample_size]):
            # Measure latency
            start_time = time.time()
            result = self.prompt_critic.critique(sample)
            end_time = time.time()
            latencies.append(end_time - start_time)

            # Measure memory
            memory = self._measure_memory(self.prompt_critic.critique, sample)
            memory_usage.append(memory["memory_used_mb"])

            # Count violations
            violations.append(len(result.violations))

        return {
            "latency_ms": {
                k: v * 1000 for k, v in self._calculate_statistics(latencies).items()
            },
            "memory_mb": self._calculate_statistics(memory_usage),
            "violations": self._calculate_statistics(violations),
        }

    def run_all_benchmarks(self, sample_size: int = 100) -> Dict[str, Any]:
        """Run all benchmarks and return results."""
        logger.info("Starting comprehensive benchmark suite...")

        results = {
            "validation": self.benchmark_validation(sample_size),
            "rule_execution": self.benchmark_rule_execution(sample_size),
            "model_provider": self.benchmark_model_provider(sample_size),
            "prompt_critic": self.benchmark_prompt_critic(sample_size),
        }

        return results


def print_benchmark_results(results: Dict[str, Any]) -> None:
    """Print benchmark results in a readable format."""
    print("\nBenchmark Results:")
    print("=" * 80)

    for component, metrics in results.items():
        print(f"\n{component.upper()} Performance:")
        print("-" * 40)

        print("Latency (ms):")
        print(f"  Mean: {metrics['latency_ms']['mean']:.2f}")
        print(f"  Median: {metrics['latency_ms']['median']:.2f}")
        print(f"  P95: {metrics['latency_ms']['p95']:.2f}")

        print("\nMemory Usage (MB):")
        print(f"  Mean: {metrics['memory_mb']['mean']:.2f}")
        print(f"  Median: {metrics['memory_mb']['median']:.2f}")
        print(f"  P95: {metrics['memory_mb']['p95']:.2f}")

        if "violations" in metrics:
            print("\nViolations:")
            print(f"  Mean: {metrics['violations']['mean']:.2f}")
            print(f"  Median: {metrics['violations']['median']:.2f}")
            print(f"  P95: {metrics['violations']['p95']:.2f}")