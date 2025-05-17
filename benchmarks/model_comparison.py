#!/usr/bin/env python
"""
Model Comparison Benchmark

This script benchmarks different model providers and critics in Sifaka.
It measures performance metrics like:
- Generation quality
- Validation accuracy
- Improvement effectiveness
- Runtime performance
"""

import sys
import os
import time
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add the project root to the path so we can import sifaka
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sifaka.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators import length, factual_accuracy
from sifaka.critics.reflexion import create_reflexion_critic
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.critics.self_rag import create_self_rag_critic
from sifaka.critics.self_refine import create_self_refine_critic
from sifaka.critics.n_critics import create_n_critics_critic


# Test prompts for benchmarking
TEST_PROMPTS = [
    {
        "name": "quantum_computing",
        "prompt": "Write a short explanation of quantum computing.",
        "category": "technical",
    },
    {
        "name": "climate_change",
        "prompt": "Explain the causes and effects of climate change.",
        "category": "scientific",
    },
    {
        "name": "shakespeare",
        "prompt": "Write a short poem in the style of Shakespeare.",
        "category": "creative",
    },
    {
        "name": "recipe",
        "prompt": "Write a recipe for chocolate chip cookies.",
        "category": "instructional",
    },
    {
        "name": "history",
        "prompt": "Explain the causes of World War I.",
        "category": "historical",
    },
]

# Model configurations to benchmark
MODEL_CONFIGS = [
    {
        "name": "openai_gpt35",
        "model": "openai:gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 500,
    },
    {
        "name": "openai_gpt4",
        "model": "openai:gpt-4",
        "temperature": 0.7,
        "max_tokens": 500,
    },
    {
        "name": "anthropic_claude3",
        "model": "anthropic:claude-3-sonnet",
        "temperature": 0.7,
        "max_tokens": 500,
    },
]

# Critic configurations to benchmark
CRITIC_CONFIGS = [
    {
        "name": "no_critic",
        "type": None,
    },
    {
        "name": "n_critics",
        "type": "n_critics",
        "max_iterations": 2,
    },
    {
        "name": "reflexion",
        "type": "reflexion",
        "max_iterations": 2,
    },
    {
        "name": "constitutional",
        "type": "constitutional",
        "constitution": [
            "Your output should be factually accurate and not misleading.",
            "Your output should be helpful, harmless, and honest.",
            "Your output should be clear, concise, and well-organized.",
        ],
    },
    {
        "name": "self_refine",
        "type": "self_refine",
        "max_iterations": 2,
    },
]


def create_benchmark_chain(
    model_config: Dict[str, Any],
    critic_config: Dict[str, Any],
    prompt: str,
) -> Chain:
    """Create a chain for benchmarking.

    Args:
        model_config: Configuration for the model
        critic_config: Configuration for the critic
        prompt: The prompt to use

    Returns:
        A configured Chain instance
    """
    # Create a chain
    chain = Chain()

    # Configure the chain
    chain.with_model(model_config["model"])
    chain.with_prompt(prompt)

    # Add a length validator
    chain.validate_with(length(min_words=50, max_words=500))

    # Add a critic if specified
    if critic_config["type"]:
        # Create a model instance for the critic
        provider, model_id = model_config["model"].split(":", 1)
        model = create_model(provider, model_id)

        if critic_config["type"] == "n_critics":
            critic = create_n_critics_critic(
                model=model,
                temperature=model_config.get("temperature", 0.7),
                max_iterations=critic_config.get("max_iterations", 2),
            )
            chain.improve_with(critic)

        elif critic_config["type"] == "reflexion":
            critic = create_reflexion_critic(
                model=model,
                temperature=model_config.get("temperature", 0.7),
                max_iterations=critic_config.get("max_iterations", 2),
            )
            chain.improve_with(critic)

        elif critic_config["type"] == "constitutional":
            critic = create_constitutional_critic(
                model=model,
                constitution=critic_config.get("constitution", []),
                temperature=model_config.get("temperature", 0.7),
            )
            chain.improve_with(critic)

        elif critic_config["type"] == "self_rag":
            critic = create_self_rag_critic(
                model=model,
                temperature=model_config.get("temperature", 0.7),
            )
            chain.improve_with(critic)

        elif critic_config["type"] == "self_refine":
            critic = create_self_refine_critic(
                model=model,
                temperature=model_config.get("temperature", 0.7),
                max_iterations=critic_config.get("max_iterations", 2),
            )
            chain.improve_with(critic)

    # Set options
    chain.with_options(
        temperature=model_config.get("temperature", 0.7),
        max_tokens=model_config.get("max_tokens", 500),
    )

    return chain


def run_benchmark(
    model_configs: List[Dict[str, Any]],
    critic_configs: List[Dict[str, Any]],
    test_prompts: List[Dict[str, Any]],
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the benchmark.

    Args:
        model_configs: List of model configurations
        critic_configs: List of critic configurations
        test_prompts: List of test prompts
        output_file: Optional file to write results to

    Returns:
        A dictionary with benchmark results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_configs": model_configs,
        "critic_configs": critic_configs,
        "test_prompts": test_prompts,
        "results": [],
    }

    # Run benchmarks for each combination
    for model_config in model_configs:
        for critic_config in critic_configs:
            for prompt_config in test_prompts:
                print(
                    f"Benchmarking {model_config['name']} with {critic_config['name']} on {prompt_config['name']}..."
                )

                try:
                    # Create the chain
                    chain = create_benchmark_chain(
                        model_config=model_config,
                        critic_config=critic_config,
                        prompt=prompt_config["prompt"],
                    )

                    # Run the chain and measure performance
                    start_time = time.time()
                    result = chain.run()
                    elapsed_time = time.time() - start_time

                    # Calculate metrics
                    word_count = len(result.text.split())
                    validation_passed = result.passed
                    improvement_made = any(r.changes_made for r in result.improvement_results)

                    # Store the result
                    benchmark_result = {
                        "model": model_config["name"],
                        "critic": critic_config["name"],
                        "prompt": prompt_config["name"],
                        "category": prompt_config["category"],
                        "elapsed_time": elapsed_time,
                        "word_count": word_count,
                        "validation_passed": validation_passed,
                        "improvement_made": improvement_made,
                        "validation_results": [
                            {
                                "passed": r.passed,
                                "message": r.message,
                            }
                            for r in result.validation_results
                        ],
                        "improvement_results": [
                            {
                                "changes_made": r.changes_made,
                                "message": r.message,
                            }
                            for r in result.improvement_results
                        ],
                    }

                    results["results"].append(benchmark_result)

                    print(f"  Elapsed time: {elapsed_time:.2f}s")
                    print(f"  Word count: {word_count}")
                    print(f"  Validation passed: {validation_passed}")
                    print(f"  Improvement made: {improvement_made}")

                except Exception as e:
                    print(f"  Error: {str(e)}")

                    # Store the error
                    benchmark_result = {
                        "model": model_config["name"],
                        "critic": critic_config["name"],
                        "prompt": prompt_config["name"],
                        "category": prompt_config["category"],
                        "error": str(e),
                    }

                    results["results"].append(benchmark_result)

    # Write results to file if specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze benchmark results.

    Args:
        results: Benchmark results

    Returns:
        A dictionary with analysis results
    """
    analysis = {
        "models": {},
        "critics": {},
        "categories": {},
        "overall": {
            "total_runs": 0,
            "successful_runs": 0,
            "average_time": 0.0,
            "validation_pass_rate": 0.0,
            "improvement_rate": 0.0,
        },
    }

    # Initialize model, critic, and category stats
    for model_config in results["model_configs"]:
        analysis["models"][model_config["name"]] = {
            "total_runs": 0,
            "successful_runs": 0,
            "average_time": 0.0,
            "validation_pass_rate": 0.0,
            "improvement_rate": 0.0,
        }

    for critic_config in results["critic_configs"]:
        analysis["critics"][critic_config["name"]] = {
            "total_runs": 0,
            "successful_runs": 0,
            "average_time": 0.0,
            "validation_pass_rate": 0.0,
            "improvement_rate": 0.0,
        }

    categories = set(prompt["category"] for prompt in results["test_prompts"])
    for category in categories:
        analysis["categories"][category] = {
            "total_runs": 0,
            "successful_runs": 0,
            "average_time": 0.0,
            "validation_pass_rate": 0.0,
            "improvement_rate": 0.0,
        }

    # Analyze results
    total_time = 0.0
    total_validation_passes = 0
    total_improvements = 0

    for result in results["results"]:
        # Skip results with errors
        if "error" in result:
            continue

        # Update overall stats
        analysis["overall"]["total_runs"] += 1
        analysis["overall"]["successful_runs"] += 1
        total_time += result["elapsed_time"]
        total_validation_passes += 1 if result["validation_passed"] else 0
        total_improvements += 1 if result["improvement_made"] else 0

        # Update model stats
        model = result["model"]
        analysis["models"][model]["total_runs"] += 1
        analysis["models"][model]["successful_runs"] += 1
        analysis["models"][model]["average_time"] += result["elapsed_time"]
        analysis["models"][model]["validation_pass_rate"] += 1 if result["validation_passed"] else 0
        analysis["models"][model]["improvement_rate"] += 1 if result["improvement_made"] else 0

        # Update critic stats
        critic = result["critic"]
        analysis["critics"][critic]["total_runs"] += 1
        analysis["critics"][critic]["successful_runs"] += 1
        analysis["critics"][critic]["average_time"] += result["elapsed_time"]
        analysis["critics"][critic]["validation_pass_rate"] += (
            1 if result["validation_passed"] else 0
        )
        analysis["critics"][critic]["improvement_rate"] += 1 if result["improvement_made"] else 0

        # Update category stats
        category = result["category"]
        analysis["categories"][category]["total_runs"] += 1
        analysis["categories"][category]["successful_runs"] += 1
        analysis["categories"][category]["average_time"] += result["elapsed_time"]
        analysis["categories"][category]["validation_pass_rate"] += (
            1 if result["validation_passed"] else 0
        )
        analysis["categories"][category]["improvement_rate"] += (
            1 if result["improvement_made"] else 0
        )

    # Calculate averages
    if analysis["overall"]["successful_runs"] > 0:
        analysis["overall"]["average_time"] = total_time / analysis["overall"]["successful_runs"]
        analysis["overall"]["validation_pass_rate"] = (
            total_validation_passes / analysis["overall"]["successful_runs"]
        )
        analysis["overall"]["improvement_rate"] = (
            total_improvements / analysis["overall"]["successful_runs"]
        )

    for model in analysis["models"]:
        if analysis["models"][model]["successful_runs"] > 0:
            analysis["models"][model]["average_time"] /= analysis["models"][model][
                "successful_runs"
            ]
            analysis["models"][model]["validation_pass_rate"] /= analysis["models"][model][
                "successful_runs"
            ]
            analysis["models"][model]["improvement_rate"] /= analysis["models"][model][
                "successful_runs"
            ]

    for critic in analysis["critics"]:
        if analysis["critics"][critic]["successful_runs"] > 0:
            analysis["critics"][critic]["average_time"] /= analysis["critics"][critic][
                "successful_runs"
            ]
            analysis["critics"][critic]["validation_pass_rate"] /= analysis["critics"][critic][
                "successful_runs"
            ]
            analysis["critics"][critic]["improvement_rate"] /= analysis["critics"][critic][
                "successful_runs"
            ]

    for category in analysis["categories"]:
        if analysis["categories"][category]["successful_runs"] > 0:
            analysis["categories"][category]["average_time"] /= analysis["categories"][category][
                "successful_runs"
            ]
            analysis["categories"][category]["validation_pass_rate"] /= analysis["categories"][
                category
            ]["successful_runs"]
            analysis["categories"][category]["improvement_rate"] /= analysis["categories"][
                category
            ]["successful_runs"]

    return analysis


def main():
    """Run the benchmark script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark Sifaka models and critics")
    parser.add_argument("--output", help="Output file to write results to")
    parser.add_argument("--models", nargs="+", help="Models to benchmark (default: all)")
    parser.add_argument("--critics", nargs="+", help="Critics to benchmark (default: all)")
    parser.add_argument("--prompts", nargs="+", help="Prompts to benchmark (default: all)")
    args = parser.parse_args()

    # Filter model configs if specified
    model_configs = MODEL_CONFIGS
    if args.models:
        model_configs = [config for config in MODEL_CONFIGS if config["name"] in args.models]

    # Filter critic configs if specified
    critic_configs = CRITIC_CONFIGS
    if args.critics:
        critic_configs = [config for config in CRITIC_CONFIGS if config["name"] in args.critics]

    # Filter test prompts if specified
    test_prompts = TEST_PROMPTS
    if args.prompts:
        test_prompts = [config for config in TEST_PROMPTS if config["name"] in args.prompts]

    # Generate output file name if not specified
    output_file = args.output
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.json"

    # Run the benchmark
    print(
        f"Running benchmark with {len(model_configs)} models, {len(critic_configs)} critics, and {len(test_prompts)} prompts..."
    )
    results = run_benchmark(
        model_configs=model_configs,
        critic_configs=critic_configs,
        test_prompts=test_prompts,
        output_file=output_file,
    )

    # Analyze the results
    analysis = analyze_results(results)

    # Print analysis
    print("\nBenchmark Analysis:")
    print(f"Total runs: {analysis['overall']['total_runs']}")
    print(f"Successful runs: {analysis['overall']['successful_runs']}")
    print(f"Average time: {analysis['overall']['average_time']:.2f}s")
    print(f"Validation pass rate: {analysis['overall']['validation_pass_rate']:.2%}")
    print(f"Improvement rate: {analysis['overall']['improvement_rate']:.2%}")

    print("\nModel Performance:")
    for model, stats in analysis["models"].items():
        print(f"  {model}:")
        print(f"    Average time: {stats['average_time']:.2f}s")
        print(f"    Validation pass rate: {stats['validation_pass_rate']:.2%}")
        print(f"    Improvement rate: {stats['improvement_rate']:.2%}")

    print("\nCritic Performance:")
    for critic, stats in analysis["critics"].items():
        print(f"  {critic}:")
        print(f"    Average time: {stats['average_time']:.2f}s")
        print(f"    Validation pass rate: {stats['validation_pass_rate']:.2%}")
        print(f"    Improvement rate: {stats['improvement_rate']:.2%}")

    print("\nCategory Performance:")
    for category, stats in analysis["categories"].items():
        print(f"  {category}:")
        print(f"    Average time: {stats['average_time']:.2f}s")
        print(f"    Validation pass rate: {stats['validation_pass_rate']:.2%}")
        print(f"    Improvement rate: {stats['improvement_rate']:.2%}")

    print(f"\nResults written to {output_file}")


if __name__ == "__main__":
    main()
