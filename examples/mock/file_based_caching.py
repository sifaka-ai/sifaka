#!/usr/bin/env python3
"""Thought Caching to File Example.

This example demonstrates:
- File-based thought persistence and caching
- Saving all thought iterations to disk
- Loading and resuming from cached thoughts
- Mock model for reliable demonstration

The chain will generate content about machine learning and save all thought
iterations to a JSON file for later analysis and recovery.
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.models.base import MockModel
from sifaka.storage.file import FileStorage
from sifaka.utils.logging import get_logger
from sifaka.validators.base import LengthValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def create_file_storage():
    """Create file-based storage for thought caching."""

    # Create storage directory if it doesn't exist
    storage_dir = "thought_cache"
    os.makedirs(storage_dir, exist_ok=True)

    # Create file storage with timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_file = os.path.join(storage_dir, f"thoughts_{timestamp}.json")

    file_storage = FileStorage(
        file_path=storage_file,
    )

    return file_storage, storage_file


def demonstrate_thought_loading(storage_file):
    """Demonstrate loading thoughts from file."""

    if not os.path.exists(storage_file):
        print("No cached thoughts file found.")
        return

    try:
        with open(storage_file, "r") as f:
            cached_data = json.load(f)

        print(f"\nCached Thoughts Analysis:")
        print(f"  File: {storage_file}")
        print(f"  Total entries: {len(cached_data)}")

        # Analyze cached thoughts
        for key, thought_data in cached_data.items():
            if isinstance(thought_data, dict) and "iteration" in thought_data:
                print(
                    f"  Thought {key}: Iteration {thought_data['iteration']}, "
                    f"Text length: {len(thought_data.get('text', ''))}"
                )

    except Exception as e:
        print(f"Error loading cached thoughts: {e}")


def main():
    """Run the File-based Thought Caching example."""

    logger.info("Creating file-based thought caching example")

    # Create file storage for thought caching
    file_storage, storage_file = create_file_storage()

    # Create mock model for reliable demonstration
    model = MockModel(
        model_name="ML Education Model",
        responses=[
            "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
            "Machine learning is a powerful subset of artificial intelligence that enables computers to automatically learn patterns and make intelligent decisions from data, without requiring explicit programming for each specific task or scenario.",
            "Machine learning represents a transformative subset of artificial intelligence that empowers computers to automatically discover patterns, learn from experience, and make intelligent decisions from data, eliminating the need for explicit programming of every possible scenario or task.",
        ],
    )

    # Create Reflexion critic for improvement
    critic = ReflexionCritic(model=model)

    # Create length validator
    length_validator = LengthValidator(min_length=100, max_length=500)

    # Create the chain with file storage
    chain = Chain(
        model=model,
        prompt="Explain what machine learning is and how it differs from traditional programming approaches.",
        storage=file_storage,  # File-based thought caching
        max_improvement_iterations=2,
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,
    )

    # Add validator and critic
    chain.validate_with(length_validator)
    chain.improve_with(critic)

    # Run the chain
    logger.info("Running chain with file-based thought caching...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 80)
    print("FILE-BASED THOUGHT CACHING EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nProcessing Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Storage File: {storage_file}")

    # Show validation results
    if result.validation_results:
        print(f"\nValidation Results:")
        for validator_name, validation_result in result.validation_results.items():
            status = "✓ PASSED" if validation_result.passed else "✗ FAILED"
            print(f"  {validator_name}: {status}")

    # Show critic feedback
    if result.critic_feedback:
        print(f"\nReflexion Critic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Suggestions: {feedback.suggestions[:200]}...")

    # Show thought history
    print(f"\nThought History ({len(result.history)} iterations):")
    for i, historical_thought in enumerate(result.history, 1):
        print(f"  Iteration {historical_thought.iteration}: {historical_thought.summary}")
        if i <= 3:  # Show first 3 iterations in detail
            print(f"    Summary: {historical_thought.summary}")

    # Demonstrate file storage features
    print(f"\nFile Storage Features:")
    print(f"  Auto-save: Enabled")
    print(f"  Backup count: 5 files")
    print(f"  Storage format: JSON")
    print(f"  File location: {storage_file}")

    # Demonstrate loading cached thoughts
    demonstrate_thought_loading(storage_file)

    # Show file size and content summary
    if os.path.exists(storage_file):
        file_size = os.path.getsize(storage_file)
        print(f"\nFile Statistics:")
        print(f"  File size: {file_size} bytes")
        print(f"  Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Demonstrate thought recovery capability
    print(f"\nThought Recovery Capability:")
    print(f"  All {result.iteration} iterations saved to file")
    print(f"  Can be loaded for analysis or resumption")
    print(f"  Includes full context and metadata")

    print(f"\nUse Cases for File-based Caching:")
    print(f"  - Debugging chain execution")
    print(f"  - Analyzing improvement patterns")
    print(f"  - Resuming interrupted processes")
    print(f"  - Audit trails for content generation")
    print(f"  - Offline analysis and research")

    print("\n" + "=" * 80)
    logger.info("File-based thought caching example completed successfully")
    print(f"\nCached thoughts saved to: {storage_file}")


if __name__ == "__main__":
    main()
