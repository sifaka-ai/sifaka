#!/usr/bin/env python3
"""
HuggingFace Local + Remote Example for Sifaka

This example demonstrates using HuggingFace models in dual mode:
- Remote model for high-quality text generation via Inference Providers API
- Local model for fast criticism (with model caching)

Setup:
1. Get HuggingFace API token from https://huggingface.co/settings/tokens
2. Set environment variable: export HUGGINGFACE_API_TOKEN="hf_your_token_here"
3. Install dependencies: pip install sifaka[huggingface]

The example uses:
- Remote: HuggingFaceH4/zephyr-7b-beta (powerful generation via PRO subscription)
- Local: google/flan-t5-base (fast, cached criticism)

Features:
- Sequential execution without user input
- Model caching for performance
- Automatic device detection
"""

import os
from sifaka.core.thought import Thought, ValidationResult
from sifaka.core.chain import Chain
from sifaka.models.huggingface import create_huggingface_model
from sifaka.critics.prompt import PromptCritic
from sifaka.validators.base import LengthValidator
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


def run_huggingface_example():
    """Run the HuggingFace local + remote example."""

    # Check for API token
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not api_token:
        print("❌ HUGGINGFACE_API_TOKEN environment variable not set!")
        print("Please get your token from https://huggingface.co/settings/tokens")
        print("Then run: export HUGGINGFACE_API_TOKEN='hf_your_token_here'")
        return

    print("🤗 HuggingFace Local + Remote Example")
    print("=" * 50)

    try:
        # Create remote model for generation (high quality)
        print("🌐 Creating remote HuggingFace model for generation...")
        print("⚠️  NOTE: Remote models require HuggingFace PRO subscription!")
        print("   Visit https://huggingface.co/subscribe/pro to upgrade your account.")
        print("   Free tier has very limited model access via Inference Providers.")
        print()

        remote_model = create_huggingface_model(
            model_name="HuggingFaceH4/zephyr-7b-beta",  # Powerful remote model for generation
            use_inference_api=True,
            api_token=api_token,
            max_tokens=150,  # Short responses for speed
            temperature=0.7,  # Moderate temperature
        )
        print(f"✅ Remote model ready: {remote_model.model_name}")

        # Create local critic using PromptCritic (fast, local)
        print("📥 Creating local HuggingFace critic...")
        local_critic_model = create_huggingface_model(
            model_name="google/flan-t5-base",  # Fast local model for criticism
            use_inference_api=False,
            device="auto",  # Auto-detect best device
            quantization=None,  # Disable quantization to avoid issues
            max_tokens=150,  # Shorter responses for criticism
            temperature=0.3,  # Lower temperature for focused criticism
        )
        local_critic = PromptCritic(model=local_critic_model)
        print(f"✅ Local critic ready: {local_critic.model.model_name}")

        # Create validator
        validator = LengthValidator(min_length=30)

        # Create chain with remote generation + local criticism (reduced iterations for speed)
        chain = (
            Chain(model=remote_model, max_improvement_iterations=1, always_apply_critics=True)
            .validate_with(validator)
            .improve_with(local_critic)
        )

        # Test prompts
        test_prompts = [
            "Write a short story about a robot learning to paint.",
            "Explain quantum computing in simple terms.",
            "Describe a perfect day in the mountains.",
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🎯 Example {i}: {prompt}")
            print("-" * 60)

            # Create thought
            thought = Thought(
                prompt=prompt, system_prompt="You are a creative and helpful assistant."
            )

            # Run the chain
            print("🔄 Running chain with remote generation + local criticism...")
            final_thought = chain.with_prompt(prompt).run()

            # Display results
            print(f"\n📝 Final Generated Text:")
            print(f"'{final_thought.text}'")

            print(f"\n🔍 Validation Results:")
            for validator_name, result in final_thought.validation_results.items():
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"  {status} {validator_name}: {result.message or 'Valid'}")

            print(f"\n💭 Critic Feedback:")
            for feedback in final_thought.critic_feedback:
                print(f"  🤖 {feedback.critic_name}:")
                print(f"     {feedback.feedback}")

            print(f"\n📊 Chain Statistics:")
            print(f"  Iterations: {final_thought.iteration}")
            print(f"  History length: {len(final_thought.history)}")

            # Continue automatically to next example

        print("\n🎉 HuggingFace example completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("  ✅ Remote model for high-quality generation")
        print("  ✅ Local model for fast criticism")
        print("  ✅ Device auto-detection (CPU/GPU/MPS)")
        print("  ✅ Memory-efficient model caching")
        print("  ✅ Dual-mode HuggingFace integration")

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install with: pip install 'sifaka[huggingface]'")
    except Exception as e:
        print(f"❌ Error running example: {e}")
        logger.error(f"HuggingFace example failed: {e}")


def main():
    """Main entry point."""
    print("🚀 Starting HuggingFace Local + Remote Example...")

    # Check if HuggingFace dependencies are available
    try:
        from transformers import AutoTokenizer
        from huggingface_hub import InferenceClient

        print("✅ HuggingFace dependencies available")
    except ImportError:
        print("❌ HuggingFace dependencies not installed")
        print("Please install with: pip install 'sifaka[huggingface]'")
        return

    # Run the example
    run_huggingface_example()


if __name__ == "__main__":
    main()
