#!/usr/bin/env python3
"""Multi-Critic Ensemble Demo with Mock Model.

This example demonstrates:
- Multiple critics working together in ensemble
- Different critic types and specializations
- Mock model for reliable testing
- Comprehensive improvement through diverse feedback

This example shows how different critics can provide specialized feedback
to improve content from multiple perspectives simultaneously.
"""

from sifaka.core.chain import Chain
from sifaka.models.base import MockModel
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.critics.prompt import PromptCritic
from sifaka.validators.base import LengthValidator
from sifaka.utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)


def create_critic_ensemble(model):
    """Create an ensemble of different critics for comprehensive feedback."""

    critics = []

    # 1. Reflexion Critic - For iterative improvement through reflection
    reflexion_critic = ReflexionCritic(
        model=model,
        reflection_depth=2,
        improvement_focus="clarity_and_accuracy",
        name="Clarity Reflexion Critic",
    )
    critics.append(reflexion_critic)

    # 2. Constitutional Critic - For ethical and principled evaluation
    constitutional_principles = [
        "Provide accurate and factual information",
        "Be helpful and constructive in tone",
        "Avoid bias and present balanced perspectives",
        "Respect diverse viewpoints and experiences",
        "Promote learning and understanding",
    ]

    constitutional_critic = ConstitutionalCritic(
        model=model,
        principles=constitutional_principles,
        name="Educational Ethics Constitutional Critic",
    )
    critics.append(constitutional_critic)

    # 3. Self-Refine Critic - For iterative self-improvement
    self_refine_critic = SelfRefineCritic(
        model=model,
        
        focus_areas=["completeness", "engagement"],
        name="Completeness Self-Refine Critic",
    )
    critics.append(self_refine_critic)

    # 4. Prompt Critic - For specialized domain feedback
    domain_prompt = """
    You are an expert educational content reviewer. Evaluate the following text for:

    1. EDUCATIONAL VALUE: Does it teach something meaningful?
    2. ACCESSIBILITY: Is it understandable to the target audience?
    3. ENGAGEMENT: Is it interesting and compelling?
    4. STRUCTURE: Is it well-organized and logical?
    5. COMPLETENESS: Does it cover the topic adequately?

    Text to evaluate: {text}

    Provide specific feedback on each criterion and suggest improvements.
    Rate overall quality: EXCELLENT / GOOD / NEEDS_IMPROVEMENT / POOR
    """

    prompt_critic = PromptCritic(
        model=model,
        critique_prompt=domain_prompt,
        improvement_prompt="Based on the educational content review, provide an improved version that addresses all identified issues while maintaining the core message.",
        name="Educational Content Prompt Critic",
    )
    critics.append(prompt_critic)

    return critics


def main():
    """Run the multi-critic ensemble demo."""

    logger.info("Creating multi-critic ensemble demo")

    # Create mock model with educational responses
    model = MockModel(
        model_name="Educational Content Model",
        responses=[
            "Learning is important for personal growth.",  # Basic response
            "Learning is a fundamental process that drives personal growth and development. It involves acquiring new knowledge, skills, and perspectives that help us adapt to changing circumstances and achieve our goals.",  # Improved response
            "Learning is a fundamental and transformative process that drives personal growth, intellectual development, and adaptive capacity. It involves the systematic acquisition of new knowledge, practical skills, and diverse perspectives that enable us to navigate complex challenges, solve problems creatively, and achieve meaningful goals while contributing positively to our communities and society at large.",  # Comprehensive response
            "Learning represents a fundamental and transformative process that drives personal growth, intellectual development, and adaptive capacity throughout our lives. It involves the systematic and intentional acquisition of new knowledge, practical skills, and diverse perspectives that enable us to navigate complex challenges, solve problems creatively, and achieve meaningful goals while contributing positively to our communities and society at large. Through continuous learning, we develop critical thinking abilities, emotional intelligence, and the resilience needed to thrive in an ever-changing world.",  # Final comprehensive response
        ],
    )

    # Create critic ensemble
    critics = create_critic_ensemble(model)

    # Create length validator
    length_validator = LengthValidator(
        min_length=100, max_length=1000
    )

    # Create the chain
    chain = Chain(
        model=model,
        prompt="Write a comprehensive explanation of why learning is important for personal development and how it contributes to success in life.",
        max_improvement_iterations=len(critics),  # Allow one iteration per critic
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,  # Always apply all critics
    )

    # Add validator
    chain.validate_with(length_validator)

    # Add all critics to the ensemble
    for critic in critics:
        chain.improve_with(critic)

    # Run the chain
    logger.info("Running chain with multi-critic ensemble...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 70)
    print("MULTI-CRITIC ENSEMBLE DEMO WITH MOCK MODEL")
    print("=" * 70)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nChain Execution Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Critics in Ensemble: {len(critics)}")

    # Show validation results
    if result.validation_results:
        print(f"\nValidation Results:")
        for validator_name, validation_result in result.validation_results.items():
            status = "✓ PASSED" if validation_result.passed else "✗ FAILED"
            print(f"  {validator_name}: {status}")

    # Show detailed critic feedback
    if result.critic_feedback:
        print(f"\nCritic Ensemble Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"\n  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.confidence}")

            if feedback.suggestions:
                # Extract quality rating if it's a prompt critic
                suggestions = feedback.suggestions
                if "EXCELLENT" in suggestions:
                    print(f"     Quality Rating: EXCELLENT")
                elif "GOOD" in suggestions:
                    print(f"     Quality Rating: GOOD")
                elif "NEEDS_IMPROVEMENT" in suggestions:
                    print(f"     Quality Rating: NEEDS_IMPROVEMENT")
                elif "POOR" in suggestions:
                    print(f"     Quality Rating: POOR")

                print(f"     Feedback: {suggestions[:250]}...")

            # Show critic-specific metrics if available
            if hasattr(feedback, "reflection_depth"):
                print(f"     Reflection Depth: {feedback.reflection_depth}")
            if hasattr(feedback, "principles_evaluated"):
                print(
                    f"     Constitutional Principles: {len(feedback.principles_evaluated)} evaluated"
                )

    # Show content evolution across iterations
    print(f"\nContent Evolution Through Critic Ensemble:")
    for i, historical_thought in enumerate(result.history, 1):
        print(f"\n  Iteration {i} ({len(historical_thought.summary)} characters):")
        print(f"    Preview: {historical_thought.summary[:120]}...")

        # Show which critic likely influenced this iteration
        if i <= len(critics):
            critic_name = critics[i - 1].name if i > 1 else "Initial Generation"
            print(f"    Influenced by: {critic_name}")

    # Analyze improvement metrics
    initial_length = len(result.history[0].text) if result.history else 0
    final_length = len(result.text)
    improvement_ratio = final_length / initial_length if initial_length > 0 else 1

    print(f"\nImprovement Analysis:")
    print(f"  Initial length: {initial_length} characters")
    print(f"  Final length: {final_length} characters")
    print(f"  Expansion ratio: {improvement_ratio:.2f}x")
    print(f"  Total iterations: {result.iteration}")

    # Show critic specializations
    print(f"\nCritic Specializations:")
    print(f"  ✓ Reflexion: Clarity and accuracy through reflection")
    print(f"  ✓ Constitutional: Ethical principles and balanced perspectives")
    print(f"  ✓ Self-Refine: Completeness and engagement")
    print(f"  ✓ Prompt: Educational value and structure")

    print(f"\nEnsemble Benefits:")
    print(f"  ✓ Diverse perspectives on content quality")
    print(f"  ✓ Comprehensive improvement coverage")
    print(f"  ✓ Specialized domain expertise")
    print(f"  ✓ Multi-dimensional feedback")
    print(f"  ✓ Iterative refinement process")

    print("\n" + "=" * 70)
    logger.info("Multi-critic ensemble demo completed successfully")


if __name__ == "__main__":
    main()
