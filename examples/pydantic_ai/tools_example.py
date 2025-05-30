#!/usr/bin/env python3
"""PydanticAI tools integration example.

This example demonstrates how to use PydanticAI agents with tools alongside
Sifaka's validation and criticism framework, including tool call logging.
"""

import os
import signal
import sys
import time

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Run the PydanticAI tools integration example."""
    try:
        # Import PydanticAI (will fail gracefully if not installed)
        from pydantic_ai import Agent

        # Import Sifaka components
        from sifaka.agents import create_pydantic_chain
        from sifaka.critics import SelfRefineCritic
        from sifaka.models import create_model
        from sifaka.storage import FileStorage
        from sifaka.validators import LengthValidator, RegexValidator

        print("🛠️  PydanticAI Tools + Sifaka Integration Example")
        print("=" * 55)

        # Create a PydanticAI agent with tools
        print("\n1. Creating PydanticAI agent with tools...")
        agent = Agent(
            "openai:gpt-4o-mini",
            system_prompt="""You are a helpful research assistant. You have access to tools
            that can help you gather information and validate content. Use them when appropriate
            to improve your responses.""",
        )

        # Add tools to the agent using the correct decorators
        @agent.tool_plain
        def search_facts(query: str) -> str:
            """Search for factual information about a topic."""
            # Enhanced fact database with more AI history
            fact_database = {
                "ai": "Artificial Intelligence was first coined as a term in 1956 at the Dartmouth Conference.",
                "artificial intelligence": "AI was founded as an academic discipline in 1956.",
                "dartmouth": "The Dartmouth Conference in 1956 is considered the birth of AI as a field.",
                "1956": "1956: The Dartmouth Conference established AI as a research field.",
                "turing": "Alan Turing proposed the Turing Test in 1950 as a test of machine intelligence.",
                "1950": "1950: Alan Turing published 'Computing Machinery and Intelligence'.",
                "expert systems": "Expert systems became popular in the 1980s, like MYCIN for medical diagnosis.",
                "1980s": "1980s: Expert systems and knowledge-based AI gained commercial success.",
                "neural networks": "Neural networks were inspired by biological neural networks in the brain.",
                "deep learning": "Deep learning gained popularity in the 2010s with advances in GPU computing.",
                "2010s": "2010s: Deep learning revolution led by advances in neural networks and big data.",
                "machine learning": "Machine learning algorithms can be traced back to the 1940s and 1950s.",
                "history": "AI history spans from 1950s theoretical foundations to modern deep learning applications.",
            }

            # Enhanced keyword matching
            query_lower = query.lower()
            for key, fact in fact_database.items():
                if key in query_lower:
                    return f"Historical fact: {fact}"

            return f"General AI fact: AI has evolved from 1950s theoretical concepts to today's practical applications in many fields."

        @agent.tool_plain
        def validate_content(text: str, criteria: str) -> str:
            """Validate content against specific criteria."""
            # Simple validation logic
            validations = {
                "length": f"Text length: {len(text)} characters",
                "factual": (
                    "Content appears to contain factual statements"
                    if any(
                        word in text.lower()
                        for word in ["first", "invented", "discovered", "year", "century"]
                    )
                    else "Content may need more factual backing"
                ),
                "engaging": (
                    "Content is engaging"
                    if any(
                        word in text.lower()
                        for word in ["amazing", "incredible", "fascinating", "remarkable"]
                    )
                    else "Content could be more engaging"
                ),
            }

            if criteria.lower() in validations:
                return validations[criteria.lower()]
            else:
                return f"Validation criteria '{criteria}' not recognized. Available: {', '.join(validations.keys())}"

        @agent.tool_plain
        def improve_text(text: str, aspect: str) -> str:
            """Suggest improvements for specific aspects of text."""
            suggestions = {
                "clarity": "Consider breaking down complex sentences and using simpler language.",
                "engagement": "Add more vivid descriptions, examples, or rhetorical questions.",
                "factual": "Include specific dates, numbers, or verifiable claims.",
                "structure": "Organize content with clear introduction, body, and conclusion.",
            }

            if aspect.lower() in suggestions:
                return f"Improvement suggestion for {aspect}: {suggestions[aspect.lower()]}"
            else:
                return f"Improvement aspect '{aspect}' not recognized. Available: {', '.join(suggestions.keys())}"

        print(f"✓ Created agent with 3 tools")

        # Create Sifaka components
        print("\n2. Creating Sifaka components...")

        # Validators - more achievable constraints
        length_validator = LengthValidator(min_length=100, max_length=400)
        fact_validator = RegexValidator(
            required_patterns=[r"\b(19|20)\d{2}\b"], name="FactValidator"  # Look for years
        )

        # Critic
        critic_model = create_model("openai:gpt-3.5-turbo")
        critic = SelfRefineCritic(model=critic_model)

        print("✓ Created validators and critic")

        # Create file storage in thoughts directory
        thoughts_dir = "thoughts"
        os.makedirs(thoughts_dir, exist_ok=True)
        storage = FileStorage(os.path.join(thoughts_dir, "pydantic_ai_tools_example.json"))
        print("✓ Created file storage in thoughts directory")

        # Create the hybrid chain
        print("\n3. Creating hybrid PydanticAI chain...")
        chain = create_pydantic_chain(
            agent=agent,
            validators=[length_validator, fact_validator],
            critics=[critic],
            storage=storage,
            max_improvement_iterations=2,  # Allow validation constraint prioritization to work
        )
        print("✓ Created hybrid chain with tools")

        # Run the chain
        print("\n4. Running the chain...")
        prompt = """Write a brief article about AI history (maximum 300 characters). Include key dates and use your tools to gather facts. Keep it concise!"""

        print(f"Prompt: {prompt}")
        print("Starting chain execution...")

        # Add timeout handling
        def timeout_handler(signum, frame):
            raise TimeoutError("Chain execution timed out after 120 seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # 2 minute timeout

        try:
            start_time = time.time()
            result = chain.run(prompt)
            end_time = time.time()
            signal.alarm(0)  # Cancel timeout
            print(f"Chain execution completed successfully in {end_time - start_time:.2f} seconds!")
        except TimeoutError as e:
            print(f"❌ Chain execution timed out: {e}")
            signal.alarm(0)
            raise
        except Exception as e:
            print(f"❌ Chain execution failed: {e}")
            signal.alarm(0)
            logger.exception("Chain execution error")
            raise

        # Display results
        print("\n" + "=" * 55)
        print("📊 RESULTS")
        print("=" * 55)

        print(f"\n📝 Generated Text ({len(result.text)} characters):")
        print("-" * 40)
        print(result.text)

        print(f"\n🔄 Iterations: {result.iteration}")

        # Show tool calls
        if hasattr(result, "tool_calls") and result.tool_calls:
            print(f"\n🛠️  Tool Calls ({len(result.tool_calls)}):")
            for i, tool_call in enumerate(result.tool_calls, 1):
                print(f"  {i}. {tool_call.tool_name}")
                if tool_call.arguments:
                    print(f"     Arguments: {tool_call.arguments}")
                if tool_call.result:
                    print(f"     Result: {str(tool_call.result)[:100]}...")
                print(f"     Success: {tool_call.success}")
        else:
            print("\n🛠️  No tool calls recorded")

        # Show validation results
        if hasattr(result, "validation_results") and result.validation_results:
            print(f"\n✅ Validation Results:")
            for name, validation_result in result.validation_results.items():
                status = "PASSED" if validation_result.passed else "FAILED"
                print(f"  - {name}: {status}")
                if not validation_result.passed and validation_result.message:
                    print(f"    Message: {validation_result.message}")

        # Show critic feedback
        if hasattr(result, "critic_feedback") and result.critic_feedback:
            print(f"\n💭 Critic Feedback:")
            for feedback in result.critic_feedback:
                print(f"  - {feedback.critic_name}: {feedback.feedback[:100]}...")
                print(f"    Confidence: {feedback.confidence:.2f}")

        # Check validation status
        validation_passed = True
        if hasattr(result, "validation_results") and result.validation_results:
            validation_passed = all(v.passed for v in result.validation_results.values())

        print(f"\n🎯 Final validation status: {'PASSED' if validation_passed else 'FAILED'}")

        # Show storage location
        print(f"\n💾 Thought saved to: {storage.file_path}")

        print("\n✨ Tools example completed successfully!")

    except ImportError as e:
        if "pydantic_ai" in str(e):
            print("❌ PydanticAI is not installed.")
            print("Please install it with: uv pip install pydantic-ai")
        else:
            print(f"❌ Import error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error running example: {e}")
        logger.exception("Tools example failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
