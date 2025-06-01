#!/usr/bin/env python3
"""Prompt Critic for Spanish to English Translation Example (PydanticAI).

This example demonstrates:
- PydanticAI agent with Gemini model for translation tasks
- Prompt critic for translation quality assessment
- Language validator to ensure proper English output
- Exactly 2 retries for translation refinement
- No retrievers (pure translation task)
- Modern agent-based workflow with hybrid Chain-Agent architecture

The PydanticAI chain will translate Spanish text to English and use a prompt critic
to ensure translation accuracy, fluency, and cultural appropriateness.

Note: This example uses async execution to avoid event loop conflicts with PydanticAI.
"""

import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent

from sifaka.agents import create_pydantic_chain
from sifaka.classifiers.language import LanguageClassifier
from sifaka.critics.prompt import PromptCritic
from sifaka.storage import FileStorage
from sifaka.utils.logging import get_logger
from sifaka.validators.classifier import ClassifierValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


async def main():
    """Run the Spanish Translation with Prompt Critic example using PydanticAI."""

    logger.info("Creating PydanticAI Spanish translation with prompt critic example")

    # Use Gemini model directly for the critic
    logger.info("Using Gemini model for prompt critic")

    # Create translation-specific prompt critic with Gemini
    critic = PromptCritic(
        model_name="google-gla:gemini-1.5-flash",
        name="Translation Quality Critic",
        criteria=[
            "Language correctness",
            "Translation accuracy and completeness",
            "Fluency and natural expression",
            "Cultural appropriateness",
        ],
        system_prompt=(
            "You are a translation expert. When text fails validation because it's in the wrong language"
            "you MUST translate it to the correct language. This is your PRIMARY responsibility."
        ),
        critique_prompt_template=(
            "Evaluate this text for translation quality:\n\n"
            "Original task: {prompt}\n\n"
            "Text to evaluate:\n{text}\n\n"
            "Context: {context}\n\n"
            "CRITICAL: If this text is not in the correct language but should be, this is a MAJOR issue.\n\n"
            "Provide your assessment:\n"
            "Issues:\n- [List any language or translation issues]\n\n"
            "Suggestions:\n- [List specific improvements needed]\n\n"
            "Overall Assessment: [Your assessment]"
        ),
        improve_prompt_template=(
            "TRANSLATION TASK: The text below needs to be corrected.\n\n"
            "Original task: {prompt}\n\n"
            "Current text (may be in wrong language):\n{text}\n\n"
            "Context: {context}\n\n"
            "Issues identified:\n{critique}\n\n"
            "INSTRUCTIONS:\n"
            "1. Preserve all content, meaning, and structure\n"
            "2. Make the writing natural and fluent\n"
            "3. Do not add commentary - just provide the corrected/translated text\n\n"
            "Corrected text:"
        ),
    )

    # Create language validator to ensure English output
    language_classifier = LanguageClassifier()
    language_validator = ClassifierValidator(
        classifier=language_classifier,
        expected_label="en",  # Must be English
        threshold=0.8,
        name="English Output Validator",
    )

    # Create PydanticAI agent with Gemini model
    agent = Agent(
        model="google-gla:gemini-1.5-flash",
        system_prompt=(
            "You are a translation assistant. Follow the user's instructions exactly. "
            "If asked to write in Spanish, write in Spanish. If asked to translate to English, "
            "provide only the English translation without any additional commentary."
        ),
    )

    # Create a prompt that will initially generate Spanish text (to trigger validation failure)
    # This makes the example more interesting by demonstrating the full validation → criticism → improvement cycle
    spanish_generation_prompt = """
    Escribe un ensayo detallado en español sobre los beneficios y desafíos de la inteligencia artificial en la educación moderna.
    Incluye ejemplos específicos de cómo la IA puede personalizar el aprendizaje, automatizar tareas administrativas,
    y mejorar la accesibilidad educativa. También discute las preocupaciones sobre la privacidad de datos estudiantiles
    y la necesidad de mantener la interacción humana en el proceso educativo.

    Responde completamente en español con al menos 200 palabras.
    """

    # Define the prompt
    prompt = spanish_generation_prompt.strip()

    # Create PydanticAI chain with exactly 2 retries
    chain = create_pydantic_chain(
        agent=agent,
        validators=[language_validator],  # Language validator to ensure English output
        critics=[critic],  # Translation prompt critic
        max_improvement_iterations=2,  # Exactly 2 retries as specified
        always_apply_critics=False,  # Only apply critics when validation fails
        analytics_storage=FileStorage(
            "./thoughts/prompt_critic_example_thoughts.json",
            overwrite=True,  # Overwrite existing file instead of appending
        ),  # Save thoughts to single JSON file for debugging
    )

    print(
        f"DEBUG: Created PydanticAI chain with {len([critic])} critics and {len([language_validator])} validators"
    )

    # Run the chain asynchronously (this is the preferred approach)
    logger.info("Running PydanticAI Spanish to English translation with prompt critic...")
    result = await chain.run(prompt)

    # Display results
    print("\n" + "=" * 80)
    print("PYDANTIC AI SPANISH GENERATION → VALIDATION FAILURE → TRANSLATION EXAMPLE")
    print("=" * 80)

    # Show the initial Spanish generation (from iteration 0 if available)
    initial_spanish = None
    if hasattr(result, "history") and result.history:
        # Try to get the first iteration's text from storage
        try:
            first_iteration_id = result.history[-1].thought_id  # Last in history is first iteration
            first_thought = chain._config.storage.get(first_iteration_id)
            if first_thought and hasattr(first_thought, "text") and first_thought.text:
                initial_spanish = first_thought.text
        except:
            pass

    if initial_spanish:
        print(f"\nInitial Spanish Generation ({len(initial_spanish)} characters):")
        print("-" * 30)
        print(initial_spanish[:500] + "..." if len(initial_spanish) > 500 else initial_spanish)
    else:
        print(f"\nNote: Initial Spanish generation not available in display")

    print(f"\nFinal English Translation ({len(result.text)} characters):")
    print("-" * 30)
    print(result.text)

    print(f"\nTranslation Process:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Max Retries: 2 (as specified)")
    print(f"  Chain ID: {result.chain_id}")

    # Show validation results
    if result.validation_results:
        print(f"\nLanguage Validation:")
        for validator_name, validation_result in result.validation_results.items():
            status = "✓ PASSED" if validation_result.passed else "✗ FAILED"
            print(f"  {validator_name}: {status}")
            if validation_result.passed:
                print(f"    Detected language: English (confidence: {validation_result.score:.2f})")
            else:
                print(f"    Error: {validation_result.message}")

    # Show translation critic feedback
    if result.critic_feedback:
        print(f"\nTranslation Critic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Translation Quality Assessment:")
                # Extract quality rating if present
                suggestions = feedback.suggestions
                if "EXCELLENT" in suggestions:
                    print(f"     Rating: EXCELLENT")
                elif "GOOD" in suggestions:
                    print(f"     Rating: GOOD")
                elif "NEEDS_IMPROVEMENT" in suggestions:
                    print(f"     Rating: NEEDS_IMPROVEMENT")
                elif "POOR" in suggestions:
                    print(f"     Rating: POOR")

                print(f"     Detailed Feedback: {suggestions[:400]}...")

    print(f"\nTranslation Features:")
    print(f"  - No retrievers used (pure translation)")
    print(f"  - Gemini Flash model processing via PydanticAI")
    print(f"  - Specialized translation critique")
    print(f"  - Language validation")
    print(f"  - Modern agent-based workflow")

    print("\n" + "=" * 80)
    logger.info("PydanticAI Spanish translation with prompt critic example completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
