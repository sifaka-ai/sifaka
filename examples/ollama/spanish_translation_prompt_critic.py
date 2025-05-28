#!/usr/bin/env python3
"""Ollama Prompt Critic for Spanish to English Translation Example.

This example demonstrates:
- Ollama local model for translation tasks
- Prompt critic for translation quality assessment
- Language validator to ensure proper English output
- Exactly 2 retries for translation refinement
- No retrievers (pure translation task)

The chain will translate Spanish text to English and use a prompt critic
to ensure translation accuracy, fluency, and cultural appropriateness.
"""

from dotenv import load_dotenv

from sifaka.classifiers.language import LanguageClassifier
from sifaka.core.chain import Chain
from sifaka.critics.prompt import PromptCritic
from sifaka.models.ollama import OllamaModel
from sifaka.storage import FileStorage
from sifaka.utils.logging import get_logger
from sifaka.validators.classifier import ClassifierValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def create_translation_prompt_critic(model):
    """Create a specialized prompt critic for Spanish-English translation."""

    # This critique prompt will detect that the text is in Spanish when English is required
    translation_critique_prompt = """
You are a language validation critic. Your job is to check if the provided text meets the language requirements.

REQUIREMENT: The text must be in English.

Analyze the provided text and determine what language it is written in.

Text to analyze: {text}

If the text contains Spanish words, phrases, or is written in Spanish, respond with:
Issues:
- Text is written in Spanish but English is required
- Spanish language detected in content

Suggestions:
- Translate the entire text from Spanish to English
- Ensure all content is in English language

If the text is properly written in English, respond with:
Assessment: Text is properly written in English and meets requirements
"""

    # This improvement prompt will actually do the translation
    improvement_prompt = """
The text you received is in Spanish, but the system requires English output.

Please translate the following Spanish text to English:

{text}

Requirements for translation:
1. ACCURACY: Correctly convey the meaning of the original Spanish
2. FLUENCY: Natural and fluent English text
3. COMPLETENESS: Translate all parts of the original text
4. CULTURAL APPROPRIATENESS: Adapt cultural references for English speakers
5. GRAMMAR: Correct English grammar and style

Provide ONLY the English translation. Do not include any Spanish text, explanations, or commentary.
"""

    return PromptCritic(
        model=model,
        critique_prompt_template=translation_critique_prompt,
        improve_prompt_template=improvement_prompt,
        name="Spanish-English Translation Critic",
    )


def main():
    """Run the Ollama Spanish Translation with Prompt Critic example."""

    logger.info("Creating Ollama Spanish translation with prompt critic example")

    # Create Ollama model (using a model good for translation)
    model = OllamaModel(
        model_name="mistral:latest",  # Using available model
        base_url="http://localhost:11434",
        temperature=0.3,  # Lower temperature for more consistent translations
        max_tokens=800,
    )

    # Test if Ollama is available
    try:
        if not model.connection.health_check():
            raise Exception("Ollama server health check failed")
        logger.info("Ollama service is available")
    except Exception as e:
        logger.error(f"Ollama service not available: {e}")
        print("Error: Ollama service is not running. Please start Ollama and try again.")
        return

    # Create translation prompt critic
    critic = create_translation_prompt_critic(model)

    # Create language validator to ensure English output
    language_classifier = LanguageClassifier()
    language_validator = ClassifierValidator(
        classifier=language_classifier,
        expected_label="en",  # Must be English
        threshold=0.8,
        name="English Output Validator",
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

    # Create the chain with exactly 2 retries
    chain = Chain(
        model=model,
        prompt=spanish_generation_prompt,
        max_improvement_iterations=2,  # Exactly 2 retries as specified
        apply_improvers_on_validation_failure=True,  # Apply critic when validation fails
        always_apply_critics=False,  # Only apply critics when validation fails
        storage=FileStorage(
            "./thoughts/spanish_translation_prompt_critic_thoughts.json",
            overwrite=True,  # Overwrite existing file instead of appending
        ),  # Save thoughts to single JSON file for debugging
    )

    # Add validator and critic (no retrievers as specified)
    chain = chain.validate_with(language_validator)
    chain = chain.improve_with(critic)

    # Run the chain
    logger.info("Running Spanish to English translation with prompt critic...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 80)
    print("OLLAMA SPANISH GENERATION → VALIDATION FAILURE → TRANSLATION EXAMPLE")
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
    print(f"  - Local Ollama model processing")
    print(f"  - Specialized translation critique")
    print(f"  - Language validation")

    print("\n" + "=" * 80)
    logger.info("Spanish translation with prompt critic example completed successfully")


if __name__ == "__main__":
    main()
