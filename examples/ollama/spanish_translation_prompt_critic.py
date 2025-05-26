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
from sifaka.utils.logging import get_logger
from sifaka.validators.classifier import ClassifierValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def create_translation_prompt_critic(model):
    """Create a specialized prompt critic for Spanish-English translation."""

    translation_critique_prompt = """
You are an expert translation critic specializing in Spanish to English translation.
Evaluate the following English translation of Spanish text based on these criteria:

1. ACCURACY: Does the translation correctly convey the meaning of the original Spanish?
2. FLUENCY: Does the English text read naturally and fluently?
3. COMPLETENESS: Are all parts of the original text translated?
4. CULTURAL APPROPRIATENESS: Are cultural references and idioms properly adapted?
5. GRAMMAR: Is the English grammar correct and proper?

Original Spanish Text: {original_spanish}
English Translation: {translation}

Provide specific feedback on:
- Any inaccuracies or mistranslations
- Awkward phrasing that could be improved
- Missing or added content
- Cultural nuances that need attention
- Grammar or style improvements

If the translation needs improvement, provide specific suggestions.
If the translation is excellent, acknowledge its quality.

Rate the translation quality: EXCELLENT / GOOD / NEEDS_IMPROVEMENT / POOR
"""

    return PromptCritic(
        model=model,
        critique_prompt=translation_critique_prompt,
        improvement_prompt="Based on the critique feedback, provide an improved English translation that addresses all identified issues while maintaining accuracy and fluency.",
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

    # Spanish text to translate
    spanish_text = """
    La inteligencia artificial está transformando rápidamente nuestra sociedad.
    Desde los asistentes virtuales en nuestros teléfonos hasta los algoritmos
    que recomiendan películas, la IA se ha vuelto parte integral de nuestra
    vida cotidiana. Sin embargo, también plantea importantes desafíos éticos
    y sociales que debemos abordar cuidadosamente. Es fundamental que
    desarrollemos esta tecnología de manera responsable, considerando su
    impacto en el empleo, la privacidad y la equidad social.
    """

    # Create translation prompt
    translation_prompt = f"""
    Translate the following Spanish text to English. Ensure the translation is:
    - Accurate and faithful to the original meaning
    - Fluent and natural in English
    - Culturally appropriate for English speakers
    - Complete (no missing parts)

    Spanish text to translate:
    {spanish_text.strip()}

    Provide only the English translation:
    """

    # Create the chain with exactly 2 retries
    chain = Chain(
        model=model,
        prompt=translation_prompt,
        max_improvement_iterations=2,  # Exactly 2 retries as specified
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,
    )

    # Add validator and critic (no retrievers as specified)
    chain.validate_with(language_validator)
    chain.improve_with(critic)

    # Run the chain
    logger.info("Running Spanish to English translation with prompt critic...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 80)
    print("OLLAMA SPANISH TO ENGLISH TRANSLATION WITH PROMPT CRITIC")
    print("=" * 80)
    print(f"\nOriginal Spanish Text:")
    print("-" * 30)
    print(spanish_text.strip())

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
