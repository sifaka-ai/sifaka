#!/usr/bin/env python3
"""Ollama Prompt Critic for Spanish to English Translation Example (PydanticAI).

This example demonstrates:
- PydanticAI agent with Ollama local model for translation tasks
- Prompt critic for translation quality assessment
- Language validator to ensure proper English output
- Exactly 2 retries for translation refinement
- No retrievers (pure translation task)
- Modern agent-based workflow with hybrid Chain-Agent architecture

The PydanticAI chain will translate Spanish text to English and use a prompt critic
to ensure translation accuracy, fluency, and cultural appropriateness.
"""

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from sifaka.agents import create_pydantic_chain
from sifaka.classifiers.language import LanguageClassifier
from sifaka.critics.prompt import PromptCritic
from sifaka.models.ollama import OllamaModel
from sifaka.storage import FileStorage
from sifaka.utils.logging import get_logger
from sifaka.validators.classifier import ClassifierValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def create_language_aware_critic(model, target_language="English", target_language_code="en"):
    """Create a generic language-aware critic that responds to validation feedback.

    This critic works in conjunction with language validators to provide appropriate
    feedback when content is in the wrong language. It's more flexible and reusable
    than a hardcoded translation critic.

    Args:
        model: The model to use for criticism and improvement
        target_language: Human-readable name of the target language (e.g., "English")
        target_language_code: Language code for the target language (e.g., "en")
    """

    # Generic critique prompt that responds to validation context
    critique_prompt = """
You are a language quality critic. Analyze the provided text for language compliance and quality.

Text to analyze: {text}

Based on the text analysis, provide feedback on:
1. Language compliance - Is the text in the expected language?
2. Content quality - Is the content well-structured and coherent?
3. Any improvements needed

If the text appears to be in the wrong language, note this as an issue.
If the text meets language requirements, focus on content quality feedback.

Provide constructive feedback for improvement.
"""

    # Generic improvement prompt that adapts based on the issue
    improvement_prompt = """
Please improve the following text based on the feedback provided.

Original text: {text}

If the text is in the wrong language, translate it to {target_language} while:
1. ACCURACY: Correctly conveying the original meaning
2. FLUENCY: Using natural, fluent {target_language}
3. COMPLETENESS: Including all parts of the original content
4. APPROPRIATENESS: Adapting cultural references for {target_language} speakers
5. GRAMMAR: Using correct {target_language} grammar and style

If the text is already in {target_language}, focus on improving:
- Clarity and readability
- Grammar and style
- Content structure and flow

Provide the improved text without additional commentary.
"""

    # Format the improvement prompt with the target language
    formatted_improvement_prompt = improvement_prompt.replace("{target_language}", target_language)

    return PromptCritic(
        model=model,
        critique_prompt_template=critique_prompt,
        improve_prompt_template=formatted_improvement_prompt,
        name=f"Language-Aware Critic ({target_language})",
    )


def main():
    """Run the Ollama Spanish Translation with Prompt Critic example using PydanticAI."""

    logger.info("Creating PydanticAI Ollama Spanish translation with prompt critic example")

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

    # Create language-aware critic that works with the validator
    critic = create_language_aware_critic(
        model, target_language="English", target_language_code="en"
    )

    # Create language validator to ensure English output
    language_classifier = LanguageClassifier()
    language_validator = ClassifierValidator(
        classifier=language_classifier,
        expected_label="en",  # Must be English
        threshold=0.8,
        name="English Output Validator",
    )

    # Create PydanticAI agent with Ollama model (using OpenAI-compatible API)
    ollama_model = OpenAIModel(
        model_name="mistral",  # Model name in Ollama
        provider=OpenAIProvider(
            base_url="http://localhost:11434/v1"
        ),  # Ollama OpenAI-compatible endpoint
    )

    agent = Agent(
        model=ollama_model,
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
        storage=FileStorage(
            "./thoughts/spanish_translation_prompt_critic_thoughts.json",
            overwrite=True,  # Overwrite existing file instead of appending
        ),  # Save thoughts to single JSON file for debugging
    )

    print(
        f"DEBUG: Created PydanticAI chain with {len([critic])} critics and {len([language_validator])} validators"
    )

    # Run the chain
    logger.info("Running PydanticAI Spanish to English translation with prompt critic...")
    result = chain.run(prompt)

    # Display results
    print("\n" + "=" * 80)
    print("PYDANTIC AI OLLAMA SPANISH GENERATION → VALIDATION FAILURE → TRANSLATION EXAMPLE")
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
    print(f"  - Local Ollama model processing via PydanticAI")
    print(f"  - Specialized translation critique")
    print(f"  - Language validation")
    print(f"  - Modern agent-based workflow")

    print("\n" + "=" * 80)
    logger.info("PydanticAI Spanish translation with prompt critic example completed successfully")


if __name__ == "__main__":
    main()
