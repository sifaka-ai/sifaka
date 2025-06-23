"""Core engine for Sifaka text improvement."""

import asyncio
import time
from typing import List, Optional, Tuple

import openai
from ..core.models import SifakaResult, Config
from ..core.interfaces import Validator, Critic
from ..core.exceptions import ValidationError, TimeoutError, classify_openai_error
from ..core.llm_client import LLMManager
from ..validators import LengthValidator
from ..storage import StorageBackend, MemoryStorage
from ..critics import create_critics


class SifakaEngine:
    """Core engine for iterative text improvement."""
    
    IMPROVEMENT_SYSTEM_PROMPT = """You are an expert text editor focused on iterative improvement. Pay careful attention to all critic feedback and validation issues. Your goal is to address each piece of feedback thoroughly while maintaining the original intent and improving the overall quality of the text."""

    def __init__(
        self, config: Optional[Config] = None, storage: Optional[StorageBackend] = None
    ):
        self.config = config or Config()
        # Initialize storage backend
        self.storage = storage or MemoryStorage()


    async def improve(
        self, text: str, validators: Optional[List[Validator]] = None
    ) -> SifakaResult:
        """Improve text through iterative critique and refinement."""
        start_time = time.time()

        # Initialize result
        result = SifakaResult(original_text=text, final_text=text)  # Will be updated

        # Use provided validators or defaults
        if validators is None:
            validators = [LengthValidator(min_length=50)]

        # Initialize critics
        critics = self._initialize_critics()

        try:
            # Iterative improvement loop
            for iteration in range(self.config.max_iterations):
                result.increment_iteration()
                current_text = result.current_text

                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > self.config.timeout_seconds:
                    raise TimeoutError(elapsed_time, self.config.timeout_seconds)

                # Run validators
                validation_passed = await self._run_validators(
                    current_text, result, validators
                )

                # Run critics
                await self._run_critics(current_text, result, critics)

                # Check if we should continue
                if not self.config.force_improvements and validation_passed and not result.needs_improvement:
                    break

                # Generate improved text (always, even on last iteration)
                improved_text, prompt_used = await self._generate_improved_text(
                    current_text, result
                )
                if improved_text and improved_text != current_text:
                    result.add_generation(
                        text=improved_text,
                        model=self.config.model,
                        prompt=prompt_used,
                        processing_time=time.time() - start_time,
                    )

        except TimeoutError:
            # Re-raise timeout errors as they're user-controllable
            raise
        except openai.OpenAIError as e:
            # Handle OpenAI-specific errors
            raise classify_openai_error(e)
        except Exception as e:
            # Handle unexpected errors gracefully
            result.add_critique(
                critic="system",
                feedback=f"Unexpected error during improvement: {str(e)}",
                suggestions=[
                    "Try again with different parameters",
                    "Check your configuration",
                ],
                needs_improvement=False,
            )

        # Finalize the result
        await self._finalize_result(result, start_time)

        return result


    async def _finalize_result(self, result: SifakaResult, start_time: float) -> None:
        """Finalize the result with timing and storage."""
        result.set_final_text(result.current_text)
        result.processing_time = time.time() - start_time
        await self.storage.save(result)

    def _initialize_critics(self) -> List[Critic]:
        """Initialize critic instances from configuration."""
        # Use critic-specific model/temperature if provided
        critic_model = self.config.critic_model or self.config.model
        critic_temp = self.config.critic_temperature if self.config.critic_temperature is not None else self.config.temperature
        
        return create_critics(
            names=self.config.critics,
            model=critic_model,
            temperature=critic_temp,
        )

    async def _run_validators(
        self, text: str, result: SifakaResult, validators: List[Validator]
    ) -> bool:
        """Run all validators and return if all passed."""
        all_passed = True

        # Run validators concurrently
        validation_tasks = [
            validator.validate(text, result) for validator in validators
        ]
        validation_results = await asyncio.gather(
            *validation_tasks, return_exceptions=True
        )

        for validator, validation_result in zip(validators, validation_results):
            if isinstance(validation_result, Exception):
                # Handle validator errors with specific exception types
                error_msg = (
                    f"Validator '{validator.name}' failed: {str(validation_result)}"
                )
                raise ValidationError(error_msg, validator.name)
            elif hasattr(
                validation_result, "validator"
            ):  # Type guard for ValidationResult
                result.add_validation(
                    validator=validation_result.validator,
                    passed=validation_result.passed,
                    score=validation_result.score,
                    details=validation_result.details,
                )
                if not validation_result.passed:
                    all_passed = False

        return all_passed

    async def _run_critics(
        self, text: str, result: SifakaResult, critics: List[Critic]
    ) -> None:
        """Run all critics and collect feedback."""
        # Run critics concurrently
        critique_tasks = [critic.critique(text, result) for critic in critics]
        critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)

        for critic, critique_result in zip(critics, critique_results):
            if isinstance(critique_result, Exception):
                # Handle critic errors with graceful degradation
                if isinstance(critique_result, openai.OpenAIError):
                    # OpenAI errors in critics are usually temporary
                    result.add_critique(
                        critic=critic.name,
                        feedback=f"Critic temporarily unavailable: {str(critique_result)}",
                        suggestions=["Try again later or use different critics"],
                        needs_improvement=True,  # Still try to improve even with errors
                    )
                else:
                    # Other critic errors
                    result.add_critique(
                        critic=critic.name,
                        feedback=f"Critic error: {str(critique_result)}",
                        suggestions=[
                            "Manual review recommended",
                            "Try different critics",
                        ],
                        needs_improvement=True,  # Still try to improve even with errors
                    )
            elif hasattr(critique_result, "critic"):  # Type guard for CritiqueResult
                result.add_critique(
                    critic=critique_result.critic,
                    feedback=critique_result.feedback,
                    suggestions=critique_result.suggestions,
                    needs_improvement=critique_result.needs_improvement,
                    confidence=critique_result.confidence,
                )

    async def _generate_improved_text(
        self, current_text: str, result: SifakaResult
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate improved text based on feedback."""
        try:
            # Collect recent feedback
            feedback_items = []

            # Add validation feedback
            for validation in list(result.validations)[-3:]:  # Last 3 validations
                if not validation.passed:
                    feedback_items.append(f"Validation issue: {validation.details}")

            # Add critique feedback
            for critique in list(result.critiques)[-3:]:  # Last 3 critiques
                if critique.needs_improvement:
                    feedback_items.append(f"Feedback: {critique.feedback}")
                    feedback_items.extend(
                        f"Suggestion: {s}" for s in critique.suggestions[:2]
                    )

            if not feedback_items and not self.config.force_improvements:
                return None, None  # No improvements needed
            
            # If force_improvements is True, ensure we have some feedback
            if not feedback_items and self.config.force_improvements:
                feedback_items.append("Please enhance the text with more detail, clarity, and examples.")

            # Create improvement prompt
            feedback_text = "\n".join(feedback_items)
            prompt = f"""Please improve the following text based on the feedback provided:

CURRENT TEXT:
{current_text}

FEEDBACK TO ADDRESS:
{feedback_text}

Please provide an improved version that addresses the feedback while maintaining the original intent and style. Only return the improved text, no explanations."""

            # Show prompt if requested
            if self.config.show_improvement_prompt:
                print("\n" + "="*50)
                print("IMPROVEMENT PROMPT:")
                print("="*50)
                print(prompt)
                print("="*50 + "\n")

            # Generate improved text using the new client
            client = LLMManager.get_client(
                model=self.config.model,
                temperature=self.config.temperature
            )
            
            response = await client.complete(
                messages=[
                    {
                        "role": "system",
                        "content": self.IMPROVEMENT_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            improved_text = response.content.strip()

            # Basic sanity check
            if len(improved_text) < len(current_text) * 0.3:
                return None, None  # Probably too short, skip

            return improved_text, prompt

        except Exception:
            return None, None  # Failed to generate, skip
