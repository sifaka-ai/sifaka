"""Consolidated Sifaka engine with text generation and critic orchestration."""

import asyncio
import time
from typing import List, Optional, Tuple

from ..core.models import SifakaResult, CritiqueResult, GenerationResult
from ..core.config import Config
from ..core.interfaces import Validator, Critic
from ..core.exceptions import ValidationError, TimeoutError, ModelProviderError, CriticError
from ..core.llm_client import LLMManager
from ..core.constants import ROLE_SYSTEM, ROLE_USER
from ..critics import create_critics
from ..validators import LengthValidator
from ..storage import StorageBackend, MemoryStorage


class TextGenerator:
    """Handles text generation and improvement."""
    
    IMPROVEMENT_SYSTEM_PROMPT = """You are an expert text editor focused on iterative improvement. Pay careful attention to all critic feedback and validation issues. Your goal is to address each piece of feedback thoroughly while maintaining the original intent and improving the overall quality of the text."""
    
    def __init__(self, model: str, temperature: float):
        """Initialize text generator.
        
        Args:
            model: LLM model to use
            temperature: Generation temperature
        """
        self.model = model
        self.temperature = temperature
        self._client = None
    
    @property
    def client(self):
        """Get or create LLM client."""
        if self._client is None:
            self._client = LLMManager.get_client(
                model=self.model,
                temperature=self.temperature
            )
        return self._client
    
    async def generate_improvement(
        self,
        current_text: str,
        result: SifakaResult,
        show_prompt: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate improved text based on feedback.
        
        Args:
            current_text: Current version of text
            result: Result object with critique history
            show_prompt: Whether to print the prompt
            
        Returns:
            Tuple of (improved_text, prompt_used)
        """
        # Build improvement prompt
        prompt = self._build_improvement_prompt(current_text, result)
        
        if show_prompt:
            print("\n" + "="*80)
            print("IMPROVEMENT PROMPT")
            print("="*80)
            print(prompt)
            print("="*80 + "\n")
        
        # Generate improvement
        messages = [
            {"role": ROLE_SYSTEM, "content": self.IMPROVEMENT_SYSTEM_PROMPT},
            {"role": ROLE_USER, "content": prompt}
        ]
        
        try:
            response = await self.client.complete(messages)
            improved_text = response.content.strip()
            
            # Validate improvement
            if not improved_text or improved_text == current_text:
                return None, prompt
                
            return improved_text, prompt
            
        except Exception as e:
            # Return None on error, let engine handle it
            return None, prompt
    
    def _build_improvement_prompt(self, text: str, result: SifakaResult) -> str:
        """Build prompt for text improvement."""
        prompt_parts = [
            "Please improve the following text based on the feedback provided.",
            f"\nCurrent text:\n{text}\n"
        ]
        
        # Add validation feedback
        if result.validations:
            validation_feedback = self._format_validation_feedback(result)
            if validation_feedback:
                prompt_parts.append(f"\nValidation issues:\n{validation_feedback}\n")
        
        # Add critique feedback
        if result.critiques:
            critique_feedback = self._format_critique_feedback(result)
            if critique_feedback:
                prompt_parts.append(f"\nCritic feedback:\n{critique_feedback}\n")
        
        # Add improvement instructions
        prompt_parts.append(
            "\nProvide an improved version that addresses all feedback while "
            "maintaining the original intent. Return only the improved text."
        )
        
        return "".join(prompt_parts)
    
    def _format_validation_feedback(self, result: SifakaResult) -> str:
        """Format validation feedback for prompt."""
        feedback_lines = []
        
        # Get recent validations
        recent_validations = list(result.validations)[-5:]
        
        for validation in recent_validations:
            if not validation.passed:
                feedback_lines.append(
                    f"- {validation.validator}: {validation.message}"
                )
        
        return "\n".join(feedback_lines)
    
    def _format_critique_feedback(self, result: SifakaResult) -> str:
        """Format critique feedback for prompt."""
        feedback_lines = []
        
        # Get recent critiques
        recent_critiques = list(result.critiques)[-5:]
        
        for critique in recent_critiques:
            if critique.needs_improvement:
                # Add main feedback
                feedback_lines.append(f"\n{critique.critic}:")
                feedback_lines.append(f"- {critique.feedback}")
                
                # Add specific suggestions
                if critique.suggestions:
                    feedback_lines.append("  Suggestions:")
                    for suggestion in critique.suggestions[:3]:
                        feedback_lines.append(f"  * {suggestion}")
        
        return "\n".join(feedback_lines)


class CriticOrchestrator:
    """Orchestrates critic execution and feedback collection."""
    
    def __init__(
        self,
        critic_names: List[str],
        model: str,
        temperature: float,
        critic_model: Optional[str] = None,
        critic_temperature: Optional[float] = None
    ):
        """Initialize orchestrator.
        
        Args:
            critic_names: Names of critics to use
            model: Default model
            temperature: Default temperature
            critic_model: Override model for critics
            critic_temperature: Override temperature for critics
        """
        self.critic_names = critic_names
        self.model = critic_model or model
        self.temperature = critic_temperature or temperature
        self._critics: Optional[List[Critic]] = None
    
    @property
    def critics(self) -> List[Critic]:
        """Get or create critics."""
        if self._critics is None:
            self._critics = create_critics(
                self.critic_names,
                model=self.model,
                temperature=self.temperature
            )
        return self._critics
    
    async def run_critics(
        self,
        text: str,
        result: SifakaResult
    ) -> List[CritiqueResult]:
        """Run all critics on the text.
        
        Args:
            text: Text to critique
            result: Result object with history
            
        Returns:
            List of critique results
        """
        if not self.critics:
            return []
        
        # Run critics in parallel
        critique_tasks = [
            critic.critique(text, result) for critic in self.critics
        ]
        
        critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)
        
        # Process results
        valid_critiques = []
        for i, critique in enumerate(critiques):
            if isinstance(critique, Exception):
                # Create error critique
                error_critique = CritiqueResult(
                    critic=self.critics[i].name,
                    feedback=f"Error during critique: {str(critique)}",
                    suggestions=["Review the text manually"],
                    needs_improvement=True,
                    confidence=0.0
                )
                valid_critiques.append(error_critique)
            else:
                valid_critiques.append(critique)
        
        return valid_critiques
    
    def analyze_consensus(self, critiques: List[CritiqueResult]) -> bool:
        """Analyze if critics agree improvement is needed.
        
        Args:
            critiques: List of critique results
            
        Returns:
            True if majority think improvement is needed
        """
        if not critiques:
            return False
        
        needs_improvement_count = sum(
            1 for c in critiques if c.needs_improvement
        )
        
        # Majority vote
        return needs_improvement_count > len(critiques) / 2
    
    def get_aggregated_confidence(self, critiques: List[CritiqueResult]) -> float:
        """Get aggregated confidence from all critics.
        
        Args:
            critiques: List of critique results
            
        Returns:
            Average confidence score
        """
        if not critiques:
            return 0.0
        
        confidences = [c.confidence for c in critiques if c.confidence > 0]
        
        if not confidences:
            return 0.0
            
        return sum(confidences) / len(confidences)


class SifakaEngine:
    """Main engine for Sifaka text improvement."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        storage: Optional[StorageBackend] = None
    ):
        """Initialize engine with configuration."""
        self.config = config or Config()
        self.storage = storage or MemoryStorage()
        
        # Initialize components
        self.generator = TextGenerator(
            model=self.config.model,
            temperature=self.config.temperature
        )
        
        self.orchestrator = CriticOrchestrator(
            critic_names=self.config.critics,
            model=self.config.model,
            temperature=self.config.temperature,
            critic_model=self.config.critic_model,
            critic_temperature=self.config.critic_temperature
        )
    
    async def improve(
        self,
        text: str,
        validators: Optional[List[Validator]] = None
    ) -> SifakaResult:
        """Improve text through iterative critique and refinement.
        
        Args:
            text: Text to improve
            validators: Optional quality validators
            
        Returns:
            SifakaResult with improved text and audit trail
        """
        start_time = time.time()
        
        # Initialize result
        result = SifakaResult(original_text=text, final_text=text)
        
        # Use default validators if none provided
        if validators is None:
            validators = [LengthValidator(min_length=50)]
        
        try:
            # Improvement loop
            for iteration in range(self.config.max_iterations):
                result.increment_iteration()
                current_text = result.current_text
                
                # Check timeout
                self._check_timeout(start_time)
                
                # Run validation
                validation_passed = await self._run_validators(
                    current_text, result, validators
                )
                
                # Run critics
                try:
                    critiques = await self.orchestrator.run_critics(
                        current_text, result
                    )
                except Exception as e:
                    # Log critic error and continue
                    result.add_critique(
                        critic="system",
                        feedback=f"Critics failed: {str(e)}",
                        suggestions=["Continue without critic feedback"],
                        needs_improvement=False,
                        confidence=0.0
                    )
                    critiques = []
                
                # Add critiques to result
                for critique in critiques:
                    result.add_critique_result(critique)
                
                # Check if improvement needed
                needs_improvement = self.orchestrator.analyze_consensus(critiques)
                
                # Decide whether to continue
                if not self._should_continue(validation_passed, needs_improvement):
                    break
                
                # Generate improvement
                try:
                    improved_text, prompt = await self.generator.generate_improvement(
                        current_text,
                        result,
                        self.config.show_improvement_prompt
                    )
                except Exception as e:
                    # Log generation error and break
                    result.add_critique(
                        critic="system",
                        feedback=f"Text generation failed: {str(e)}",
                        suggestions=["Try again later"],
                        needs_improvement=False,
                        confidence=0.0
                    )
                    break
                
                if improved_text and improved_text != current_text:
                    result.add_generation(
                        text=improved_text,
                        model=self.config.model,
                        prompt=prompt,
                        processing_time=time.time() - start_time
                    )
        
        except TimeoutError:
            # Re-raise with context
            raise
        except (ModelProviderError, CriticError, ValidationError) as e:
            # Re-raise known errors
            raise
        except Exception as e:
            # Log unexpected errors as system critique
            result.add_critique(
                critic="system",
                feedback=f"Unexpected error: {type(e).__name__}: {str(e)}",
                suggestions=["Check logs for details"],
                needs_improvement=False,
                confidence=0.0
            )
        
        finally:
            # Finalize result
            await self._finalize_result(result, start_time)
        
        return result
    
    def _check_timeout(self, start_time: float) -> None:
        """Check if operation has timed out."""
        elapsed = time.time() - start_time
        if elapsed > self.config.timeout_seconds:
            raise TimeoutError(
                f"Operation timed out after {elapsed:.1f} seconds"
            )
    
    def _should_continue(
        self,
        validation_passed: bool,
        needs_improvement: bool
    ) -> bool:
        """Determine if improvement should continue."""
        if self.config.force_improvements:
            return True
        
        return not validation_passed or needs_improvement
    
    async def _run_validators(
        self,
        text: str,
        result: SifakaResult,
        validators: List[Validator]
    ) -> bool:
        """Run validators on text."""
        all_passed = True
        
        for validator in validators:
            try:
                validation_result = await validator.validate(text)
                result.add_validation(validation_result)
                
                if not validation_result.passed:
                    all_passed = False
                    
            except Exception as e:
                # Create error validation
                result.add_validation_error(
                    validator=validator.name,
                    error=str(e)
                )
                all_passed = False
        
        return all_passed
    
    async def _finalize_result(
        self,
        result: SifakaResult,
        start_time: float
    ) -> None:
        """Finalize the result object."""
        result.processing_time = time.time() - start_time
        
        # Store result if storage is configured
        if self.storage:
            await self.storage.save(result)