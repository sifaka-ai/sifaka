"""Core Sifaka engine that coordinates text improvement."""

import time
from typing import List, Optional

from ..models import SifakaResult, Generation
from ..config import Config
from ..interfaces import Validator
from ..exceptions import TimeoutError, ModelProviderError
from ...validators import LengthValidator
from ...storage import StorageBackend, MemoryStorage
from .generation import TextGenerator
from .orchestration import CriticOrchestrator
from .validation import ValidationRunner


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
        
        self.validator = ValidationRunner()
    
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
                validation_passed = await self.validator.run_validators(
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
                    result.add_critique(
                        critic=critique.critic,
                        feedback=critique.feedback,
                        suggestions=critique.suggestions,
                        needs_improvement=critique.needs_improvement,
                        confidence=critique.confidence
                    )
                
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
                    
                    if improved_text:
                        # Add generation to result
                        result.add_generation(
                            text=improved_text,
                            model=self.config.model,
                            prompt=prompt
                        )
                        result.final_text = improved_text
                        
                except ModelProviderError:
                    raise
                except Exception as e:
                    # Log generation error
                    result.add_critique(
                        critic="system",
                        feedback=f"Generation failed: {str(e)}",
                        suggestions=["Try with different parameters"],
                        needs_improvement=False,
                        confidence=0.0
                    )
                    break
                
                # Check memory bounds
                self.validator.check_memory_bounds(result)
        
        except TimeoutError:
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
                elapsed_time=elapsed,
                limit=self.config.timeout_seconds
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
    
    async def _finalize_result(
        self,
        result: SifakaResult,
        start_time: float
    ) -> None:
        """Finalize the result object."""
        result.processing_time = time.time() - start_time
        
        # Store result if storage is configured
        try:
            await self.storage.save(result)
        except Exception:
            # Storage errors are non-fatal
            pass