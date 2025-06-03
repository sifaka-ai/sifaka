"""Coherence validator for checking text coherence and readability.

This module provides validators for checking text coherence, readability,
and logical flow. Designed for the new PydanticAI-based Sifaka architecture.
"""

import re
from typing import List, Optional, Dict, Any
import asyncio

from sifaka.core.thought import SifakaThought
from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger
from sifaka.validators.base import BaseValidator, ValidationResult, TimingMixin

logger = get_logger(__name__)


class CoherenceValidator(BaseValidator, TimingMixin):
    """Validator that checks text coherence and readability.
    
    This validator analyzes text for various coherence metrics including:
    - Sentence structure and variety
    - Paragraph organization
    - Transition words and phrases
    - Repetition and redundancy
    - Logical flow indicators
    
    Attributes:
        min_sentences: Minimum number of sentences required
        max_repetition_ratio: Maximum allowed repetition ratio
        min_transition_ratio: Minimum required transition word ratio
        check_paragraphs: Whether to check paragraph structure
        strict: Whether to fail validation on any violation
    """
    
    def __init__(
        self,
        min_sentences: int = 2,
        max_repetition_ratio: float = 0.3,
        min_transition_ratio: float = 0.05,
        check_paragraphs: bool = True,
        strict: bool = False,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the coherence validator.
        
        Args:
            min_sentences: Minimum number of sentences required
            max_repetition_ratio: Maximum allowed repetition ratio (0.0-1.0)
            min_transition_ratio: Minimum required transition word ratio (0.0-1.0)
            check_paragraphs: Whether to check paragraph structure
            strict: Whether to fail validation on any violation
            name: Custom name for the validator
            description: Custom description for the validator
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if min_sentences < 1:
            raise ValidationError(
                "Minimum sentences must be at least 1",
                error_code="invalid_config",
                context={"min_sentences": min_sentences},
                suggestions=["Use a positive number for min_sentences"]
            )
        
        if not 0.0 <= max_repetition_ratio <= 1.0:
            raise ValidationError(
                "Max repetition ratio must be between 0.0 and 1.0",
                error_code="invalid_config",
                context={"max_repetition_ratio": max_repetition_ratio},
                suggestions=["Use a ratio between 0.0 and 1.0"]
            )
        
        if not 0.0 <= min_transition_ratio <= 1.0:
            raise ValidationError(
                "Min transition ratio must be between 0.0 and 1.0",
                error_code="invalid_config",
                context={"min_transition_ratio": min_transition_ratio},
                suggestions=["Use a ratio between 0.0 and 1.0"]
            )
        
        # Set default name and description
        if name is None:
            name = "coherence"
        
        if description is None:
            description = f"Validates text coherence and readability"
        
        super().__init__(name=name, description=description)
        
        self.min_sentences = min_sentences
        self.max_repetition_ratio = max_repetition_ratio
        self.min_transition_ratio = min_transition_ratio
        self.check_paragraphs = check_paragraphs
        self.strict = strict
        
        # Transition words and phrases
        self.transition_words = {
            "however", "therefore", "furthermore", "moreover", "additionally",
            "consequently", "nevertheless", "meanwhile", "subsequently", "thus",
            "hence", "accordingly", "likewise", "similarly", "conversely",
            "in contrast", "on the other hand", "for example", "for instance",
            "in addition", "as a result", "in conclusion", "to summarize",
            "first", "second", "third", "finally", "next", "then", "also",
            "besides", "indeed", "certainly", "obviously", "clearly",
        }
        
        logger.debug(
            f"Created CoherenceValidator",
            extra={
                "validator_name": self.name,
                "min_sentences": self.min_sentences,
                "max_repetition_ratio": self.max_repetition_ratio,
                "min_transition_ratio": self.min_transition_ratio,
                "check_paragraphs": self.check_paragraphs,
                "strict": self.strict,
            }
        )
    
    async def validate_async(self, thought: SifakaThought) -> ValidationResult:
        """Validate text coherence asynchronously.
        
        Args:
            thought: The SifakaThought to validate
            
        Returns:
            ValidationResult with coherence validation information
        """
        # Check if we have text to validate
        text = thought.current_text
        if not text:
            logger.debug(
                f"Coherence validation failed: no text",
                extra={"validator": self.name, "thought_id": thought.id}
            )
            return self.create_empty_text_result()
        
        with self.time_operation("coherence_validation") as timer:
            try:
                # Analyze text coherence
                analysis = await self._analyze_coherence(text)
                
                # Check coherence criteria
                issues = []
                suggestions = []
                violations = 0
                
                # Check sentence count
                if analysis["sentence_count"] < self.min_sentences:
                    violations += 1
                    issues.append(
                        f"Too few sentences: {analysis['sentence_count']} "
                        f"(minimum: {self.min_sentences})"
                    )
                    suggestions.append(f"Add more sentences to reach minimum of {self.min_sentences}")
                
                # Check repetition
                if analysis["repetition_ratio"] > self.max_repetition_ratio:
                    violations += 1
                    issues.append(
                        f"Excessive repetition: {analysis['repetition_ratio']:.2f} "
                        f"(maximum: {self.max_repetition_ratio:.2f})"
                    )
                    suggestions.append("Reduce repetitive words and phrases")
                
                # Check transitions
                if analysis["transition_ratio"] < self.min_transition_ratio:
                    violations += 1
                    issues.append(
                        f"Insufficient transitions: {analysis['transition_ratio']:.2f} "
                        f"(minimum: {self.min_transition_ratio:.2f})"
                    )
                    suggestions.append("Add transition words to improve flow")
                
                # Check paragraph structure if enabled
                if self.check_paragraphs and analysis["paragraph_count"] > 1:
                    if analysis["avg_paragraph_sentences"] < 2:
                        violations += 1
                        issues.append("Paragraphs too short (less than 2 sentences on average)")
                        suggestions.append("Combine short paragraphs or add more content")
                
                # Determine if validation passed
                passed = violations == 0
                
                # Calculate score
                if passed:
                    score = 1.0
                elif self.strict:
                    score = 0.0
                else:
                    # Calculate proportional score
                    score = self._calculate_coherence_score(analysis, violations)
                
                # Create result message
                if passed:
                    message = f"Coherence validation passed: {analysis['sentence_count']} sentences, good flow"
                else:
                    message = f"Coherence validation failed: {violations} issue(s)"
                
                # Get processing time from timer context
                processing_time = getattr(timer, 'duration_ms', 0.0)
                
                result = self.create_validation_result(
                    passed=passed,
                    message=message,
                    score=score,
                    issues=issues,
                    suggestions=suggestions,
                    metadata={
                        **analysis,
                        "min_sentences": self.min_sentences,
                        "max_repetition_ratio": self.max_repetition_ratio,
                        "min_transition_ratio": self.min_transition_ratio,
                        "check_paragraphs": self.check_paragraphs,
                        "violations": violations,
                        "strict_mode": self.strict,
                    },
                    processing_time_ms=processing_time,
                )
                
                logger.debug(
                    f"Coherence validation completed",
                    extra={
                        "validator": self.name,
                        "thought_id": thought.id,
                        "passed": passed,
                        "violations": violations,
                        "sentence_count": analysis["sentence_count"],
                        "repetition_ratio": analysis["repetition_ratio"],
                        "transition_ratio": analysis["transition_ratio"],
                        "score": score,
                    }
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    f"Coherence validation failed",
                    extra={
                        "validator": self.name,
                        "thought_id": thought.id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True
                )
                raise ValidationError(
                    f"Coherence validation failed: {str(e)}",
                    error_code="coherence_validation_error",
                    context={
                        "validator": self.name,
                        "text_length": len(text),
                        "error_type": type(e).__name__,
                    },
                    suggestions=[
                        "Check text format and content",
                        "Verify text contains valid sentences",
                        "Try with simpler text structure",
                    ]
                ) from e
    
    async def _analyze_coherence(self, text: str) -> Dict[str, Any]:
        """Analyze text coherence metrics."""
        def analyze():
            # Split into sentences and paragraphs
            sentences = self._split_sentences(text)
            paragraphs = text.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            # Count words
            words = text.lower().split()
            word_count = len(words)
            
            # Calculate repetition ratio
            if word_count > 0:
                unique_words = len(set(words))
                repetition_ratio = 1.0 - (unique_words / word_count)
            else:
                repetition_ratio = 0.0
            
            # Calculate transition ratio
            transition_count = sum(1 for word in words if word in self.transition_words)
            transition_ratio = transition_count / max(1, word_count)
            
            # Calculate sentence variety
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_sentence_length = sum(sentence_lengths) / max(1, len(sentence_lengths))
            sentence_length_variance = self._calculate_variance(sentence_lengths)
            
            # Calculate paragraph metrics
            paragraph_sentences = []
            for paragraph in paragraphs:
                para_sentences = self._split_sentences(paragraph)
                paragraph_sentences.append(len(para_sentences))
            
            avg_paragraph_sentences = sum(paragraph_sentences) / max(1, len(paragraph_sentences))
            
            return {
                "sentence_count": len(sentences),
                "paragraph_count": len(paragraphs),
                "word_count": word_count,
                "unique_words": unique_words if word_count > 0 else 0,
                "repetition_ratio": repetition_ratio,
                "transition_count": transition_count,
                "transition_ratio": transition_ratio,
                "avg_sentence_length": avg_sentence_length,
                "sentence_length_variance": sentence_length_variance,
                "avg_paragraph_sentences": avg_paragraph_sentences,
                "text_length": len(text),
            }
        
        # Run analysis in thread pool for CPU-bound work
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, analyze)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex."""
        # Simple sentence splitting - could be improved with NLTK
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_coherence_score(self, analysis: Dict[str, Any], violations: int) -> float:
        """Calculate a proportional coherence score."""
        # Base score starts at 1.0
        score = 1.0
        
        # Penalty for violations
        score -= violations * 0.2
        
        # Bonus for good metrics
        if analysis["transition_ratio"] > self.min_transition_ratio * 1.5:
            score += 0.1
        
        if analysis["repetition_ratio"] < self.max_repetition_ratio * 0.5:
            score += 0.1
        
        if analysis["sentence_length_variance"] > 10:  # Good sentence variety
            score += 0.05
        
        return max(0.1, min(1.0, score))


def create_coherence_validator(
    min_sentences: int = 2,
    max_repetition_ratio: float = 0.3,
    min_transition_ratio: float = 0.05,
    check_paragraphs: bool = True,
    strict: bool = False,
    name: Optional[str] = None,
) -> CoherenceValidator:
    """Create a coherence validator with the specified parameters.
    
    Args:
        min_sentences: Minimum number of sentences required
        max_repetition_ratio: Maximum allowed repetition ratio
        min_transition_ratio: Minimum required transition word ratio
        check_paragraphs: Whether to check paragraph structure
        strict: Whether to fail validation on any violation
        name: Custom name for the validator
        
    Returns:
        Configured CoherenceValidator instance
    """
    return CoherenceValidator(
        min_sentences=min_sentences,
        max_repetition_ratio=max_repetition_ratio,
        min_transition_ratio=min_transition_ratio,
        check_paragraphs=check_paragraphs,
        strict=strict,
        name=name,
    )
