"""Text generation component of the Sifaka engine."""

from typing import Optional, Tuple

from ..models import SifakaResult
from ..llm_client import LLMManager, LLMClient
from ..constants import ROLE_SYSTEM, ROLE_USER


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
        self._client: Optional[LLMClient] = None
    
    @property
    def client(self) -> LLMClient:
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
            
        except Exception:
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
                    f"- {validation.validator}: {validation.details}"
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