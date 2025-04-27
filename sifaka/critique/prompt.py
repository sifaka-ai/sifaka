"""
Prompt-based critique for Sifaka.
"""
from typing import Dict, Any, List, Optional
from .base import Critique
from ..models.base import ModelProvider

class PromptCritique(Critique):
    """
    A critique that uses the LLM itself to improve the output.
    
    Args:
        model (ModelProvider): The LLM provider to use for critique
        name (Optional[str]): The name of the critique
    """
    
    def __init__(self, model: ModelProvider, name: Optional[str] = None):
        super().__init__(name or "prompt_critique")
        self.model = model
    
    def improve(
        self, 
        output: str, 
        prompt: Optional[str] = None,
        rule_violations: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """
        Improve the output by asking the LLM to critique and revise it.
        
        Args:
            output (str): The LLM output to improve
            prompt (Optional[str]): The original prompt that generated the output
            rule_violations (Optional[List[Dict[str, Any]]]): List of rule violations
            **kwargs: Additional context for improvement
            
        Returns:
            str: The improved output
        """
        critique_prompt = self._build_critique_prompt(output, prompt, rule_violations)
        improved_output = self.model.generate(critique_prompt)
        return improved_output
    
    def _build_critique_prompt(
        self, 
        output: str, 
        prompt: Optional[str] = None,
        rule_violations: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build a prompt for the LLM to critique and revise the output.
        
        Args:
            output (str): The LLM output to improve
            prompt (Optional[str]): The original prompt that generated the output
            rule_violations (Optional[List[Dict[str, Any]]]): List of rule violations
            
        Returns:
            str: The critique prompt
        """
        critique_prompt = "You are a helpful assistant that improves text. "
        
        if prompt:
            critique_prompt += f"The original request was: \"{prompt}\"\n\n"
        
        critique_prompt += f"Here is the text to improve:\n\n{output}\n\n"
        
        if rule_violations:
            critique_prompt += "The text has the following issues that need to be fixed:\n"
            for violation in rule_violations:
                critique_prompt += f"- {violation['message']}\n"
            critique_prompt += "\n"
        
        critique_prompt += "Please provide an improved version of the text that addresses these issues. "
        critique_prompt += "Only return the improved text without explanations or additional commentary."
        
        return critique_prompt
