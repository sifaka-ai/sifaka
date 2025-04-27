"""
Content validation rules for Sifaka.
"""
from typing import List, Dict, Any, Optional
import re
from .base import Rule, RuleResult

class ProhibitedContentRule(Rule):
    """
    Rule that checks for prohibited content in the output.
    
    Args:
        prohibited_terms (List[str]): List of terms that should not appear in the output
        case_sensitive (bool): Whether the check should be case-sensitive
        name (Optional[str]): The name of the rule
    """
    
    def __init__(
        self, 
        prohibited_terms: List[str], 
        case_sensitive: bool = False,
        name: Optional[str] = None
    ):
        super().__init__(name or "prohibited_content")
        self.prohibited_terms = prohibited_terms
        self.case_sensitive = case_sensitive
    
    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output does not contain any prohibited terms.
        
        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation
            
        Returns:
            RuleResult: The result of the validation
        """
        check_output = output if self.case_sensitive else output.lower()
        found_terms = []
        
        for term in self.prohibited_terms:
            search_term = term if self.case_sensitive else term.lower()
            if search_term in check_output:
                found_terms.append(term)
        
        if found_terms:
            return RuleResult(
                passed=False,
                message=f"Output contains prohibited terms: {', '.join(found_terms)}",
                metadata={"found_terms": found_terms}
            )
        
        return RuleResult(
            passed=True,
            message="No prohibited terms found in the output"
        )


class ToneConsistencyRule(Rule):
    """
    Rule that checks for consistency in tone throughout the output.
    
    Args:
        expected_tone (str): The expected tone of the output (formal, informal, etc.)
        name (Optional[str]): The name of the rule
    """
    
    def __init__(self, expected_tone: str, name: Optional[str] = None):
        super().__init__(name or f"{expected_tone}_tone")
        self.expected_tone = expected_tone.lower()
        
        # Tone indicators (simplified for proof of concept)
        self.tone_indicators = {
            "formal": {
                "positive": ["therefore", "consequently", "furthermore", "thus", "hence"],
                "negative": ["yeah", "cool", "awesome", "btw", "gonna", "wanna"]
            },
            "informal": {
                "positive": ["yeah", "cool", "awesome", "btw", "gonna", "wanna"],
                "negative": ["therefore", "consequently", "furthermore", "thus", "hence"]
            }
        }
    
    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output maintains the expected tone.
        
        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation
            
        Returns:
            RuleResult: The result of the validation
        """
        if self.expected_tone not in self.tone_indicators:
            return RuleResult(
                passed=False,
                message=f"Unknown tone: {self.expected_tone}",
                metadata={"available_tones": list(self.tone_indicators.keys())}
            )
        
        output_lower = output.lower()
        
        # Check for positive indicators
        positive_indicators = []
        for term in self.tone_indicators[self.expected_tone]["positive"]:
            if term in output_lower:
                positive_indicators.append(term)
        
        # Check for negative indicators
        negative_indicators = []
        for term in self.tone_indicators[self.expected_tone]["negative"]:
            if term in output_lower:
                negative_indicators.append(term)
        
        # Simple scoring (in a real implementation, this would be more sophisticated)
        if len(negative_indicators) > len(positive_indicators):
            return RuleResult(
                passed=False,
                message=f"Output does not maintain {self.expected_tone} tone",
                metadata={
                    "positive_indicators": positive_indicators,
                    "negative_indicators": negative_indicators
                }
            )
        
        return RuleResult(
            passed=True,
            message=f"Output maintains {self.expected_tone} tone",
            metadata={
                "positive_indicators": positive_indicators,
                "negative_indicators": negative_indicators
            }
        )
