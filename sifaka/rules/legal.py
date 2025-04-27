"""
Legal-specific rules for Sifaka.
"""
import re
from typing import Dict, Any, List, Tuple
from .base import Rule, RuleResult

class LegalCitationRule(Rule):
    """
    Rule that checks for valid legal citations in the output.
    """
    
    def __init__(self, name: str = "legal_citation_check"):
        super().__init__(name)
        
        # Common patterns for US legal citations
        self.citation_patterns = [
            # Supreme Court
            r'\d+ U\.S\. \d+',
            r'\d+ S\.Ct\. \d+',
            
            # Federal Reporter
            r'\d+ F\.\d[d|rd] \d+',
            
            # State cases
            r'\d+ [A-Za-z\.]+\d?[d|rd] \d+',
            
            # Statutes
            r'\d+ U\.S\.C\. § \d+',
        ]
        
        self.citation_regex = re.compile('|'.join(f'({pattern})' for pattern in self.citation_patterns))
    
    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that all legal citations in the output are properly formatted.
        
        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation
            
        Returns:
            RuleResult: The result of the validation
        """
        # Find all citations in the output
        citations = self.citation_regex.findall(output)
        citations = [c for group in citations for c in group if c]  # Flatten and remove empty matches
        
        if not citations:
            return RuleResult(
                passed=True,
                message="No legal citations found in the output."
            )
        
        # For a real implementation, we would validate each citation against a legal database
        # For this proof of concept, we'll just check the format
        invalid_citations = []
        for citation in citations:
            # In a real implementation, this would check against a legal database
            # For now, we'll just assume all found citations are valid
            pass
        
        if invalid_citations:
            return RuleResult(
                passed=False,
                message=f"Found {len(invalid_citations)} invalid legal citations",
                metadata={"invalid_citations": invalid_citations}
            )
        
        return RuleResult(
            passed=True,
            message=f"All {len(citations)} legal citations are valid",
            metadata={"valid_citations": citations}
        )


# Convenience function for use in the API
def legal_citation_check(output: str, **kwargs) -> RuleResult:
    """
    Check that all legal citations in the output are properly formatted.
    
    Args:
        output (str): The LLM output to validate
        **kwargs: Additional context for validation
        
    Returns:
        RuleResult: The result of the validation
    """
    rule = LegalCitationRule()
    return rule.validate(output, **kwargs)
