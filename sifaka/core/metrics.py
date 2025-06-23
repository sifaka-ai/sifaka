"""Metrics for tracking text improvement quality.

This module contains only the metrics that are actually used in the codebase.
The analyze_suggestion_implementation function is used by SifakaResult to track
which critic suggestions were implemented in text generations.
"""

from typing import Dict, List, Any
import re


def analyze_suggestion_implementation(
    suggestions: List[str], 
    old_text: str, 
    new_text: str
) -> Dict[str, Any]:
    """Analyze which suggestions were likely implemented.
    
    This function examines critic suggestions and determines which ones
    appear to have been implemented by comparing old and new text.
    
    Args:
        suggestions: List of suggestions from critics
        old_text: Text before improvement
        new_text: Text after improvement
        
    Returns:
        Dictionary containing:
        - suggestions_given: All suggestions provided
        - suggestions_implemented: Suggestions that appear implemented
        - suggestions_not_implemented: Suggestions not implemented
        - implementation_rate: Ratio of implemented suggestions
        - implementation_count: Number of implemented suggestions
    """
    implemented = []
    not_implemented = []
    
    old_text_lower = old_text.lower()
    new_text_lower = new_text.lower()
    
    for suggestion in suggestions:
        # Extract key phrases from suggestion
        suggestion_lower = suggestion.lower()
        
        # Look for key action words and their associated content
        key_phrases = []
        
        # Common patterns in suggestions
        patterns = [
            r'add\s+(.+?)(?:\.|,|$)',
            r'include\s+(.+?)(?:\.|,|$)',
            r'provide\s+(.+?)(?:\.|,|$)',
            r'expand\s+on\s+(.+?)(?:\.|,|$)',
            r'mention\s+(.+?)(?:\.|,|$)',
            r'discuss\s+(.+?)(?:\.|,|$)',
            r'examples?\s+of\s+(.+?)(?:\.|,|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, suggestion_lower)
            key_phrases.extend(matches)
        
        # Check if key phrases appear more in new text
        implementation_score = 0
        for phrase in key_phrases:
            # Extract important words (3+ chars, not common)
            important_words = [w for w in phrase.split() 
                             if len(w) > 3 and w not in {'the', 'and', 'for', 'with'}]
            
            for word in important_words:
                old_count = old_text_lower.count(word)
                new_count = new_text_lower.count(word)
                if new_count > old_count:
                    implementation_score += 1
        
        # Consider implemented if score > 0
        if implementation_score > 0 or any(phrase in new_text_lower for phrase in key_phrases):
            implemented.append(suggestion)
        else:
            not_implemented.append(suggestion)
    
    return {
        "suggestions_given": suggestions,
        "suggestions_implemented": implemented,
        "suggestions_not_implemented": not_implemented,
        "implementation_rate": len(implemented) / len(suggestions) if suggestions else 0,
        "implementation_count": len(implemented),
    }