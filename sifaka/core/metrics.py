"""Metrics for tracking text improvement quality."""

from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher
import re
from collections import Counter


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using SequenceMatcher."""
    return SequenceMatcher(None, text1, text2).ratio()


def calculate_improvement_metrics(old_text: str, new_text: str) -> Dict[str, Any]:
    """Calculate comprehensive metrics showing how text improved."""
    old_words = old_text.split()
    new_words = new_text.split()
    
    old_sentences = re.split(r'[.!?]+', old_text)
    new_sentences = re.split(r'[.!?]+', new_text)
    
    old_paragraphs = old_text.split('\n\n')
    new_paragraphs = new_text.split('\n\n')
    
    return {
        "length_change": len(new_text) - len(old_text),
        "length_ratio": len(new_text) / len(old_text) if len(old_text) > 0 else 0,
        "word_count_change": len(new_words) - len(old_words),
        "word_count_ratio": len(new_words) / len(old_words) if len(old_words) > 0 else 0,
        "sentence_count_change": len(new_sentences) - len(old_sentences),
        "paragraph_count_change": len(new_paragraphs) - len(old_paragraphs),
        "similarity": calculate_text_similarity(old_text, new_text),
        "avg_word_length_change": (
            sum(len(w) for w in new_words) / len(new_words) - 
            sum(len(w) for w in old_words) / len(old_words)
        ) if old_words and new_words else 0,
    }


def analyze_suggestion_implementation(
    suggestions: List[str], 
    old_text: str, 
    new_text: str
) -> Dict[str, Any]:
    """Analyze which suggestions were likely implemented."""
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


def calculate_readability_score(text: str) -> float:
    """Calculate a simple readability score (0-10)."""
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    if not words or not sentences:
        return 5.0  # Default middle score
    
    # Average words per sentence
    avg_words_per_sentence = len(words) / len(sentences)
    
    # Average word length
    avg_word_length = sum(len(w) for w in words) / len(words)
    
    # Simple readability formula
    # Ideal is around 15-20 words per sentence, 4-5 chars per word
    sentence_score = 10 - abs(avg_words_per_sentence - 17.5) * 0.3
    word_score = 10 - abs(avg_word_length - 4.5) * 1.5
    
    # Average the scores
    readability = (sentence_score + word_score) / 2
    
    # Ensure in range 0-10
    readability = max(0, min(10, readability))
    
    return round(readability, 1)


def calculate_information_density(text: str) -> float:
    """Calculate information density (unique concepts per 100 words)."""
    words = text.lower().split()
    
    # Filter out common words and short words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                   'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 
                   'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
                   'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
    
    meaningful_words = [w for w in words if len(w) > 3 and w not in common_words]
    unique_concepts = len(set(meaningful_words))
    
    if not words:
        return 0.0
    
    # Unique concepts per 100 words
    density = (unique_concepts / len(words)) * 100
    
    return round(density, 2)


def track_quality_progression(text_versions: List[str]) -> Dict[str, List[float]]:
    """Track quality metrics across text versions."""
    readability_scores = []
    complexity_scores = []
    information_density_scores = []
    
    for text in text_versions:
        readability_scores.append(calculate_readability_score(text))
        
        # Complexity as inverse of readability
        complexity_scores.append(10 - calculate_readability_score(text))
        
        information_density_scores.append(calculate_information_density(text))
    
    return {
        "readability_progression": readability_scores,
        "complexity_progression": complexity_scores,
        "information_density_progression": information_density_scores,
    }


def analyze_text_evolution(old_text: str, new_text: str) -> Dict[str, Any]:
    """Analyze how text evolved between versions."""
    # Find what was added/removed
    old_words = set(old_text.lower().split())
    new_words = set(new_text.lower().split())
    
    added_words = new_words - old_words
    removed_words = old_words - new_words
    
    # Key concepts added (longer, meaningful words)
    key_additions = [w for w in added_words if len(w) > 5]
    
    return {
        "words_added": len(added_words),
        "words_removed": len(removed_words),
        "key_concepts_added": key_additions[:10],  # Top 10
        "expansion_focused": len(added_words) > len(removed_words),
        "major_revision": calculate_text_similarity(old_text, new_text) < 0.5,
    }