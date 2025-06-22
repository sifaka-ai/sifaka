"""Track and display actual feedback progression."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from ..core.models import SifakaResult


class FeedbackTracker:
    """Track how feedback evolves across iterations."""
    
    @staticmethod
    def track_feedback_progression(result: SifakaResult) -> Dict[str, Any]:
        """Track the actual progression of feedback and text changes."""
        progression = {
            "original_text": result.original_text,
            "final_text": result.final_text,
            "total_iterations": result.iteration,
            "feedback_evolution": [],
            "text_evolution": []
        }
        
        # Build text versions
        text_versions = [result.original_text]
        for gen in result.generations:
            text_versions.append(gen.text)
        
        # Track each iteration's feedback and resulting changes
        critiques = list(result.critiques)
        generations = list(result.generations)
        
        for i in range(result.iteration):
            iteration_data = {
                "iteration": i + 1,
                "critique": {},
                "changes": {},
                "text_snapshot": {}
            }
            
            # Get the critique for this iteration
            if i < len(critiques):
                critique = critiques[i]
                iteration_data["critique"] = {
                    "critic": critique.critic,
                    "feedback": critique.feedback,
                    "suggestions": critique.suggestions,
                    "confidence": critique.confidence,
                    "needs_improvement": critique.needs_improvement
                }
            
            # Show what changed in the text
            if i < len(text_versions) - 1:
                old_text = text_versions[i]
                new_text = text_versions[i + 1]
                
                iteration_data["changes"] = FeedbackTracker._analyze_changes(old_text, new_text)
                iteration_data["text_snapshot"] = {
                    "before": old_text[:200] + "..." if len(old_text) > 200 else old_text,
                    "after": new_text[:200] + "..." if len(new_text) > 200 else new_text
                }
            
            # Track which suggestions were actually implemented
            if i < len(generations) and generations[i].suggestion_implementation:
                impl = generations[i].suggestion_implementation
                iteration_data["implementation"] = {
                    "suggestions_given": impl.get("suggestions_given", []),
                    "suggestions_implemented": impl.get("suggestions_implemented", []),
                    "implementation_rate": impl.get("implementation_rate", 0)
                }
            
            progression["feedback_evolution"].append(iteration_data)
        
        # Add summary of how feedback evolved
        progression["feedback_summary"] = FeedbackTracker._summarize_feedback_evolution(
            progression["feedback_evolution"]
        )
        
        return progression
    
    @staticmethod
    def _analyze_changes(old_text: str, new_text: str) -> Dict[str, Any]:
        """Analyze what actually changed between text versions."""
        old_words = set(old_text.lower().split())
        new_words = set(new_text.lower().split())
        
        added_words = new_words - old_words
        removed_words = old_words - new_words
        
        # Find key additions (longer, meaningful words)
        key_additions = sorted([w for w in added_words if len(w) > 5], 
                             key=len, reverse=True)[:10]
        
        return {
            "length_change": len(new_text) - len(old_text),
            "word_count_change": len(new_text.split()) - len(old_text.split()),
            "key_concepts_added": key_additions,
            "expansion_ratio": len(new_text) / len(old_text) if old_text else 0,
            "major_revision": len(added_words) > len(new_words) * 0.3
        }
    
    @staticmethod
    def _summarize_feedback_evolution(iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize how feedback evolved across iterations."""
        if not iterations:
            return {}
        
        # Track feedback themes
        all_suggestions = []
        confidence_progression = []
        
        for iteration in iterations:
            if "critique" in iteration and iteration["critique"]:
                critique = iteration["critique"]
                all_suggestions.extend(critique.get("suggestions", []))
                confidence_progression.append(critique.get("confidence", 0))
        
        # Analyze suggestion patterns
        suggestion_themes = {
            "examples": sum(1 for s in all_suggestions if "example" in s.lower()),
            "details": sum(1 for s in all_suggestions if any(w in s.lower() for w in ["detail", "expand", "elaborate"])),
            "clarity": sum(1 for s in all_suggestions if any(w in s.lower() for w in ["clarity", "clear", "simplify"])),
            "evidence": sum(1 for s in all_suggestions if any(w in s.lower() for w in ["data", "statistic", "fact", "evidence"])),
            "structure": sum(1 for s in all_suggestions if any(w in s.lower() for w in ["structure", "organize", "reorder"]))
        }
        
        return {
            "total_suggestions": len(all_suggestions),
            "suggestion_themes": suggestion_themes,
            "confidence_trend": "increasing" if confidence_progression and confidence_progression[-1] > confidence_progression[0] else "stable",
            "average_confidence": sum(confidence_progression) / len(confidence_progression) if confidence_progression else 0
        }
    
    @staticmethod
    def format_feedback_narrative(progression: Dict[str, Any]) -> str:
        """Create a narrative view of how feedback evolved."""
        lines = []
        lines.append("FEEDBACK PROGRESSION NARRATIVE")
        lines.append("=" * 50)
        lines.append(f"\nOriginal text ({len(progression['original_text'])} chars):")
        lines.append(f"{progression['original_text'][:150]}...")
        
        for iteration in progression["feedback_evolution"]:
            lines.append(f"\n{'='*50}")
            lines.append(f"ITERATION {iteration['iteration']}")
            lines.append(f"{'='*50}")
            
            # Show the critique
            if iteration["critique"]:
                critique = iteration["critique"]
                lines.append(f"\nCritic: {critique['critic']}")
                lines.append(f"Confidence: {critique['confidence']}/10")
                lines.append(f"\nFeedback:")
                lines.append(f"  {critique['feedback']}")
                
                if critique["suggestions"]:
                    lines.append(f"\nSuggestions ({len(critique['suggestions'])}):")
                    for i, suggestion in enumerate(critique["suggestions"], 1):
                        lines.append(f"  {i}. {suggestion}")
            
            # Show what changed
            if iteration["changes"]:
                changes = iteration["changes"]
                lines.append(f"\nChanges Applied:")
                lines.append(f"  • Text grew {changes['expansion_ratio']:.1f}x ({changes['length_change']:+d} chars)")
                lines.append(f"  • Added {changes['word_count_change']:+d} words")
                if changes["key_concepts_added"]:
                    lines.append(f"  • Key concepts added: {', '.join(changes['key_concepts_added'][:5])}")
            
            # Show implementation tracking
            if "implementation" in iteration:
                impl = iteration["implementation"]
                lines.append(f"\nImplementation:")
                lines.append(f"  • {len(impl['suggestions_implemented'])}/{len(impl['suggestions_given'])} suggestions implemented")
                for implemented in impl["suggestions_implemented"]:
                    lines.append(f"    ✓ {implemented[:80]}...")
        
        # Summary
        if progression["feedback_summary"]:
            summary = progression["feedback_summary"]
            lines.append(f"\n{'='*50}")
            lines.append("SUMMARY")
            lines.append(f"{'='*50}")
            lines.append(f"Total suggestions across all iterations: {summary['total_suggestions']}")
            lines.append(f"Average confidence: {summary['average_confidence']:.1f}/10")
            lines.append(f"Confidence trend: {summary['confidence_trend']}")
            
            if summary["suggestion_themes"]:
                lines.append("\nSuggestion themes:")
                for theme, count in summary["suggestion_themes"].items():
                    if count > 0:
                        lines.append(f"  • {theme}: {count}")
        
        lines.append(f"\n{'='*50}")
        lines.append(f"Final text ({len(progression['final_text'])} chars):")
        lines.append(f"{progression['final_text'][:200]}...")
        
        return "\n".join(lines)