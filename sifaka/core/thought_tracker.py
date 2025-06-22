"""Enhanced thought tracking for complete transparency."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from ..core.models import SifakaResult
from ..core.metrics import calculate_text_similarity, analyze_text_evolution


class ThoughtTracker:
    """Track and format the complete thought process."""
    
    @staticmethod
    def create_iteration_thoughts(
        result: SifakaResult,
        critic_name: str,
        system_prompts: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create detailed thought tracking for each iteration."""
        thoughts = {
            "metadata": {
                "critic": critic_name,
                "timestamp": datetime.now().isoformat(),
                "total_iterations": result.iteration,
            },
            "system_prompts": system_prompts,
            "iterations": [],
            "quality_progression": result.get_quality_progression(),
            "overall_metrics": {},
        }
        
        # Get all data as lists
        critiques = list(result.critiques)
        generations = list(result.generations)
        
        # Build text versions
        text_versions = [result.original_text]
        for gen in generations:
            text_versions.append(gen.text)
        
        # Track each iteration
        for i in range(result.iteration):
            iter_num = i + 1
            
            # Get data for this iteration
            critique = critiques[i] if i < len(critiques) else None
            generation = generations[i] if i < len(generations) else None
            
            text_before = text_versions[i] if i < len(text_versions) else result.original_text
            text_after = text_versions[i+1] if i+1 < len(text_versions) else text_before
            
            iteration_thought = {
                "iteration_number": iter_num,
                "critique_phase": {},
                "improvement_phase": {},
                "results": {},
            }
            
            # Critique phase
            if critique:
                iteration_thought["critique_phase"] = {
                    "critic": critique.critic,
                    "feedback": critique.feedback,
                    "suggestions": critique.suggestions,
                    "confidence": critique.confidence,
                    "needs_improvement": critique.needs_improvement,
                    "timestamp": critique.timestamp.isoformat() if hasattr(critique, 'timestamp') else None,
                }
            
            # Improvement phase
            if generation:
                iteration_thought["improvement_phase"] = {
                    "prompt_used": generation.prompt,
                    "model": generation.model,
                    "improvement_metrics": generation.improvement_metrics,
                    "suggestion_implementation": generation.suggestion_implementation,
                    "quality_indicators": generation.quality_indicators,
                }
            
            # Results
            iteration_thought["results"] = {
                "text_changed": text_before != text_after,
                "similarity_to_previous": calculate_text_similarity(text_before, text_after),
                "text_evolution": analyze_text_evolution(text_before, text_after),
                "text_preview": {
                    "before": text_before[:150] + "..." if len(text_before) > 150 else text_before,
                    "after": text_after[:150] + "..." if len(text_after) > 150 else text_after,
                }
            }
            
            thoughts["iterations"].append(iteration_thought)
        
        # Calculate overall metrics
        if text_versions:
            thoughts["overall_metrics"] = {
                "total_improvement": analyze_text_evolution(result.original_text, result.final_text),
                "similarity_to_original": calculate_text_similarity(result.original_text, result.final_text),
                "text_growth": {
                    "length_ratio": len(result.final_text) / len(result.original_text) if result.original_text else 0,
                    "word_count_ratio": len(result.final_text.split()) / len(result.original_text.split()) if result.original_text.split() else 0,
                },
                "suggestions_summary": ThoughtTracker._analyze_suggestions(critiques, generations),
            }
        
        return thoughts
    
    @staticmethod
    def _analyze_suggestions(critiques: List[Any], generations: List[Any]) -> Dict[str, Any]:
        """Analyze suggestion patterns across iterations."""
        all_suggestions = []
        implementation_rates = []
        
        for critique in critiques:
            if critique.suggestions:
                all_suggestions.extend(critique.suggestions)
        
        for gen in generations:
            if gen.suggestion_implementation:
                implementation_rates.append(gen.suggestion_implementation.get("implementation_rate", 0))
        
        return {
            "total_suggestions": len(all_suggestions),
            "unique_suggestions": len(set(all_suggestions)),
            "average_implementation_rate": sum(implementation_rates) / len(implementation_rates) if implementation_rates else 0,
            "suggestion_categories": ThoughtTracker._categorize_suggestions(all_suggestions),
        }
    
    @staticmethod
    def _categorize_suggestions(suggestions: List[str]) -> Dict[str, int]:
        """Categorize suggestions by type."""
        categories = {
            "add_examples": 0,
            "add_details": 0,
            "improve_clarity": 0,
            "add_data": 0,
            "restructure": 0,
            "other": 0,
        }
        
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            if "example" in suggestion_lower:
                categories["add_examples"] += 1
            elif any(word in suggestion_lower for word in ["detail", "expand", "elaborate"]):
                categories["add_details"] += 1
            elif any(word in suggestion_lower for word in ["clarity", "clear", "simplify"]):
                categories["improve_clarity"] += 1
            elif any(word in suggestion_lower for word in ["data", "statistic", "fact", "evidence"]):
                categories["add_data"] += 1
            elif any(word in suggestion_lower for word in ["structure", "organize", "reorder"]):
                categories["restructure"] += 1
            else:
                categories["other"] += 1
        
        return {k: v for k, v in categories.items() if v > 0}
    
    @staticmethod
    def create_comparative_thoughts(
        results: Dict[str, SifakaResult],
        system_prompts: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create comparative analysis across multiple critics."""
        comparative = {
            "timestamp": datetime.now().isoformat(),
            "critics_compared": list(results.keys()),
            "system_prompts": system_prompts,
            "individual_performance": {},
            "comparative_analysis": {},
        }
        
        # Analyze each critic
        for critic_name, result in results.items():
            if isinstance(result, dict) and "error" in result:
                comparative["individual_performance"][critic_name] = {"error": result["error"]}
                continue
            
            quality_progression = result.get_quality_progression()
            
            comparative["individual_performance"][critic_name] = {
                "iterations": result.iteration,
                "final_similarity": calculate_text_similarity(result.original_text, result.final_text),
                "text_growth": len(result.final_text) / len(result.original_text) if result.original_text else 0,
                "quality_improvement": {
                    "readability_change": (
                        quality_progression["readability_progression"][-1] - 
                        quality_progression["readability_progression"][0]
                    ) if quality_progression["readability_progression"] else 0,
                    "density_change": (
                        quality_progression["information_density_progression"][-1] - 
                        quality_progression["information_density_progression"][0]
                    ) if quality_progression["information_density_progression"] else 0,
                },
                "processing_time": result.processing_time,
            }
        
        # Comparative analysis
        valid_results = {k: v for k, v in comparative["individual_performance"].items() if "error" not in v}
        
        if valid_results:
            comparative["comparative_analysis"] = {
                "most_transformative": max(valid_results.items(), key=lambda x: 1 - x[1]["final_similarity"])[0],
                "best_growth": max(valid_results.items(), key=lambda x: x[1]["text_growth"])[0],
                "best_readability": max(
                    valid_results.items(), 
                    key=lambda x: x[1]["quality_improvement"]["readability_change"]
                )[0],
                "fastest": min(valid_results.items(), key=lambda x: x[1]["processing_time"])[0],
            }
        
        return comparative