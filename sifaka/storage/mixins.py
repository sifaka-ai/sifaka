"""Mixins for common storage functionality."""

from typing import List, Dict, Any
from ..core.models import SifakaResult


class SearchMixin:
    """Mixin providing common search functionality for storage backends."""
    
    def _build_searchable_text(self, result: SifakaResult) -> str:
        """Build searchable text from a SifakaResult.
        
        Args:
            result: The result to extract text from
            
        Returns:
            Combined searchable text
        """
        # Start with original and final text
        parts = [result.original_text, result.final_text]
        
        # Add all generation texts
        for gen in result.generations:
            parts.append(gen.text)
        
        # Add critique feedback and suggestions
        for critique in result.critiques:
            parts.append(critique.feedback)
            parts.extend(critique.suggestions)
        
        # Add validation details
        for validation in result.validations:
            parts.append(validation.details)
        
        # Join all parts with space
        return " ".join(filter(None, parts))
    
    def _text_matches_query(self, text: str, query: str, case_sensitive: bool = False) -> bool:
        """Check if text matches the search query.
        
        Args:
            text: The text to search in
            query: The search query
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            True if query is found in text
        """
        if not case_sensitive:
            text = text.lower()
            query = query.lower()
        
        return query in text
    
    def _rank_search_results(
        self, 
        results: List[tuple[str, float]], 
        limit: int
    ) -> List[str]:
        """Rank and limit search results.
        
        Args:
            results: List of (result_id, score) tuples
            limit: Maximum number of results to return
            
        Returns:
            List of result IDs sorted by score
        """
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Extract IDs and limit
        return [result_id for result_id, _ in results[:limit]]
    
    def _calculate_relevance_score(
        self, 
        result: SifakaResult, 
        query: str,
        searchable_text: str
    ) -> float:
        """Calculate relevance score for a search result.
        
        Args:
            result: The SifakaResult being scored
            query: The search query
            searchable_text: Pre-built searchable text
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        query_lower = query.lower()
        text_lower = searchable_text.lower()
        
        # Base score from occurrence count
        occurrence_count = text_lower.count(query_lower)
        base_score = min(occurrence_count * 0.1, 0.5)
        
        # Boost if query appears in original or final text
        if query_lower in result.original_text.lower():
            base_score += 0.2
        if query_lower in result.final_text.lower():
            base_score += 0.3
        
        # Small boost for recent results
        if hasattr(result, 'updated_at'):
            # This is a simple heuristic - can be improved with proper time-based scoring
            base_score += 0.05
        
        return min(base_score, 1.0)


class MetadataSearchMixin(SearchMixin):
    """Extended search functionality with metadata support."""
    
    def _search_by_metadata(
        self,
        results: Dict[str, SifakaResult],
        metadata_filters: Dict[str, Any],
        limit: int = 10
    ) -> List[str]:
        """Search results by metadata fields.
        
        Args:
            results: Dictionary of result_id -> SifakaResult
            metadata_filters: Metadata fields to filter by
            limit: Maximum number of results
            
        Returns:
            List of matching result IDs
        """
        matches = []
        
        for result_id, result in results.items():
            if self._matches_metadata(result, metadata_filters):
                matches.append(result_id)
                
            if len(matches) >= limit:
                break
        
        return matches
    
    def _matches_metadata(
        self,
        result: SifakaResult,
        filters: Dict[str, Any]
    ) -> bool:
        """Check if result matches metadata filters.
        
        Args:
            result: The result to check
            filters: Metadata filters to apply
            
        Returns:
            True if all filters match
        """
        for key, value in filters.items():
            if key == "iteration":
                if result.iteration != value:
                    return False
            elif key == "num_critiques":
                if len(result.critiques) != value:
                    return False
            elif key == "has_validations":
                if bool(result.validations) != value:
                    return False
            elif key == "min_processing_time":
                if result.processing_time < value:
                    return False
            elif key == "max_processing_time":
                if result.processing_time > value:
                    return False
        
        return True