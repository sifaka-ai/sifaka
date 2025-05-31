#!/usr/bin/env python3
"""
Critic Formatters for Thought Visualization

This module provides specialized formatters for different critic types,
allowing for customized visualization of critic feedback and metadata.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseCriticFormatter(ABC):
    """Base class for critic formatters"""

    @abstractmethod
    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format critic details for HTML display"""
        pass

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    def _format_violations_and_suggestions(self, violations: List[str], suggestions: List[str]) -> str:
        """Common formatting for violations and suggestions"""
        html = ""
        
        if violations:
            html += f'''
            <div class="violations">
                <strong>⚠️ Issues Found:</strong>
                {"".join(f'<div class="suggestion">{self._escape_html(violation)}</div>' for violation in violations[:10])}
            </div>
            '''

        if suggestions:
            html += f'''
            <div class="suggestions">
                <strong>💡 Suggestions:</strong>
                {"".join(f'<div class="suggestion">{self._escape_html(suggestion)}</div>' for suggestion in suggestions[:5])}
            </div>
            '''
        
        return html

    def _format_feedback(self, feedback: str, max_length: int = 1500) -> str:
        """Common formatting for feedback text"""
        if not feedback:
            return ""
        
        return f'''
        <div class="feedback">
            <strong>📝 Feedback:</strong>
            <div class="feedback-content" style="white-space: pre-wrap; background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 5px; font-family: monospace; font-size: 0.9em;">{self._escape_html(feedback[:max_length])}{"..." if len(feedback) > max_length else ""}</div>
        </div>
        '''


class ConstitutionalCriticFormatter(BaseCriticFormatter):
    """Formatter for Constitutional Critic"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format Constitutional Critic details"""
        confidence = critic.get("confidence", 0.0)
        violations = critic.get("violations", [])
        suggestions = critic.get("suggestions", [])
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})

        html = f'<div><strong>Confidence:</strong> {confidence:.1f}</div>'

        # Add violations and suggestions
        html += self._format_violations_and_suggestions(violations, suggestions)

        # Add detailed feedback
        html += self._format_feedback(feedback)

        # Add constitutional-specific metadata
        html += self._format_constitutional_metadata(metadata)

        return html

    def _format_constitutional_metadata(self, metadata: dict) -> str:
        """Format constitutional critic specific metadata"""
        if not metadata:
            return ""

        html = '''
        <div class="metadata">
            <strong>📋 Constitutional Analysis:</strong><br>
        '''

        # Principle violations
        principle_violations = metadata.get("principle_violations", [])
        if principle_violations:
            html += f"<strong>⚖️ Principle Violations ({len(principle_violations)}):</strong><br>"
            for violation in principle_violations[:5]:  # Show first 5
                violation_desc = violation.get("description", "Unknown violation")
                severity = violation.get("severity", "medium")
                html += f'<div style="margin: 5px 0; padding: 5px; background: #ffebee; border-left: 3px solid #f44336; border-radius: 3px;">'
                html += f'<strong>Severity:</strong> {severity.title()}<br>'
                html += f'{self._escape_html(violation_desc)}'
                html += '</div>'

        # Principles evaluated
        principles_evaluated = metadata.get("principles_evaluated", 0)
        if principles_evaluated:
            html += f"Principles Evaluated: {principles_evaluated}<br>"

        # Strict mode
        strict_mode = metadata.get("strict_mode", False)
        html += f"Strict Mode: {'Yes' if strict_mode else 'No'}<br>"

        html += '</div>'
        return html


class SelfConsistencyCriticFormatter(BaseCriticFormatter):
    """Formatter for Self Consistency Critic"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format Self Consistency Critic details"""
        confidence = critic.get("confidence", 0.0)
        violations = critic.get("violations", [])
        suggestions = critic.get("suggestions", [])
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})

        html = f'<div><strong>Confidence:</strong> {confidence:.1f}</div>'

        # Add violations and suggestions
        html += self._format_violations_and_suggestions(violations, suggestions)

        # Add detailed feedback
        html += self._format_feedback(feedback)

        # Add self-consistency specific metadata
        html += self._format_self_consistency_metadata(metadata)

        return html

    def _format_self_consistency_metadata(self, metadata: dict) -> str:
        """Format self consistency specific metadata"""
        if not metadata:
            return ""

        html = '''
        <div class="metadata">
            <strong>📋 Consistency Analysis:</strong><br>
        '''

        # Iterations
        num_iterations = metadata.get("num_iterations", "N/A")
        html += f"Iterations: {num_iterations}<br>"

        # Consensus statistics
        consensus_stats = metadata.get("consensus_stats", {})
        if consensus_stats:
            stats = consensus_stats.get("stats", {})
            consensus_items = stats.get("consensus_items", "N/A")
            agreement_ratio = stats.get("agreement_ratio", 0)
            html += f"Consensus Items: {consensus_items}<br>"
            html += f"Agreement Ratio: {agreement_ratio:.1%}<br>"

        html += '</div>'
        return html


class ReflexionCriticFormatter(BaseCriticFormatter):
    """Formatter for Reflexion Critic"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format Reflexion Critic details"""
        confidence = critic.get("confidence", 0.0)
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})
        reflection = metadata.get("reflection", "")
        memory_size = metadata.get("memory_size", 0)
        trial_number = metadata.get("trial_number", 0)
        reflexion_memory = metadata.get("reflexion_memory", {})

        html = f'''
        <div><strong>Confidence:</strong> {confidence:.1f}</div>
        <div><strong>Trial Number:</strong> {trial_number}</div>
        <div><strong>Memory Size:</strong> {memory_size}</div>
        '''

        # Add critique feedback
        html += self._format_feedback(feedback, max_length=2000)

        # Add self-reflection
        if reflection:
            html += f'''
            <div class="reflection">
                <strong>🔄 Self-Reflection:</strong>
                <div class="feedback-content" style="white-space: pre-wrap; background: #fff3e0; padding: 10px; border-radius: 4px; margin-top: 5px; font-family: monospace; font-size: 0.9em; border-left: 4px solid #ff9800;">{self._escape_html(reflection[:2000])}{"..." if len(reflection) > 2000 else ""}</div>
            </div>
            '''

        # Add reflexion memory
        if reflexion_memory.get("sessions"):
            html += f'''
            <div class="reflexion-memory">
                <strong>🧠 Reflexion Memory:</strong>
                <div style="margin-top: 10px;">
                    {self._format_reflexion_sessions(reflexion_memory.get("sessions", []))}
                </div>
            </div>
            '''

        return html

    def _format_reflexion_sessions(self, sessions: list) -> str:
        """Format reflexion memory sessions"""
        if not sessions:
            return "<div>No memory sessions</div>"

        html = ""
        for session in sessions:
            trial_num = session.get("trial_number", "Unknown")
            summary = session.get("summary", "No summary")
            timestamp = session.get("timestamp", "")

            html += f'''
            <div class="memory-session" style="background: #f0f8ff; padding: 8px; margin: 5px 0; border-radius: 4px; border-left: 3px solid #2196f3;">
                <strong>Trial {trial_num}:</strong> {self._escape_html(summary)}<br>
                <small style="color: #666;">Timestamp: {timestamp}</small>
            </div>
            '''

        return html


class MetaRewardingCriticFormatter(BaseCriticFormatter):
    """Formatter for Meta Rewarding Critic"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format Meta Rewarding Critic details"""
        confidence = critic.get("confidence", 0.0)
        violations = critic.get("violations", [])
        suggestions = critic.get("suggestions", [])
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})

        html = f'<div><strong>Confidence:</strong> {confidence:.1f}</div>'

        # Add violations and suggestions
        html += self._format_violations_and_suggestions(violations, suggestions)

        # Add detailed feedback
        html += self._format_feedback(feedback)

        # Add meta-rewarding specific metadata
        html += self._format_meta_rewarding_metadata(metadata)

        return html

    def _format_meta_rewarding_metadata(self, metadata: dict) -> str:
        """Format meta rewarding specific metadata"""
        if not metadata:
            return ""

        html = '''
        <div class="metadata">
            <strong>📋 Meta-Rewarding Analysis:</strong><br>
        '''

        # Initial score
        initial_score = metadata.get("initial_score", "N/A")
        html += f"Initial Score: {initial_score}<br>"

        # Judgment criteria
        judgment_criteria = metadata.get("judgment_criteria", [])
        if judgment_criteria:
            html += f"Judgment Criteria: {', '.join(judgment_criteria)}<br>"

        # Meta judgment criteria
        meta_judgment_criteria = metadata.get("meta_judgment_criteria", [])
        if meta_judgment_criteria:
            html += f"Meta-Judgment Criteria: {', '.join(meta_judgment_criteria)}<br>"

        # Base critic used
        base_critic_used = metadata.get("base_critic_used", "N/A")
        html += f"Base Critic Used: {base_critic_used}<br>"

        # Meta judge model
        meta_judge_model = metadata.get("meta_judge_model", "N/A")
        html += f"Meta Judge Model: {meta_judge_model}<br>"

        html += '</div>'
        return html


class GenericCriticFormatter(BaseCriticFormatter):
    """Generic formatter for unknown critic types"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format generic critic details"""
        confidence = critic.get("confidence", 0.0)
        violations = critic.get("violations", [])
        suggestions = critic.get("suggestions", [])
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})

        html = f'<div><strong>Confidence:</strong> {confidence:.1f}</div>'

        # Add violations and suggestions
        html += self._format_violations_and_suggestions(violations, suggestions)

        # Add detailed feedback
        html += self._format_feedback(feedback)

        # Add generic metadata
        html += self._format_generic_metadata(metadata)

        return html

    def _format_generic_metadata(self, metadata: dict) -> str:
        """Format generic metadata"""
        if not metadata:
            return ""

        html = '''
        <div class="metadata">
            <strong>📋 Metadata:</strong><br>
        '''

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                html += f"{key}: {value}<br>"
            elif isinstance(value, list) and len(value) <= 5:
                html += f"{key}: {', '.join(str(v) for v in value)}<br>"
            elif isinstance(value, dict) and len(value) <= 3:
                html += f"{key}: {str(value)}<br>"

        html += '</div>'
        return html


class CriticFormatterFactory:
    """Factory for creating appropriate critic formatters"""

    _formatters = {
        "ConstitutionalCritic": ConstitutionalCriticFormatter,
        "SelfConsistencyCritic": SelfConsistencyCriticFormatter,
        "ReflexionCritic": ReflexionCriticFormatter,
        "MetaRewardingCritic": MetaRewardingCriticFormatter,
    }

    @classmethod
    def get_formatter(cls, critic_name: str) -> BaseCriticFormatter:
        """Get appropriate formatter for critic type"""
        formatter_class = cls._formatters.get(critic_name, GenericCriticFormatter)
        return formatter_class()

    @classmethod
    def register_formatter(cls, critic_name: str, formatter_class: type):
        """Register a new formatter for a critic type"""
        cls._formatters[critic_name] = formatter_class
