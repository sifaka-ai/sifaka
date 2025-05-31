#!/usr/bin/env python3
"""
Critic Formatters for Thought Visualization

This module provides specialized formatters for different critic types,
allowing for customized visualization of critic feedback and metadata.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseCriticFormatter(ABC):
    """Base class for critic formatters"""

    @abstractmethod
    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format critic details for HTML display"""

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    def _format_violations_and_suggestions(
        self, violations: List[str], suggestions: List[str]
    ) -> str:
        """Common formatting for violations and suggestions"""
        html = ""

        if violations:
            html += f"""
            <div class="violations">
                <strong>⚠️ Issues Found:</strong>
                {"".join(f'<div class="suggestion">{self._escape_html(violation)}</div>' for violation in violations[:10])}
            </div>
            """

        if suggestions:
            html += f"""
            <div class="suggestions">
                <strong>💡 Suggestions:</strong>
                {"".join(f'<div class="suggestion">{self._escape_html(suggestion)}</div>' for suggestion in suggestions[:5])}
            </div>
            """

        return html

    def _format_feedback(self, feedback: str, max_length: int = 1500) -> str:
        """Common formatting for feedback text"""
        if not feedback:
            return ""

        return f"""
        <div class="feedback">
            <strong>📝 Feedback:</strong>
            <div class="feedback-content" style="white-space: pre-wrap; background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 5px; font-family: monospace; font-size: 0.9em;">{self._escape_html(feedback[:max_length])}{"..." if len(feedback) > max_length else ""}</div>
        </div>
        """


class ConstitutionalCriticFormatter(BaseCriticFormatter):
    """Formatter for Constitutional Critic"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format Constitutional Critic details"""
        confidence = critic.get("confidence", 0.0)
        violations = critic.get("violations", [])
        suggestions = critic.get("suggestions", [])
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})

        html = f"<div><strong>Confidence:</strong> {confidence:.1f}</div>"

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

        html = """
        <div class="metadata">
            <strong>📋 Constitutional Analysis:</strong><br>
        """

        # Principle violations
        principle_violations = metadata.get("principle_violations", [])
        if principle_violations:
            html += f"<strong>⚖️ Principle Violations ({len(principle_violations)}):</strong><br>"
            for violation in principle_violations[:5]:  # Show first 5
                violation_desc = violation.get("description", "Unknown violation")
                severity = violation.get("severity", "medium")
                html += '<div style="margin: 5px 0; padding: 5px; background: #ffebee; border-left: 3px solid #f44336; border-radius: 3px;">'
                html += f"<strong>Severity:</strong> {severity.title()}<br>"
                html += f"{self._escape_html(violation_desc)}"
                html += "</div>"

        # Principles evaluated
        principles_evaluated = metadata.get("principles_evaluated", 0)
        if principles_evaluated:
            html += f"Principles Evaluated: {principles_evaluated}<br>"

        # Strict mode
        strict_mode = metadata.get("strict_mode", False)
        html += f"Strict Mode: {'Yes' if strict_mode else 'No'}<br>"

        html += "</div>"
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

        html = f"<div><strong>Confidence:</strong> {confidence:.1f}</div>"

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

        html = """
        <div class="metadata">
            <strong>📋 Consistency Analysis:</strong><br>
        """

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

        html += "</div>"
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

        html = f"""
        <div><strong>Confidence:</strong> {confidence:.1f}</div>
        <div><strong>Trial Number:</strong> {trial_number}</div>
        <div><strong>Memory Size:</strong> {memory_size}</div>
        """

        # Add critique feedback
        html += self._format_feedback(feedback, max_length=2000)

        # Add self-reflection
        if reflection:
            html += f"""
            <div class="reflection">
                <strong>🔄 Self-Reflection:</strong>
                <div class="feedback-content" style="white-space: pre-wrap; background: #fff3e0; padding: 10px; border-radius: 4px; margin-top: 5px; font-family: monospace; font-size: 0.9em; border-left: 4px solid #ff9800;">{self._escape_html(reflection[:2000])}{"..." if len(reflection) > 2000 else ""}</div>
            </div>
            """

        # Add reflexion memory
        if reflexion_memory.get("sessions"):
            html += f"""
            <div class="reflexion-memory">
                <strong>🧠 Reflexion Memory:</strong>
                <div style="margin-top: 10px;">
                    {self._format_reflexion_sessions(reflexion_memory.get("sessions", []))}
                </div>
            </div>
            """

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

            html += f"""
            <div class="memory-session" style="background: #f0f8ff; padding: 8px; margin: 5px 0; border-radius: 4px; border-left: 3px solid #2196f3;">
                <strong>Trial {trial_num}:</strong> {self._escape_html(summary)}<br>
                <small style="color: #666;">Timestamp: {timestamp}</small>
            </div>
            """

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

        html = f"<div><strong>Confidence:</strong> {confidence:.1f}</div>"

        # Add violations and suggestions
        html += self._format_violations_and_suggestions(violations, suggestions)

        # Add detailed feedback
        html += self._format_feedback(feedback)

        # Add meta-rewarding specific metadata
        html += self._format_meta_rewarding_metadata(metadata)

        return html

    def _format_meta_rewarding_metadata(self, metadata: dict) -> str:
        """Format meta rewarding specific metadata with enhanced visualization"""
        if not metadata:
            return ""

        html = """
        <div class="metadata">
            <strong>🎯 Meta-Rewarding Analysis:</strong><br>
        """

        # Score information prominently displayed
        initial_score = metadata.get("initial_score", "N/A")
        html += f"""
        <div style="background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #2196f3;">
            <strong>📊 Initial Score: {initial_score}/10</strong>
        </div>
        """

        # Judgment criteria
        judgment_criteria = metadata.get("judgment_criteria", [])
        if judgment_criteria:
            html += f"""
            <div style="margin: 10px 0;">
                <strong>📋 Judgment Criteria:</strong><br>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    {"".join(f"<li>{criterion}</li>" for criterion in judgment_criteria)}
                </ul>
            </div>
            """

        # Meta judgment criteria
        meta_judgment_criteria = metadata.get("meta_judgment_criteria", [])
        if meta_judgment_criteria:
            html += f"""
            <div style="margin: 10px 0;">
                <strong>🔍 Meta-Judgment Criteria:</strong><br>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    {"".join(f"<li>{criterion}</li>" for criterion in meta_judgment_criteria)}
                </ul>
            </div>
            """

        # Initial judgment section (expandable)
        initial_judgment = metadata.get("initial_judgment", "")
        if initial_judgment:
            html += self._format_expandable_section(
                "📝 Initial Judgment", initial_judgment, "initial-judgment", "#fff3e0", "#ff9800"
            )

        # Meta judgment section (expandable)
        meta_judgment = metadata.get("meta_judgment", "")
        if meta_judgment:
            html += self._format_expandable_section(
                "🔄 Meta-Judgment", meta_judgment, "meta-judgment", "#f3e5f5", "#9c27b0"
            )

        # Technical details
        base_critic_used = metadata.get("base_critic_used", "N/A")
        meta_judge_model = metadata.get("meta_judge_model", "N/A")

        html += f"""
        <div style="margin-top: 15px; padding: 8px; background: #f5f5f5; border-radius: 4px; font-size: 0.9em;">
            <strong>🔧 Technical Details:</strong><br>
            Base Critic Used: {base_critic_used}<br>
            Meta Judge Model: {meta_judge_model}
        </div>
        """

        html += "</div>"
        return html

    def _format_expandable_section(
        self, title: str, content: str, section_id: str, bg_color: str, border_color: str
    ) -> str:
        """Format an expandable section for detailed content"""
        if not content:
            return ""

        # Generate unique ID for this section
        unique_id = f"{section_id}-{hash(content) % 10000}"

        return f"""
        <div style="margin: 10px 0;">
            <div style="cursor: pointer; padding: 8px; background: {bg_color}; border-left: 4px solid {border_color}; border-radius: 4px;"
                 onclick="toggleSection('{unique_id}')">
                <strong>{title}</strong> <span id="{unique_id}-toggle">▼</span>
            </div>
            <div id="{unique_id}" style="display: none; padding: 10px; background: #fafafa; border: 1px solid #ddd; border-top: none; border-radius: 0 0 4px 4px;">
                <pre style="white-space: pre-wrap; font-family: monospace; font-size: 0.85em; margin: 0; line-height: 1.4;">{self._escape_html(content)}</pre>
            </div>
        </div>
        """


class GenericCriticFormatter(BaseCriticFormatter):
    """Generic formatter for unknown critic types"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format generic critic details"""
        confidence = critic.get("confidence", 0.0)
        violations = critic.get("violations", [])
        suggestions = critic.get("suggestions", [])
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})

        html = f"<div><strong>Confidence:</strong> {confidence:.1f}</div>"

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

        html = """
        <div class="metadata">
            <strong>📋 Metadata:</strong><br>
        """

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                html += f"{key}: {value}<br>"
            elif isinstance(value, list) and len(value) <= 5:
                html += f"{key}: {', '.join(str(v) for v in value)}<br>"
            elif isinstance(value, dict) and len(value) <= 3:
                html += f"{key}: {str(value)}<br>"

        html += "</div>"
        return html


class NCriticsCriticFormatter(BaseCriticFormatter):
    """Formatter for N Critics Critic (Ensemble Critic)"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format N Critics Critic details"""
        confidence = critic.get("confidence", 0.0)
        violations = critic.get("violations", [])
        suggestions = critic.get("suggestions", [])
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})

        html = f"<div><strong>Confidence:</strong> {confidence:.1f}</div>"

        # Add violations and suggestions
        html += self._format_violations_and_suggestions(violations, suggestions)

        # Add detailed feedback
        html += self._format_feedback(feedback)

        # Add N Critics specific metadata
        html += self._format_n_critics_metadata(metadata)

        return html

    def _format_n_critics_metadata(self, metadata: dict) -> str:
        """Format N Critics specific metadata with ensemble visualization"""
        if not metadata:
            return ""

        html = """
        <div class="metadata">
            <strong>🎯 N Critics Ensemble Analysis:</strong><br>
        """

        # Aggregated score information prominently displayed
        aggregated_score = metadata.get("aggregated_score", "N/A")
        improvement_threshold = metadata.get("improvement_threshold", "N/A")
        num_critics = metadata.get("num_critics", "N/A")

        html += f"""
        <div style="background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #2196f3;">
            <strong>📊 Ensemble Score: {aggregated_score:.1f}/10</strong><br>
            <strong>🎯 Improvement Threshold: {improvement_threshold}</strong><br>
            <strong>👥 Number of Critics: {num_critics}</strong>
        </div>
        """

        # Individual critic feedback
        critic_feedback = metadata.get("critic_feedback", [])
        if critic_feedback:
            html += f"""
            <div style="margin: 10px 0;">
                <strong>👥 Individual Critic Feedback ({len(critic_feedback)} critics):</strong><br>
            """

            for i, individual_critic in enumerate(critic_feedback):
                role = individual_critic.get("role", "Unknown Critic")
                score = individual_critic.get("score", "N/A")
                needs_improvement = individual_critic.get("needs_improvement", False)
                issues = individual_critic.get("issues", [])
                suggestions = individual_critic.get("suggestions", [])
                critique = individual_critic.get("critique", "")

                status_icon = "❌" if needs_improvement else "✅"
                status_color = "#ffebee" if needs_improvement else "#e8f5e8"
                border_color = "#f44336" if needs_improvement else "#4caf50"

                html += f"""
                <div style="margin: 10px 0; padding: 12px; background: {status_color}; border-left: 4px solid {border_color}; border-radius: 4px;">
                    <div style="margin-bottom: 8px;">
                        <strong>{status_icon} {self._escape_html(role)}</strong>
                        <span style="float: right; font-weight: bold; color: #2196f3;">Score: {score}/10</span>
                    </div>
                """

                # Issues
                if issues:
                    html += f"""
                    <div style="margin: 8px 0;">
                        <strong>⚠️ Issues ({len(issues)}):</strong>
                        <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                            {"".join(f"<li>{self._escape_html(issue)}</li>" for issue in issues[:3])}
                            {f"<li><em>... and {len(issues) - 3} more</em></li>" if len(issues) > 3 else ""}
                        </ul>
                    </div>
                    """

                # Suggestions
                if suggestions:
                    html += f"""
                    <div style="margin: 8px 0;">
                        <strong>💡 Suggestions ({len(suggestions)}):</strong>
                        <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                            {"".join(f"<li>{self._escape_html(suggestion)}</li>" for suggestion in suggestions[:3])}
                            {f"<li><em>... and {len(suggestions) - 3} more</em></li>" if len(suggestions) > 3 else ""}
                        </ul>
                    </div>
                    """

                # Detailed critique (expandable)
                if critique:
                    unique_id = f"critique-{i}-{hash(critique) % 10000}"
                    html += f"""
                    <div style="margin: 8px 0;">
                        <div style="cursor: pointer; padding: 6px; background: #f5f5f5; border-radius: 3px; font-size: 0.9em;"
                             onclick="toggleSection('{unique_id}')">
                            <strong>📝 Detailed Critique</strong> <span id="{unique_id}-toggle">▼</span>
                        </div>
                        <div id="{unique_id}" style="display: none; padding: 8px; background: #fafafa; border: 1px solid #ddd; border-top: none; border-radius: 0 0 3px 3px;">
                            <pre style="white-space: pre-wrap; font-family: monospace; font-size: 0.8em; margin: 0; line-height: 1.3;">{self._escape_html(critique[:1500])}{"..." if len(critique) > 1500 else ""}</pre>
                        </div>
                    </div>
                    """

                html += "</div>"

            html += "</div>"

        html += "</div>"
        return html


class PromptCriticFormatter(BaseCriticFormatter):
    """Formatter for Prompt Critic"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format Prompt Critic details"""
        confidence = critic.get("confidence", 0.0)
        violations = critic.get("violations", [])
        suggestions = critic.get("suggestions", [])
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})

        html = f"<div><strong>Confidence:</strong> {confidence:.1f}</div>"

        # Add violations and suggestions
        html += self._format_violations_and_suggestions(violations, suggestions)

        # Add detailed feedback
        html += self._format_feedback(feedback)

        # Add prompt-specific metadata
        html += self._format_prompt_metadata(metadata)

        return html

    def _format_prompt_metadata(self, metadata: dict) -> str:
        """Format Prompt Critic specific metadata"""
        if not metadata:
            return ""

        html = """
        <div class="metadata">
            <strong>📝 Prompt Critic Configuration:</strong><br>
        """

        # Evaluation criteria
        criteria = metadata.get("criteria", [])
        if criteria:
            html += f"""
            <div style="margin: 10px 0;">
                <strong>🎯 Evaluation Criteria ({len(criteria)}):</strong>
                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                    {"".join(f"<li>{self._escape_html(criterion)}</li>" for criterion in criteria)}
                </ul>
            </div>
            """

        # System prompt
        system_prompt = metadata.get("system_prompt", "")
        if system_prompt:
            unique_id = f"system-prompt-{hash(system_prompt) % 10000}"
            html += f"""
            <div style="margin: 10px 0;">
                <div style="cursor: pointer; padding: 6px; background: #e3f2fd; border-radius: 3px; font-size: 0.9em;"
                     onclick="toggleSection('{unique_id}')">
                    <strong>🤖 System Prompt</strong> <span id="{unique_id}-toggle">▼</span>
                </div>
                <div id="{unique_id}" style="display: none; padding: 8px; background: #f8f9fa; border: 1px solid #ddd; border-top: none; border-radius: 0 0 3px 3px;">
                    <pre style="white-space: pre-wrap; font-family: monospace; font-size: 0.8em; margin: 0; line-height: 1.3;">{self._escape_html(system_prompt)}</pre>
                </div>
            </div>
            """

        html += "</div>"
        return html


class SelfRefineCriticFormatter(BaseCriticFormatter):
    """Formatter for Self-Refine Critic"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format Self-Refine Critic details"""
        confidence = critic.get("confidence", 0.0)
        violations = critic.get("violations", [])
        suggestions = critic.get("suggestions", [])
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})

        html = f"<div><strong>Confidence:</strong> {confidence:.1f}</div>"

        # Add violations and suggestions
        html += self._format_violations_and_suggestions(violations, suggestions)

        # Add detailed feedback
        html += self._format_feedback(feedback)

        # Add Self-Refine specific metadata
        html += self._format_self_refine_metadata(metadata)

        return html

    def _format_self_refine_metadata(self, metadata: dict) -> str:
        """Format Self-Refine specific metadata"""
        if not metadata:
            return ""

        html = """
        <div class="metadata">
            <strong>🔄 Self-Refine Configuration:</strong><br>
        """

        # Max iterations
        max_iterations = metadata.get("max_iterations", "N/A")
        html += f"""
        <div style="background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #2196f3;">
            <strong>🔁 Max Iterations: {max_iterations}</strong>
        </div>
        """

        # Improvement criteria
        improvement_criteria = metadata.get("improvement_criteria", [])
        if improvement_criteria:
            html += f"""
            <div style="margin: 10px 0;">
                <strong>🎯 Improvement Criteria ({len(improvement_criteria)}):</strong>
                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                    {"".join(f"<li><strong>{criterion.title()}</strong></li>" for criterion in improvement_criteria)}
                </ul>
            </div>
            """

        # Self-Refine process explanation
        html += f"""
        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; border-left: 3px solid #6c757d;">
            <strong>📋 Self-Refine Process:</strong><br>
            <small style="color: #666;">
                This critic uses iterative self-refinement where the same model critiques its own output
                and then revises it based on that critique. The process continues until either the
                maximum iterations are reached or the critic determines no further improvement is needed.
            </small>
        </div>
        """

        html += "</div>"
        return html


class SelfRAGCriticFormatter(BaseCriticFormatter):
    """Formatter for Self-RAG Critic"""

    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format Self-RAG Critic details"""
        confidence = critic.get("confidence", 0.0)
        violations = critic.get("violations", [])
        suggestions = critic.get("suggestions", [])
        feedback = critic.get("feedback", "")
        metadata = critic.get("metadata", {})

        html = f"<div><strong>Confidence:</strong> {confidence:.1f}</div>"

        # Add violations and suggestions
        html += self._format_violations_and_suggestions(violations, suggestions)

        # Add detailed feedback
        html += self._format_feedback(feedback)

        # Add Self-RAG specific metadata
        html += self._format_self_rag_metadata(metadata)

        return html

    def _format_self_rag_metadata(self, metadata: dict) -> str:
        """Format Self-RAG specific metadata"""
        if not metadata:
            return ""

        html = """
        <div class="metadata">
            <strong>🔍 Self-RAG Assessment:</strong><br>
        """

        # RAG Assessments (reflection tokens)
        rag_assessments = metadata.get("rag_assessments", {})
        if rag_assessments:
            html += f"""
            <div style="margin: 10px 0;">
                <strong>🎯 Reflection Tokens:</strong>
                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                    <li><strong>Retrieval:</strong> {self._escape_html(rag_assessments.get('retrieval', 'N/A'))}</li>
                    <li><strong>Relevance:</strong> {self._escape_html(rag_assessments.get('relevance', 'N/A'))}</li>
                    <li><strong>Support:</strong> {self._escape_html(rag_assessments.get('support', 'N/A'))}</li>
                    <li><strong>Utility:</strong> {self._escape_html(rag_assessments.get('utility', 'N/A'))}</li>
                </ul>
            </div>
            """

        # Utility Score
        utility_score = metadata.get("utility_score")
        if utility_score is not None:
            score_color = self._get_utility_color(utility_score)
            html += f"""
            <div style="margin: 10px 0;">
                <strong>⭐ Utility Score:</strong>
                <span style="color: {score_color}; font-weight: bold;">{utility_score}/5</span>
            </div>
            """

        # Retrieval Information
        retrieval_needed = metadata.get("retrieval_needed")
        retrieved_docs_count = metadata.get("retrieved_docs_count", 0)
        retriever_used = metadata.get("retriever_used")
        context_length = metadata.get("context_length", 0)

        if retrieval_needed is not None:
            html += f"""
            <div style="margin: 10px 0;">
                <strong>📚 Retrieval Information:</strong>
                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                    <li><strong>Retrieval Needed:</strong> {'✅ Yes' if retrieval_needed else '❌ No'}</li>
                    <li><strong>Documents Retrieved:</strong> {retrieved_docs_count}</li>
                    <li><strong>Context Length:</strong> {context_length} characters</li>
                    {f'<li><strong>Retriever Used:</strong> {retriever_used}</li>' if retriever_used else ''}
                </ul>
            </div>
            """

        html += "</div>"
        return html

    def _get_utility_color(self, score: int) -> str:
        """Get color for utility score"""
        if score >= 4:
            return "#28a745"  # Green
        elif score >= 3:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red


class CriticFormatterFactory:
    """Factory for creating appropriate critic formatters"""

    _formatters = {
        "ConstitutionalCritic": ConstitutionalCriticFormatter,
        "SelfConsistencyCritic": SelfConsistencyCriticFormatter,
        "ReflexionCritic": ReflexionCriticFormatter,
        "MetaRewardingCritic": MetaRewardingCriticFormatter,
        "NCriticsCritic": NCriticsCriticFormatter,
        "PromptCritic": PromptCriticFormatter,
        "SelfRefineCritic": SelfRefineCriticFormatter,
        "SelfRAGCritic": SelfRAGCriticFormatter,
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
