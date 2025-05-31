#!/usr/bin/env python3
"""
HTML Thought Visualizer

Creates an interactive HTML visualization of Sifaka thought data.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class HTMLThoughtVisualizer:
    """Creates HTML visualizations of thought data"""

    def __init__(self, json_file: str):
        self.json_file = Path(json_file)
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """Load JSON data from file"""
        with open(self.json_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    def _format_text_preview(self, text: str, max_length: int = 200) -> str:
        """Format text for preview with truncation"""
        if len(text) <= max_length:
            return self._escape_html(text)
        return self._escape_html(text[:max_length]) + "..."

    def _format_validation_results(self, validation_results: dict) -> str:
        """Format validation results for display"""
        if not validation_results:
            return '<div class="content-box">No validation results</div>'

        html = ""
        for validator_name, result in validation_results.items():
            passed = result.get("passed", False)
            message = result.get("message", "")
            score = result.get("score", 0.0)
            issues = result.get("issues", [])
            suggestions = result.get("suggestions", [])

            status_icon = "✅" if passed else "❌"
            status_class = "passed" if passed else "failed"

            html += f"""
            <div class="validation-result {status_class}">
                <div class="validator-header">
                    <strong>{status_icon} {self._escape_html(validator_name)}</strong>
                    <span class="score">Score: {score:.1f}</span>
                </div>
                <div class="validator-message">{self._escape_html(message)}</div>
                {f'<div class="validator-issues"><strong>Issues:</strong><ul>{"".join(f"<li>{self._escape_html(issue)}</li>" for issue in issues)}</ul></div>' if issues else ''}
                {f'<div class="validator-suggestions"><strong>Suggestions:</strong><ul>{"".join(f"<li>{self._escape_html(suggestion)}</li>" for suggestion in suggestions)}</ul></div>' if suggestions else ''}
            </div>
            """

        return html

    def generate_html(self, output_file: str):
        """Generate interactive HTML visualization"""

        # Sort thoughts by iteration
        thoughts = []
        for thought_id, thought_data in self.data.items():
            thought_data["id"] = thought_id
            thoughts.append(thought_data)
        thoughts.sort(key=lambda t: t.get("iteration", 0))

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sifaka Thought Analysis - {self.json_file.name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .timeline {{
            padding: 20px;
        }}
        .thought {{
            border-left: 4px solid #667eea;
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 0 8px 8px 0;
            position: relative;
        }}
        .thought::before {{
            content: '';
            position: absolute;
            left: -8px;
            top: 20px;
            width: 12px;
            height: 12px;
            background: #667eea;
            border-radius: 50%;
            border: 3px solid white;
        }}
        .thought-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .iteration {{
            font-size: 1.2em;
            font-weight: bold;
            color: #667eea;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        .model-info {{
            background: #e3f2fd;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
        }}
        .response-preview {{
            background: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #ddd;
            margin-bottom: 15px;
        }}
        .critics {{
            margin-top: 15px;
        }}
        .critic {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 10px;
            overflow: hidden;
        }}
        .critic-header {{
            background: #f8f9fa;
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .critic-header:hover {{
            background: #e9ecef;
        }}
        .critic-name {{
            font-weight: bold;
        }}
        .critic-status {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .status-badge {{
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .needs-improvement {{
            background: #ffebee;
            color: #c62828;
        }}
        .approved {{
            background: #e8f5e8;
            color: #2e7d32;
        }}
        .critic-details {{
            padding: 15px;
            display: none;
        }}
        .critic-details.expanded {{
            display: block;
        }}
        .suggestions, .violations {{
            margin-top: 10px;
        }}
        .suggestion {{
            background: #fff3e0;
            padding: 8px 12px;
            margin: 5px 0;
            border-left: 3px solid #ff9800;
            border-radius: 0 4px 4px 0;
        }}
        .violations .suggestion {{
            background: #ffebee;
            border-left: 3px solid #f44336;
        }}
        .metadata {{
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 0.9em;
        }}
        .score {{
            font-size: 1.1em;
            font-weight: bold;
            color: #667eea;
        }}
        .expand-btn {{
            background: none;
            border: none;
            color: #667eea;
            cursor: pointer;
            font-size: 0.9em;
        }}
        .expandable-section {{
            margin: 15px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .section-header {{
            background: #f8f9fa;
            padding: 12px 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #e0e0e0;
            transition: background-color 0.2s;
        }}
        .section-header:hover {{
            background: #e9ecef;
        }}
        .section-details {{
            display: none;
            padding: 0;
        }}
        .section-details.expanded {{
            display: block;
        }}
        .content-box {{
            background: #ffffff;
            padding: 15px;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
            line-height: 1.4;
            border-radius: 0 0 8px 8px;
            max-height: 400px;
            overflow-y: auto;
        }}
        .validation-result {{
            background: #f8f9fa;
            margin: 10px;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #28a745;
        }}
        .validation-result.failed {{
            border-left-color: #dc3545;
        }}
        .validator-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .validator-message {{
            margin-bottom: 8px;
            font-style: italic;
        }}
        .validator-issues, .validator-suggestions {{
            margin-top: 8px;
        }}
        .validator-issues ul, .validator-suggestions ul {{
            margin: 5px 0;
            padding-left: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Sifaka Thought Analysis</h1>
            <p>File: {self.json_file.name} | {len(thoughts)} iterations</p>
        </div>

        <div class="timeline">
"""

        for thought in thoughts:
            iteration = thought.get("iteration", 0)
            timestamp = thought.get("timestamp", "")
            model_name = thought.get("model_name", "Unknown")
            text = thought.get("text", "")
            prompt = thought.get("prompt", "")
            critics = thought.get("critic_feedback", [])

            html_content += f"""
            <div class="thought">
                <div class="thought-header">
                    <div class="iteration">Iteration {iteration}</div>
                    <div class="timestamp">{timestamp}</div>
                </div>

                <div class="model-info">
                    <strong>🤖 Model:</strong> {self._escape_html(model_name)}
                </div>

                <div class="expandable-section">
                    <div class="section-header" onclick="toggleSection('prompt-{iteration}')">
                        <strong>📝 Prompt</strong>
                        <span class="expand-btn">▼</span>
                    </div>
                    <div class="section-details" id="prompt-{iteration}">
                        <div class="content-box">{self._escape_html(prompt)}</div>
                    </div>
                </div>

                <div class="expandable-section">
                    <div class="section-header" onclick="toggleSection('response-{iteration}')">
                        <strong>💬 Response ({len(text):,} characters)</strong>
                        <span class="expand-btn">▼</span>
                    </div>
                    <div class="section-details" id="response-{iteration}">
                        <div class="content-box">{self._escape_html(text)}</div>
                    </div>
                </div>

                <div class="expandable-section">
                    <div class="section-header" onclick="toggleSection('validation-{iteration}')">
                        <strong>✅ Validation Results</strong>
                        <span class="expand-btn">▼</span>
                    </div>
                    <div class="section-details" id="validation-{iteration}">
                        {self._format_validation_results(thought.get('validation_results', {}))}
                    </div>
                </div>

                <div class="critics">
                    <strong>🔍 Critics ({len(critics)}):</strong>
"""

            for i, critic in enumerate(critics):
                critic_name = critic.get("critic_name", "Unknown")
                needs_improvement = critic.get("needs_improvement", False)
                confidence = critic.get("confidence", 0.0)
                suggestions = critic.get("suggestions", [])
                violations = critic.get("violations", [])
                feedback = critic.get("feedback", "")
                metadata = critic.get("metadata", {})

                status_class = "needs-improvement" if needs_improvement else "approved"
                status_text = "Needs Improvement" if needs_improvement else "Approved"
                status_icon = "❌" if needs_improvement else "✅"

                initial_score = metadata.get("initial_score", "N/A")

                html_content += f"""
                    <div class="critic">
                        <div class="critic-header" onclick="toggleCritic('critic-{iteration}-{i}')">
                            <div class="critic-name">{self._escape_html(critic_name)}</div>
                            <div class="critic-status">
                                <span class="score">Score: {initial_score}</span>
                                <span class="status-badge {status_class}">{status_icon} {status_text}</span>
                                <span class="expand-btn">▼</span>
                            </div>
                        </div>
                        <div class="critic-details" id="critic-{iteration}-{i}">
                            <div><strong>Confidence:</strong> {confidence:.1f}</div>

                            {f'''
                            <div class="violations">
                                <strong>⚠️ Issues Found:</strong>
                                {"".join(f'<div class="suggestion">{self._escape_html(violation)}</div>' for violation in violations[:10])}
                            </div>
                            ''' if violations else ''}

                            {f'''
                            <div class="suggestions">
                                <strong>💡 Suggestions:</strong>
                                {"".join(f'<div class="suggestion">{self._escape_html(suggestion)}</div>' for suggestion in suggestions[:5])}
                            </div>
                            ''' if suggestions else ''}

                            {f'''
                            <div class="feedback">
                                <strong>📝 Detailed Feedback:</strong>
                                <div class="feedback-content" style="white-space: pre-wrap; background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 5px; font-family: monospace; font-size: 0.9em;">{self._escape_html(feedback[:1500])}{"..." if len(feedback) > 1500 else ""}</div>
                            </div>
                            ''' if feedback else ''}

                            {f'''
                            <div class="metadata">
                                <strong>📋 Metadata:</strong><br>
                                {f"Iterations: {metadata.get('num_iterations', 'N/A')}<br>" if 'num_iterations' in metadata else ''}
                                {f"Consensus Items: {metadata.get('consensus_stats', {}).get('stats', {}).get('consensus_items', 'N/A')}<br>" if metadata.get('consensus_stats') else ''}
                                {f"Agreement Ratio: {metadata.get('consensus_stats', {}).get('stats', {}).get('agreement_ratio', 'N/A'):.1%}<br>" if metadata.get('consensus_stats') else ''}
                                {f"Judgment Criteria: {', '.join(metadata.get('judgment_criteria', []))}<br>" if metadata.get('judgment_criteria') else ''}
                                {f"Base Critic Used: {metadata.get('base_critic_used', 'N/A')}<br>" if metadata.get('base_critic_used') else ''}
                                {f"Meta Judge Model: {metadata.get('meta_judge_model', 'N/A')}" if metadata.get('meta_judge_model') else ''}
                            </div>
                            ''' if metadata else ''}
                        </div>
                    </div>
"""

            html_content += """
                </div>
            </div>
"""

        html_content += """
        </div>
    </div>

    <script>
        function toggleCritic(criticId) {
            const details = document.getElementById(criticId);
            const button = details.previousElementSibling.querySelector('.expand-btn');

            if (details.classList.contains('expanded')) {
                details.classList.remove('expanded');
                button.textContent = '▼';
            } else {
                details.classList.add('expanded');
                button.textContent = '▲';
            }
        }

        function toggleSection(sectionId) {
            const details = document.getElementById(sectionId);
            const button = details.previousElementSibling.querySelector('.expand-btn');

            if (details.classList.contains('expanded')) {
                details.classList.remove('expanded');
                button.textContent = '▼';
            } else {
                details.classList.add('expanded');
                button.textContent = '▲';
            }
        }

        // Auto-expand first critic for demo
        document.addEventListener('DOMContentLoaded', function() {
            const firstCritic = document.querySelector('.critic-details');
            if (firstCritic) {
                firstCritic.classList.add('expanded');
                const button = firstCritic.previousElementSibling.querySelector('.expand-btn');
                if (button) button.textContent = '▲';
            }
        });
    </script>
</body>
</html>
"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"🌐 HTML visualization created: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create HTML visualization of Sifaka thought data")
    parser.add_argument("json_file", help="Path to the thought JSON file")
    parser.add_argument("--output", "-o", help="Output HTML file", default="thought_analysis.html")

    args = parser.parse_args()

    try:
        visualizer = HTMLThoughtVisualizer(args.json_file)
        visualizer.generate_html(args.output)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
