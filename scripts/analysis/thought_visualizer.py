#!/usr/bin/env python3
"""
HTML Thought Visualizer

Creates an interactive HTML visualization of Sifaka thought data.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    from .critic_formatters import CriticFormatterFactory
except ImportError:
    from critic_formatters import CriticFormatterFactory


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

    def _format_critic_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        """Format critic details based on critic type"""
        critic_name = critic.get("critic_name", "Unknown")
        formatter = CriticFormatterFactory.get_formatter(critic_name)
        return formatter.format_details(critic, iteration, critic_index)

    def generate_html(self, output_file: str = None):
        """Generate interactive HTML visualization"""

        if output_file is None:
            # Create default output file in /analysis/html directory
            analysis_dir = Path("analysis")
            html_dir = analysis_dir / "html"
            html_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename based on input file
            base_name = self.json_file.stem
            output_file = html_dir / f"{base_name}_visualization.html"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort thoughts by iteration and show all critic feedback for each iteration
        thoughts = []
        for thought_id, thought_data in self.data.items():
            # Filter out metadata entries that aren't actual thought iterations
            if thought_id in ["initial_prompt", "final_result"]:
                continue
            # Only include entries that have the required thought fields
            if not isinstance(thought_data, dict) or "iteration" not in thought_data:
                continue

            thought_data["id"] = thought_id
            # Show all critic feedback for this iteration (no filtering needed)
            thoughts.append(thought_data)
        thoughts.sort(key=lambda t: t.get("iteration", 0))

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sifaka Thought Analysis - {self.json_file.name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary-color: #6366f1;
            --primary-dark: #4f46e5;
            --primary-light: #a5b4fc;
            --secondary-color: #8b5cf6;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-400: #9ca3af;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
            --border-radius: 12px;
            --border-radius-sm: 8px;
            --border-radius-lg: 16px;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: var(--gray-800);
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: var(--border-radius-lg);
            box-shadow: var(--shadow-xl);
            overflow: hidden;
            backdrop-filter: blur(10px);
        }}

        .header {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="0.5" fill="white" opacity="0.1"/><circle cx="90" cy="40" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            pointer-events: none;
        }}

        .header h1 {{
            margin: 0;
            font-size: 3em;
            font-weight: 700;
            letter-spacing: -0.02em;
            position: relative;
            z-index: 1;
        }}

        .header p {{
            margin: 15px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
            font-weight: 400;
            position: relative;
            z-index: 1;
        }}

        .timeline {{
            padding: 30px;
            background: var(--gray-50);
        }}
        .thought {{
            border-left: 4px solid var(--primary-color);
            margin: 30px 0;
            padding: 25px;
            background: white;
            border-radius: 0 var(--border-radius) var(--border-radius) 0;
            position: relative;
            box-shadow: var(--shadow-md);
            transition: all 0.3s ease;
        }}

        .thought:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }}

        .thought::before {{
            content: '';
            position: absolute;
            left: -10px;
            top: 25px;
            width: 16px;
            height: 16px;
            background: var(--primary-color);
            border-radius: 50%;
            border: 4px solid white;
            box-shadow: var(--shadow-md);
        }}

        .thought-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--gray-100);
        }}

        .iteration {{
            font-size: 1.4em;
            font-weight: 600;
            color: var(--primary-color);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .iteration::before {{
            content: '🔄';
            font-size: 0.8em;
        }}

        .timestamp {{
            color: var(--gray-500);
            font-size: 0.9em;
            font-weight: 500;
            background: var(--gray-100);
            padding: 4px 12px;
            border-radius: 20px;
        }}

        .model-info {{
            background: linear-gradient(135deg, #e0f2fe 0%, #f3e5f5 100%);
            padding: 15px;
            border-radius: var(--border-radius-sm);
            margin-bottom: 20px;
            border: 1px solid var(--gray-200);
            font-weight: 500;
        }}

        .model-info strong {{
            color: var(--primary-color);
        }}
        .critics {{
            margin-top: 20px;
        }}

        .critics > strong {{
            display: block;
            margin-bottom: 15px;
            font-size: 1.1em;
            color: var(--gray-700);
        }}

        .critic {{
            background: white;
            border: 1px solid var(--gray-200);
            border-radius: var(--border-radius);
            margin-bottom: 15px;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
        }}

        .critic:hover {{
            box-shadow: var(--shadow-md);
            border-color: var(--primary-light);
        }}

        .critic-header {{
            background: linear-gradient(135deg, var(--gray-50) 0%, var(--gray-100) 100%);
            padding: 15px 20px;
            border-bottom: 1px solid var(--gray-200);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s ease;
        }}

        .critic-header:hover {{
            background: linear-gradient(135deg, var(--gray-100) 0%, var(--gray-200) 100%);
        }}

        .critic-name {{
            font-weight: 600;
            color: var(--gray-700);
            font-size: 1.05em;
        }}

        .critic-status {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .status-badge {{
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .needs-improvement {{
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            color: var(--error-color);
            border: 1px solid #fecaca;
        }}

        .approved {{
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            color: var(--success-color);
            border: 1px solid #bbf7d0;
        }}

        .critic-details {{
            padding: 20px;
            display: none;
            background: var(--gray-50);
            animation: slideDown 0.3s ease;
        }}

        .critic-details.expanded {{
            display: block;
        }}

        @keyframes slideDown {{
            from {{
                opacity: 0;
                transform: translateY(-10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        .suggestions, .violations {{
            margin-top: 15px;
        }}

        .suggestion {{
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            padding: 12px 16px;
            margin: 8px 0;
            border-left: 4px solid var(--warning-color);
            border-radius: 0 var(--border-radius-sm) var(--border-radius-sm) 0;
            box-shadow: var(--shadow-sm);
        }}

        .violations .suggestion {{
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border-left-color: var(--error-color);
        }}

        .metadata {{
            background: linear-gradient(135deg, var(--gray-100) 0%, var(--gray-200) 100%);
            padding: 15px;
            border-radius: var(--border-radius-sm);
            margin-top: 15px;
            font-size: 0.9em;
            border: 1px solid var(--gray-300);
        }}

        .score {{
            font-size: 1.1em;
            font-weight: 600;
            color: var(--primary-color);
            background: white;
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid var(--primary-light);
        }}

        .expand-btn {{
            background: none;
            border: none;
            color: var(--primary-color);
            cursor: pointer;
            font-size: 1.2em;
            transition: all 0.2s ease;
            padding: 4px;
            border-radius: 4px;
        }}

        .expand-btn:hover {{
            background: var(--primary-light);
            color: white;
        }}
        .expandable-section {{
            margin: 20px 0;
            border: 1px solid var(--gray-200);
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
        }}

        .expandable-section:hover {{
            box-shadow: var(--shadow-md);
        }}

        .section-header {{
            background: linear-gradient(135deg, var(--gray-50) 0%, var(--gray-100) 100%);
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--gray-200);
            transition: all 0.2s ease;
            font-weight: 500;
        }}

        .section-header:hover {{
            background: linear-gradient(135deg, var(--gray-100) 0%, var(--gray-200) 100%);
        }}

        .section-details {{
            display: none;
            padding: 0;
            animation: slideDown 0.3s ease;
        }}

        .section-details.expanded {{
            display: block;
        }}

        .content-box {{
            background: #fafafa;
            padding: 20px;
            white-space: pre-wrap;
            font-family: 'JetBrains Mono', 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
            line-height: 1.5;
            border-radius: 0 0 var(--border-radius) var(--border-radius);
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid var(--gray-200);
            border-top: none;
        }}

        .content-box::-webkit-scrollbar {{
            width: 8px;
        }}

        .content-box::-webkit-scrollbar-track {{
            background: var(--gray-100);
            border-radius: 4px;
        }}

        .content-box::-webkit-scrollbar-thumb {{
            background: var(--gray-400);
            border-radius: 4px;
        }}

        .content-box::-webkit-scrollbar-thumb:hover {{
            background: var(--gray-500);
        }}
        .validation-result {{
            background: white;
            margin: 15px;
            padding: 16px;
            border-radius: var(--border-radius-sm);
            border-left: 4px solid var(--success-color);
            box-shadow: var(--shadow-sm);
        }}

        .validation-result.failed {{
            border-left-color: var(--error-color);
        }}

        .validator-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}

        .validator-message {{
            margin-bottom: 10px;
            font-style: italic;
            color: var(--gray-600);
        }}

        .validator-issues, .validator-suggestions {{
            margin-top: 10px;
        }}

        .validator-issues ul, .validator-suggestions ul {{
            margin: 8px 0;
            padding-left: 20px;
        }}

        .validator-issues li, .validator-suggestions li {{
            margin: 4px 0;
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}

            .container {{
                border-radius: var(--border-radius);
            }}

            .header {{
                padding: 30px 20px;
            }}

            .header h1 {{
                font-size: 2.2em;
            }}

            .timeline {{
                padding: 20px;
            }}

            .thought {{
                padding: 20px;
                margin: 20px 0;
            }}

            .thought-header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }}

            .critic-header {{
                padding: 12px 15px;
            }}

            .critic-status {{
                flex-wrap: wrap;
                gap: 8px;
            }}
        }}

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {{
            :root {{
                --gray-50: #1f2937;
                --gray-100: #374151;
                --gray-200: #4b5563;
                --gray-300: #6b7280;
                --gray-400: #9ca3af;
                --gray-500: #d1d5db;
                --gray-600: #e5e7eb;
                --gray-700: #f3f4f6;
                --gray-800: #f9fafb;
                --gray-900: #ffffff;
            }}

            body {{
                color: var(--gray-800);
            }}

            .container {{
                background: var(--gray-100);
            }}

            .content-box {{
                background: var(--gray-200);
                color: var(--gray-800);
            }}
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
                        <strong>📝 User Prompt</strong>
                        <span class="expand-btn">▼</span>
                    </div>
                    <div class="section-details" id="prompt-{iteration}">
                        <div class="content-box">{self._escape_html(prompt)}</div>
                    </div>
                </div>

                <div class="expandable-section">
                    <div class="section-header" onclick="toggleSection('model-prompt-{iteration}')">
                        <strong>🤖 Model Prompt</strong>
                        <span class="expand-btn">▼</span>
                    </div>
                    <div class="section-details" id="model-prompt-{iteration}">
                        <div class="content-box">{self._escape_html(thought.get('model_prompt', 'No model prompt available'))}</div>
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
                metadata = critic.get("metadata", {})

                status_class = "needs-improvement" if needs_improvement else "approved"
                status_text = "Needs Improvement" if needs_improvement else "Approved"
                status_icon = "❌" if needs_improvement else "✅"

                # Get score based on critic type
                if critic_name == "NCriticsCritic":
                    score = metadata.get("aggregated_score", "N/A")
                    if score != "N/A":
                        score = f"{score:.1f}/10"
                else:
                    score = metadata.get("initial_score", "N/A")

                html_content += f"""
                    <div class="critic">
                        <div class="critic-header" onclick="toggleCritic('critic-{iteration}-{i}')">
                            <div class="critic-name">{self._escape_html(critic_name)}</div>
                            <div class="critic-status">
                                <span class="score">Score: {score}</span>
                                <span class="status-badge {status_class}">{status_icon} {status_text}</span>
                                <span class="expand-btn">▼</span>
                            </div>
                        </div>
                        <div class="critic-details" id="critic-{iteration}-{i}">
                            {self._format_critic_details(critic, iteration, i)}
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

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"🌐 HTML visualization created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create HTML visualization of Sifaka thought data")
    parser.add_argument("json_file", help="Path to the thought JSON file")
    parser.add_argument(
        "--output",
        "-o",
        help="Output HTML file (default: analysis/html/{filename}_visualization.html)",
        default=None,
    )

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
