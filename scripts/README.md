# Sifaka Scripts

This directory contains utility scripts for analyzing and visualizing Sifaka thought data.

## Analysis Tools

### Thought Analyzer (`analysis/thought_analyzer.py`)

Analyzes thought data from JSON files and provides detailed insights into critic feedback, model performance, and iteration patterns.

**Usage:**
```bash
# Basic analysis (prints to console)
python scripts/analysis/thought_analyzer.py thoughts/your_file.json

# Auto-export to /analysis directory
python scripts/analysis/thought_analyzer.py thoughts/your_file.json --auto-export

# Export to specific file
python scripts/analysis/thought_analyzer.py thoughts/your_file.json --export custom_analysis.txt

# Overview only
python scripts/analysis/thought_analyzer.py thoughts/your_file.json --overview-only
```

**Features:**
- Overview of thought data with statistics
- Visual timeline of iterations
- Detailed critic analysis with type-specific formatting
- Prompt evolution tracking
- Support for all critic types including Constitutional, Self-Consistency, Reflexion, and Meta-Rewarding

### HTML Visualizer (`analysis/thought_visualizer.py`)

Creates interactive HTML visualizations of thought data with expandable sections and detailed critic information.

**Usage:**
```bash
# Generate HTML visualization (saves to /analysis directory)
python scripts/analysis/thought_visualizer.py thoughts/your_file.json

# Save to specific file
python scripts/analysis/thought_visualizer.py thoughts/your_file.json --output custom_viz.html
```

**Features:**
- Interactive HTML interface with expandable sections
- Detailed critic formatting based on critic type
- Validation results display
- Model prompts and responses
- Constitutional critic support with principle violations
- Self-consistency consensus analysis
- Reflexion memory sessions
- Meta-rewarding judgment details

## Critic Support

The analysis tools support specialized formatting for different critic types:

- **ConstitutionalCritic**: Shows principle violations, compliance rates, and constitutional analysis
- **SelfConsistencyCritic**: Displays consensus statistics and agreement ratios
- **ReflexionCritic**: Shows memory sessions, trial numbers, and self-reflection content
- **MetaEvaluationCritic**: Displays initial scores, judgment criteria, and meta-evaluation
- **GenericCritic**: Fallback formatting for any other critic types

## Output Directory

By default, both tools save their output to the `/analysis` directory:

- Text analysis: `analysis/{filename}_analysis_summary.txt`
- HTML visualization: `analysis/{filename}_visualization.html`

## Modular Architecture

The analysis tools use a modular architecture with:

- `critic_formatters.py`: Specialized formatters for different critic types
- `CriticFormatterFactory`: Factory pattern for creating appropriate formatters
- Extensible design for adding new critic types

## Adding New Critic Types

To add support for a new critic type:

1. Create a new formatter class inheriting from `BaseCriticFormatter`
2. Implement the `format_details()` method
3. Register the formatter with `CriticFormatterFactory.register_formatter()`

Example:
```python
class MyCriticFormatter(BaseCriticFormatter):
    def format_details(self, critic: dict, iteration: int, critic_index: int) -> str:
        # Your custom formatting logic here
        return formatted_html

# Register the formatter
CriticFormatterFactory.register_formatter("MyCritic", MyCriticFormatter)
```
