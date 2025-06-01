# Analysis Directory

This directory contains analysis outputs from Sifaka thought processing tools.

## Directory Structure

```
analysis/
├── html/           # Interactive HTML visualizations
│   └── *_visualization.html
├── reports/        # Text-based analysis summaries  
│   └── *_analysis_summary.txt
└── README.md       # This file
```

## File Types

### HTML Visualizations (`html/`)
Interactive web-based visualizations created by `thought_visualizer.py`:
- Timeline view of thought iterations
- Expandable sections for model prompts, responses, and critic feedback
- Specialized formatting for different critic types
- Color-coded validation results

### Analysis Reports (`reports/`)
Detailed text-based analysis summaries created by `thought_analyzer.py`:
- Overview of thought data and chains
- Visual timeline representation
- Iteration-by-iteration breakdown
- Critic feedback analysis
- Prompt evolution tracking

## Usage

### Generate HTML Visualization
```bash
python scripts/analysis/thought_visualizer.py thoughts/example_thoughts.json
# Output: analysis/html/example_thoughts_visualization.html
```

### Generate Analysis Report
```bash
python scripts/analysis/thought_analyzer.py thoughts/example_thoughts.json --auto-export
# Output: analysis/reports/example_thoughts_analysis_summary.txt
```

## Organization Benefits

- **Separation of Concerns**: HTML and text files are clearly separated
- **Better Navigation**: Easier to find specific file types
- **Cleaner Structure**: Reduces clutter in the main analysis directory
- **Scalability**: Can easily add more subdirectories for different analysis types
