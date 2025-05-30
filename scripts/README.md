# Sifaka Scripts

This directory contains utility scripts for working with Sifaka data and outputs.

## thought_digest.py

A tool to convert Sifaka thought JSON files into a more readable format. This script presents model prompts followed by text in chronological order, with validation and critic feedback summaries in between.

### Features

- **Chronological presentation**: Shows iterations in order with clear timestamps
- **Feedback summaries**: Brief summaries of validation results and critic feedback
- **Multiple file support**: Process multiple thought files at once
- **Output options**: Print to stdout or save to file
- **Glob pattern support**: Use wildcards to process multiple files

### Usage

```bash
# Process a single file (auto-saves to digests/ folder)
python scripts/thought_digest.py thoughts/example.json

# Process multiple files (auto-saves to digests/ folder)
python scripts/thought_digest.py thoughts/*.json

# Print to console instead of saving
python scripts/thought_digest.py thoughts/example.json --stdout

# Save to custom location
python scripts/thought_digest.py thoughts/example.json -o custom_digest.txt

# Get help
python scripts/thought_digest.py --help
```

### Output Format

The tool generates a structured digest with:

1. **Header**: File name, chain ID, and total iterations
2. **For each iteration**:
   - Iteration number, timestamp, and model name
   - Model prompt (the exact prompt sent to the model)
   - Feedback summary (validation results and critic feedback)
   - Generated text

### Understanding the Flow

**Important**: Sifaka's execution flow can be confusing at first. Here's what actually happens:

- **Iteration 1**: Initial generation only (no feedback yet)
- **Iteration 2+**: Shows feedback from evaluating the *previous* iteration's text, plus new improved text

The feedback summary clarifies this with notes like:
- `(No feedback yet - this is initial generation)` for iteration 1
- `(Feedback from evaluating iteration N)` for subsequent iterations

**Why critics can be "satisfied" while validators fail**: Critics evaluate content quality/style, while validators check hard constraints (like length limits). They use different criteria!

### Example Output

```
================================================================================
🧠 THOUGHT DIGEST: example.json
================================================================================
Chain ID: abc123
Total Iterations: 2

────────────────────────────────────────────────────────────
📝 ITERATION 1 | 2025-05-29 09:37:22
🤖 Model: gpt-4o-mini
────────────────────────────────────────────────────────────

🎯 MODEL PROMPT:
Write a short article about AI.

📊 FEEDBACK SUMMARY:
   ✅ No validation issues
   💭 No critic feedback

📄 GENERATED TEXT:
Artificial Intelligence (AI) represents...
```

### Requirements

- Python 3.7+
- Standard library only (no external dependencies)

### Notes

- The tool automatically detects and skips non-JSON files
- Files without the expected thought structure will show "No thoughts found"
- Timestamps are formatted for readability
- Validation and critic feedback are summarized for quick scanning
