#!/usr/bin/env python3
"""
Sifaka Thought Analysis Tool

A convenient script to analyze and visualize Sifaka thoughts from the new PydanticAI architecture.
Supports both text analysis and HTML visualization.

Usage:
    python scripts/analyze_thoughts.py thoughts/my_thought.json
    python scripts/analyze_thoughts.py thoughts/my_thought.json --html
    python scripts/analyze_thoughts.py thoughts/my_thought.json --both
    python scripts/analyze_thoughts.py thoughts/ --all
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add the scripts directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from analysis.thought_analyzer import ThoughtAnalyzer
from analysis.thought_visualizer import HTMLThoughtVisualizer


def analyze_single_file(file_path: Path, html: bool = False, text: bool = True) -> None:
    """Analyze a single thought file"""
    print(f"🧠 Analyzing: {file_path}")
    print("=" * 80)
    
    if text:
        try:
            analyzer = ThoughtAnalyzer(str(file_path))
            analyzer.print_overview()
            analyzer.print_visual_timeline()
            analyzer.print_iteration_summary()
            analyzer.print_critic_analysis()
        except Exception as e:
            print(f"❌ Text analysis failed: {e}")
    
    if html:
        try:
            visualizer = HTMLThoughtVisualizer(str(file_path))
            visualizer.generate_html()
        except Exception as e:
            print(f"❌ HTML visualization failed: {e}")


def find_thought_files(directory: Path) -> List[Path]:
    """Find all thought JSON files in a directory"""
    thought_files = []
    
    # Look for JSON files that look like thoughts
    for json_file in directory.glob("*.json"):
        # Skip if it's clearly not a thought file
        if json_file.name.startswith('.'):
            continue
        thought_files.append(json_file)
    
    return sorted(thought_files)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize Sifaka thoughts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single thought file (text only)
  python scripts/analyze_thoughts.py thoughts/my_thought.json
  
  # Generate HTML visualization only
  python scripts/analyze_thoughts.py thoughts/my_thought.json --html
  
  # Generate both text analysis and HTML
  python scripts/analyze_thoughts.py thoughts/my_thought.json --both
  
  # Analyze all thought files in a directory
  python scripts/analyze_thoughts.py thoughts/ --all
  
  # Generate HTML for all files in a directory
  python scripts/analyze_thoughts.py thoughts/ --all --html
        """
    )
    
    parser.add_argument(
        "path",
        help="Path to thought file or directory containing thought files"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML visualization only (no text analysis)"
    )
    
    parser.add_argument(
        "--both",
        action="store_true",
        help="Generate both text analysis and HTML visualization"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all JSON files in the directory (if path is a directory)"
    )
    
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help="Show only overview (for text analysis)"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"❌ Error: Path '{path}' does not exist")
        return 1
    
    # Determine what to generate
    generate_html = args.html or args.both
    generate_text = not args.html or args.both
    
    if path.is_file():
        # Single file
        if not path.suffix == '.json':
            print(f"❌ Error: '{path}' is not a JSON file")
            return 1
        
        analyze_single_file(path, html=generate_html, text=generate_text)
        
    elif path.is_dir():
        # Directory
        if not args.all:
            print(f"❌ Error: '{path}' is a directory. Use --all to process all files in it.")
            return 1
        
        thought_files = find_thought_files(path)
        
        if not thought_files:
            print(f"❌ No JSON files found in '{path}'")
            return 1
        
        print(f"🔍 Found {len(thought_files)} JSON files in '{path}'")
        print()
        
        for i, file_path in enumerate(thought_files, 1):
            print(f"\n{'='*80}")
            print(f"📁 File {i}/{len(thought_files)}: {file_path.name}")
            print('='*80)
            
            try:
                analyze_single_file(file_path, html=generate_html, text=generate_text)
            except Exception as e:
                print(f"❌ Failed to analyze {file_path}: {e}")
                continue
    
    else:
        print(f"❌ Error: '{path}' is neither a file nor a directory")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
