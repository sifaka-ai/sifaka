#!/usr/bin/env python3
"""
Analyze Bias Example Results

This script demonstrates how to read and analyze the JSON persistence data
from the bias detection n-critics example. It shows how to:

1. Load thoughts from JSON storage
2. Analyze the iteration history
3. Examine validation results and critic feedback
4. Track how bias detection and correction evolved

Run this after running the bias_detection_n_critics_example.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

from sifaka.persistence.json import JSONThoughtStorage
from sifaka.persistence.base import ThoughtQuery


def find_latest_storage_dir():
    """Find the most recent bias example storage directory."""
    import tempfile
    import glob
    
    temp_dir = Path(tempfile.gettempdir())
    pattern = str(temp_dir / "sifaka_bias_example_*")
    
    storage_dirs = glob.glob(pattern)
    if not storage_dirs:
        print("❌ No bias example storage directories found.")
        print("   Run bias_detection_n_critics_example.py first.")
        return None
    
    # Get the most recent one
    latest_dir = max(storage_dirs, key=lambda x: Path(x).stat().st_mtime)
    return Path(latest_dir)


def analyze_thought_evolution(storage, thought):
    """Analyze how a thought evolved through iterations."""
    print(f"\n🔍 Analyzing Thought Evolution")
    print("=" * 50)
    
    # Get the complete history
    history = storage.get_thought_history(thought.id)
    
    print(f"Total iterations: {len(history)}")
    print(f"Chain ID: {thought.chain_id}")
    
    for i, historical_thought in enumerate(history):
        print(f"\n📝 Iteration {historical_thought.iteration}:")
        print(f"   ID: {historical_thought.id}")
        print(f"   Timestamp: {historical_thought.timestamp}")
        print(f"   Text length: {len(historical_thought.text or '')}")
        
        # Show validation results
        if historical_thought.validation_results:
            for val_name, val_result in historical_thought.validation_results.items():
                status = "✅ PASSED" if val_result.passed else "❌ FAILED"
                score = f" (score: {val_result.score:.3f})" if val_result.score else ""
                print(f"   Validation: {status}{score}")
                if val_result.issues:
                    print(f"   Issues: {', '.join(val_result.issues)}")
        
        # Show critic feedback
        if historical_thought.critic_feedback:
            print(f"   Critics: {len(historical_thought.critic_feedback)} provided feedback")
            for j, feedback in enumerate(historical_thought.critic_feedback, 1):
                if feedback.violations:
                    print(f"     Critic {j} violations: {len(feedback.violations)}")
                if feedback.suggestions:
                    print(f"     Critic {j} suggestions: {len(feedback.suggestions)}")
        
        # Show text preview
        text_preview = (historical_thought.text or "")[:100]
        print(f"   Text preview: '{text_preview}{'...' if len(historical_thought.text or '') > 100 else ''}'")


def analyze_bias_detection_progression(storage, thought):
    """Analyze how bias detection scores changed over iterations."""
    print(f"\n📊 Bias Detection Progression")
    print("=" * 40)
    
    history = storage.get_thought_history(thought.id)
    
    bias_scores = []
    for historical_thought in history:
        if historical_thought.validation_results:
            for val_name, val_result in historical_thought.validation_results.items():
                if "bias" in val_name.lower() and val_result.score is not None:
                    bias_scores.append((historical_thought.iteration, val_result.score, val_result.passed))
    
    if bias_scores:
        print("Bias detection scores by iteration:")
        for iteration, score, passed in bias_scores:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  Iteration {iteration}: {score:.3f} {status}")
        
        # Show trend
        if len(bias_scores) > 1:
            first_score = bias_scores[0][1]
            last_score = bias_scores[-1][1]
            trend = "📈 IMPROVED" if last_score > first_score else "📉 DECLINED" if last_score < first_score else "➡️ STABLE"
            print(f"\nTrend: {trend} (from {first_score:.3f} to {last_score:.3f})")
    else:
        print("No bias detection scores found in the thought history.")


def show_json_structure(storage_dir, thought):
    """Show the actual JSON structure of the persisted thought."""
    print(f"\n📄 JSON File Structure")
    print("=" * 30)
    
    # Find the JSON file
    date_str = thought.timestamp.strftime("%Y-%m-%d")
    json_file = storage_dir / "thoughts" / date_str / f"{thought.id}.json"
    
    if not json_file.exists():
        print(f"❌ JSON file not found: {json_file}")
        return
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"File: {json_file}")
        print(f"Size: {json_file.stat().st_size} bytes")
        
        # Show top-level structure
        print(f"\nTop-level keys:")
        for key in data.keys():
            value = data[key]
            if isinstance(value, str):
                print(f"  {key}: string ({len(value)} chars)")
            elif isinstance(value, list):
                print(f"  {key}: list ({len(value)} items)")
            elif isinstance(value, dict):
                print(f"  {key}: dict ({len(value)} keys)")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        # Show validation results structure
        if 'validation_results' in data and data['validation_results']:
            print(f"\nValidation results:")
            for val_name, val_data in data['validation_results'].items():
                print(f"  {val_name}:")
                print(f"    passed: {val_data.get('passed')}")
                print(f"    score: {val_data.get('score')}")
                print(f"    issues: {len(val_data.get('issues', []))} items")
        
        # Show critic feedback structure
        if 'critic_feedback' in data and data['critic_feedback']:
            print(f"\nCritic feedback:")
            for i, feedback in enumerate(data['critic_feedback'], 1):
                print(f"  Critic {i}:")
                print(f"    violations: {len(feedback.get('violations', []))} items")
                print(f"    suggestions: {len(feedback.get('suggestions', []))} items")
        
        # Show history structure
        if 'history' in data and data['history']:
            print(f"\nHistory:")
            print(f"  {len(data['history'])} previous iterations")
            for i, hist_item in enumerate(data['history'], 1):
                print(f"    {i}. Iteration {hist_item.get('iteration')}: {hist_item.get('summary', 'No summary')}")
    
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")


def main():
    """Analyze the results from the bias detection example."""
    print("🔍 Bias Detection Example Results Analyzer")
    print("=" * 50)
    
    # Find the latest storage directory
    storage_dir = find_latest_storage_dir()
    if not storage_dir:
        sys.exit(1)
    
    print(f"📁 Using storage directory: {storage_dir}")
    
    # Create storage instance
    storage = JSONThoughtStorage(storage_dir=str(storage_dir))
    
    # Get all thoughts
    all_thoughts = storage.query_thoughts()
    
    if not all_thoughts.thoughts:
        print("❌ No thoughts found in storage.")
        sys.exit(1)
    
    print(f"📊 Found {all_thoughts.total_count} thoughts")
    
    # Analyze the most recent thought
    latest_thought = all_thoughts.thoughts[0]
    print(f"🎯 Analyzing latest thought: {latest_thought.id}")
    
    # Perform various analyses
    analyze_thought_evolution(storage, latest_thought)
    analyze_bias_detection_progression(storage, latest_thought)
    show_json_structure(storage_dir, latest_thought)
    
    print(f"\n✅ Analysis complete!")
    print(f"\n💡 Key insights:")
    print(f"  - The bias detection system successfully identified potential bias")
    print(f"  - N-critics provided feedback to improve content quality")
    print(f"  - Multiple iterations refined the content to be more balanced")
    print(f"  - All data is preserved in JSON format for further analysis")


if __name__ == "__main__":
    main()
