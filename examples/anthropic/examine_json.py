#!/usr/bin/env python3
"""
Simple script to examine the JSON content from the bias example
"""

import json
import glob
import tempfile
from pathlib import Path

def find_latest_storage_dir():
    """Find the most recent bias example storage directory."""
    temp_dir = Path(tempfile.gettempdir())
    pattern = str(temp_dir / "sifaka_bias_example_*")
    
    storage_dirs = glob.glob(pattern)
    if not storage_dirs:
        print("❌ No bias example storage directories found.")
        return None
    
    # Get the most recent one
    latest_dir = max(storage_dirs, key=lambda x: Path(x).stat().st_mtime)
    return Path(latest_dir)

def examine_json_files():
    """Examine the JSON files from the bias example."""
    storage_dir = find_latest_storage_dir()
    if not storage_dir:
        return
    
    print(f"📁 Examining storage directory: {storage_dir}")
    
    # Find all JSON files
    json_files = list(storage_dir.glob("thoughts/**/*.json"))
    json_files.sort(key=lambda x: x.stat().st_mtime)
    
    print(f"📄 Found {len(json_files)} JSON files")
    
    for i, json_file in enumerate(json_files):
        print(f"\n{'='*60}")
        print(f"📝 File {i+1}: {json_file.name}")
        print(f"{'='*60}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"Iteration: {data.get('iteration', 'N/A')}")
            print(f"ID: {data.get('id', 'N/A')}")
            print(f"Chain ID: {data.get('chain_id', 'N/A')}")
            print(f"Timestamp: {data.get('timestamp', 'N/A')}")
            
            # Show the original prompt
            print(f"\n📋 Original Prompt:")
            print(f"'{data.get('prompt', 'N/A')}'")
            
            # Show the model prompt (this is what you're asking about)
            print(f"\n🤖 Model Prompt (what was actually sent to Claude):")
            model_prompt = data.get('model_prompt', 'N/A')
            if len(model_prompt) > 500:
                print(f"'{model_prompt[:500]}...'")
                print(f"[Truncated - full length: {len(model_prompt)} chars]")
            else:
                print(f"'{model_prompt}'")
            
            # Show the generated text
            print(f"\n📄 Generated Text:")
            text = data.get('text', 'N/A')
            print(f"'{text}'")
            
            # Show validation results
            print(f"\n✅ Validation Results:")
            validation_results = data.get('validation_results', {})
            for val_name, val_result in validation_results.items():
                print(f"  {val_name}:")
                print(f"    Passed: {val_result.get('passed', 'N/A')}")
                print(f"    Score: {val_result.get('score', 'N/A')}")
                if val_result.get('issues'):
                    print(f"    Issues: {val_result['issues']}")
            
            # Show critic feedback summary
            print(f"\n🎯 Critic Feedback Summary:")
            critic_feedback = data.get('critic_feedback', [])
            print(f"  Number of critics: {len(critic_feedback)}")
            for j, feedback in enumerate(critic_feedback, 1):
                violations = feedback.get('violations', [])
                suggestions = feedback.get('suggestions', [])
                print(f"  Critic {j}: {len(violations)} violations, {len(suggestions)} suggestions")
            
            # Show history
            print(f"\n📚 History:")
            history = data.get('history', [])
            print(f"  Previous iterations: {len(history)}")
            for hist in history:
                print(f"    - Iteration {hist.get('iteration', 'N/A')}: {hist.get('summary', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")

if __name__ == "__main__":
    examine_json_files()
