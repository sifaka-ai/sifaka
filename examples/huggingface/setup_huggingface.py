#!/usr/bin/env python3
"""
HuggingFace Setup Helper for Sifaka

This script helps you set up HuggingFace integration by:
1. Checking if dependencies are installed
2. Helping you get and configure your API token
3. Testing the connection
4. Providing next steps

Run this before running the HuggingFace examples.
"""

import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if HuggingFace dependencies are installed."""
    print("🔍 Checking HuggingFace dependencies...")
    
    missing_deps = []
    
    try:
        import transformers
        print(f"  ✅ transformers: {transformers.__version__}")
    except ImportError:
        missing_deps.append("transformers")
        print("  ❌ transformers: Not installed")
    
    try:
        import torch
        print(f"  ✅ torch: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
        print("  ❌ torch: Not installed")
    
    try:
        import huggingface_hub
        print(f"  ✅ huggingface_hub: {huggingface_hub.__version__}")
    except ImportError:
        missing_deps.append("huggingface_hub")
        print("  ❌ huggingface_hub: Not installed")
    
    try:
        import accelerate
        print(f"  ✅ accelerate: {accelerate.__version__}")
    except ImportError:
        missing_deps.append("accelerate")
        print("  ❌ accelerate: Not installed")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install 'sifaka[huggingface]'")
        print("Or manually: pip install transformers torch huggingface-hub accelerate")
        return False
    
    print("✅ All HuggingFace dependencies are installed!")
    return True


def check_api_token():
    """Check if HuggingFace API token is configured."""
    print("\n🔑 Checking HuggingFace API token...")
    
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if token:
        if token.startswith("hf_"):
            print("✅ HuggingFace API token is configured!")
            print(f"   Token: {token[:10]}...{token[-4:]}")
            return True
        else:
            print("❌ Invalid token format (should start with 'hf_')")
    else:
        print("❌ HUGGINGFACE_API_TOKEN environment variable not set")
    
    print("\n📝 To get your HuggingFace API token:")
    print("1. Go to https://huggingface.co/join (sign up if needed)")
    print("2. Go to https://huggingface.co/settings/tokens")
    print("3. Click 'New token'")
    print("4. Give it a name like 'sifaka-api'")
    print("5. Select 'Read' permissions")
    print("6. Copy the token (starts with 'hf_')")
    print("7. Set environment variable:")
    print("   export HUGGINGFACE_API_TOKEN='hf_your_token_here'")
    
    return False


def test_connection():
    """Test HuggingFace connection."""
    print("\n🌐 Testing HuggingFace connection...")
    
    try:
        from huggingface_hub import InferenceClient
        
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            print("❌ No API token available for testing")
            return False
        
        # Test with a simple model
        client = InferenceClient(model="microsoft/DialoGPT-small", token=token)
        
        # Try a simple inference
        response = client.text_generation(
            "Hello, how are you?",
            max_new_tokens=10,
            temperature=0.7
        )
        
        print("✅ HuggingFace Inference API connection successful!")
        print(f"   Test response: '{response[:50]}...'")
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        print("This might be due to:")
        print("  - Invalid API token")
        print("  - Network connectivity issues")
        print("  - Model not available")
        return False


def check_device_capabilities():
    """Check available compute devices."""
    print("\n🖥️  Checking device capabilities...")
    
    try:
        import torch
        
        print(f"  CPU: Available")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"  ✅ CUDA GPU: {gpu_count} device(s) - {gpu_name}")
        else:
            print("  ❌ CUDA GPU: Not available")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✅ Apple MPS: Available")
        else:
            print("  ❌ Apple MPS: Not available")
        
        # Recommend device strategy
        if torch.cuda.is_available():
            print("\n💡 Recommendation: Use CUDA GPU for local models")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("\n💡 Recommendation: Use Apple MPS for local models")
        else:
            print("\n💡 Recommendation: Use CPU (consider smaller models)")
            
    except ImportError:
        print("  ❌ PyTorch not available for device detection")


def show_next_steps():
    """Show next steps for using HuggingFace with Sifaka."""
    print("\n🎯 Next Steps:")
    print("1. Run the HuggingFace example:")
    print("   python examples/huggingface_local_remote_example.py")
    print("\n2. Try different model combinations:")
    print("   - Local: microsoft/DialoGPT-small (fast)")
    print("   - Local: microsoft/DialoGPT-medium (balanced)")
    print("   - Remote: microsoft/DialoGPT-large (high quality)")
    print("   - Remote: facebook/blenderbot-400M-distill (conversational)")
    print("\n3. Experiment with quantization:")
    print("   - quantization='4bit' (most memory efficient)")
    print("   - quantization='8bit' (balanced)")
    print("   - quantization=None (full precision)")
    print("\n4. Check the documentation:")
    print("   - HuggingFace Models: https://huggingface.co/models")
    print("   - Sifaka docs: README.md")


def main():
    """Main setup function."""
    print("🤗 HuggingFace Setup Helper for Sifaka")
    print("=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check API token
    token_ok = check_api_token()
    
    # Test connection if token is available
    if token_ok:
        connection_ok = test_connection()
    else:
        connection_ok = False
    
    # Check device capabilities
    if deps_ok:
        check_device_capabilities()
    
    # Summary
    print("\n📋 Setup Summary:")
    print(f"  Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"  API Token: {'✅' if token_ok else '❌'}")
    print(f"  Connection: {'✅' if connection_ok else '❌'}")
    
    if deps_ok and token_ok and connection_ok:
        print("\n🎉 HuggingFace setup is complete!")
        show_next_steps()
    else:
        print("\n⚠️  Setup incomplete. Please address the issues above.")
        if not deps_ok:
            print("   Install dependencies first: pip install 'sifaka[huggingface]'")
        if not token_ok:
            print("   Get and set your HuggingFace API token")


if __name__ == "__main__":
    main()
