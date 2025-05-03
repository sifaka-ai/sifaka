"""Test different import paths for RegexMatch in guardrails."""

def test_imports():
    """Test different import paths for RegexMatch."""
    print("Testing imports for RegexMatch...")

    # Try importing from guardrails.hub
    try:
        from guardrails.hub import RegexMatch
        print("SUCCESS: Imported RegexMatch from guardrails.hub")
        return "guardrails.hub"
    except ImportError as e:
        print(f"FAILED: Could not import from guardrails.hub - {e}")

    # Try importing validator modules directly
    try:
        import guardrails.validators
        print(f"Available in guardrails.validators: {dir(guardrails.validators)}")
    except ImportError as e:
        print(f"FAILED: Could not import guardrails.validators - {e}")

    # Try directly from regex_match package if installed separately
    try:
        from regex_match import RegexMatch
        print("SUCCESS: Imported RegexMatch from regex_match")
        return "regex_match"
    except ImportError as e:
        print(f"FAILED: Could not import from regex_match - {e}")

    # Check if regex_match validator is installed
    try:
        import subprocess
        result = subprocess.run(['guardrails', 'hub', 'list'], capture_output=True, text=True)
        print(f"Installed Guardrails validators: {result.stdout}")
    except Exception as e:
        print(f"FAILED: Could not check installed validators - {e}")

    return None

if __name__ == "__main__":
    import_path = test_imports()

    if import_path:
        print(f"\nUse the following import in your code:\nfrom {import_path} import RegexMatch")
    else:
        print("\nRegexMatch could not be found. You may need to install it:")
        print("guardrails hub install hub://guardrails/regex_match")