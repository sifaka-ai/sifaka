"""Setup module for CI integration tests."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.integration.mock_helpers import setup_ci_environment

# Set up CI environment if running in CI
if __name__ == "__main__":
    setup_ci_environment()
    print("CI environment configured for integration tests")
    print(f"USE_MOCK_LLM: {os.getenv('USE_MOCK_LLM', 'false')}")
    print(f"CI: {os.getenv('CI', 'false')}")
