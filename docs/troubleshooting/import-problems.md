# Import Problems and Solutions

This guide helps you resolve import-related issues when using Sifaka.

## Quick Import Test

Run this script to test your Sifaka imports:

```python
#!/usr/bin/env python3
"""Test Sifaka imports."""

def test_imports():
    """Test all major Sifaka imports."""
    print("🔍 Testing Sifaka Imports")
    print("=" * 40)

    # Core imports
    tests = [
        ("Core", "from sifaka import Thought"),
        ("PydanticAI Chain", "from sifaka.agents import create_pydantic_chain"),
        ("Models", "from sifaka.models import create_model"),
        ("Validators", "from sifaka.validators import LengthValidator"),
        ("Critics", "from sifaka.critics import ReflexionCritic"),
        ("Storage", "from sifaka.storage import MemoryStorage"),
        ("Utils", "from sifaka.utils.logging import get_logger"),
    ]

    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name}: {import_stmt}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            print(f"   Statement: {import_stmt}")

    # Optional imports
    optional_tests = [
        ("OpenAI", "import openai"),
        ("Anthropic", "import anthropic"),
        ("Redis", "import redis"),
        ("Milvus", "import pymilvus"),
        ("TextBlob", "import textblob"),
        ("Scikit-learn", "import sklearn"),
    ]

    print("\n🔧 Optional Dependencies:")
    for name, import_stmt in optional_tests:
        try:
            exec(import_stmt)
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"⚠️  {name}: Not installed (optional)")

if __name__ == "__main__":
    test_imports()
```

## Common Import Errors

### 1. `ModuleNotFoundError: No module named 'sifaka'`

**Cause:** Sifaka is not installed or not in Python path.

**Solutions:**

```bash
# Using uv (recommended)
uv add sifaka

# Or using pip
# 1. Install Sifaka
pip install sifaka

# 2. Check if installed
pip list | grep sifaka

# 3. Install in current environment
python -m pip install sifaka

# 4. Install with Python 3.11 (required version)
python3.11 -m pip install sifaka

# 5. Install in virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
pip install sifaka
```

### 2. `ImportError: cannot import name 'Chain' from 'sifaka'`

**Cause:** Outdated Sifaka version or corrupted installation.

**Solutions:**

```bash
# Using uv (recommended)
uv sync --upgrade

# Or using pip
# 1. Upgrade Sifaka
pip install --upgrade sifaka

# 2. Reinstall Sifaka
pip uninstall sifaka
pip install sifaka

# 3. Clear pip cache
pip cache purge
pip install sifaka

# 4. Install from source
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka
pip install -e .
```

### 3. `ModuleNotFoundError: No module named 'openai'`

**Cause:** OpenAI package not installed.

**Solutions:**

```bash
# Using uv (recommended)
uv add sifaka[openai]

# Or using pip
# 1. Install OpenAI support
pip install sifaka[openai]

# 2. Or install OpenAI directly
pip install openai

# 3. Install all optional dependencies
pip install sifaka[all]
```

### 4. `ModuleNotFoundError: No module named 'anthropic'`

**Cause:** Anthropic package not installed.

**Solutions:**

```bash
# Using uv (recommended)
uv add sifaka[anthropic]

# Or using pip
# 1. Install Anthropic support
pip install sifaka[anthropic]

# 2. Or install Anthropic directly
pip install anthropic

# 3. Install all model providers
pip install sifaka[models]
```

## Environment-Specific Issues

### Virtual Environment Problems

**Problem:** Imports work in one environment but not another.

**Solutions:**

```bash
# 1. Check which Python you're using
which python
which pip

# 2. Check virtual environment
echo $VIRTUAL_ENV

# 3. Activate correct environment
source /path/to/your/venv/bin/activate

# 4. Install in correct environment
python -m pip install sifaka

# 5. Verify installation location
python -c "import sifaka; print(sifaka.__file__)"
```

### Conda Environment Issues

**Problem:** Imports fail in Conda environments.

**Solutions:**

```bash
# 1. Use conda-forge if available
conda install -c conda-forge sifaka

# 2. Use pip within conda
conda activate your-env
pip install sifaka

# 3. Create new conda environment (Python 3.11 required)
conda create -n sifaka-env python=3.11
conda activate sifaka-env
pip install sifaka
```

### Docker Container Issues

**Problem:** Imports fail in Docker containers.

**Solutions:**

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install Sifaka
RUN pip install sifaka[all]

# Or install specific dependencies
RUN pip install sifaka[openai,anthropic]

# Copy your code
COPY . /app
WORKDIR /app

# Set Python path if needed
ENV PYTHONPATH=/app
```

## Path and PYTHONPATH Issues

### Problem: Module found but imports fail

**Cause:** Python path configuration issues.

**Solutions:**

```python
# 1. Check Python path
import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")

# 2. Add current directory to path
import sys
import os
sys.path.insert(0, os.getcwd())

# 3. Set PYTHONPATH environment variable
# Linux/Mac:
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project"

# Windows:
set PYTHONPATH=%PYTHONPATH%;C:\path\to\your\project
```

### Problem: Relative imports fail

**Cause:** Incorrect relative import usage.

**Solutions:**

```python
# ❌ Don't use relative imports with Sifaka
from .models import create_model

# ✅ Use absolute imports
from sifaka.models import create_model

# ✅ Or import from main package
from sifaka import Chain, Thought
```

## Version Compatibility Issues

### Problem: Import works but functionality fails

**Cause:** Version incompatibilities.

**Solutions:**

```python
# 1. Check versions
import sifaka
print(f"Sifaka version: {sifaka.__version__}")

try:
    import openai
    print(f"OpenAI version: {openai.__version__}")
except ImportError:
    print("OpenAI not installed")

try:
    import anthropic
    print(f"Anthropic version: {anthropic.__version__}")
except ImportError:
    print("Anthropic not installed")

# 2. Check compatibility
def check_compatibility():
    """Check if versions are compatible."""
    import pkg_resources

    requirements = {
        "openai": ">=1.76.0",
        "anthropic": ">=0.50.0",
        "pydantic": ">=2.11.3"
    }

    for package, version_req in requirements.items():
        try:
            pkg_resources.require(f"{package}{version_req}")
            print(f"✅ {package}: Compatible")
        except pkg_resources.DistributionNotFound:
            print(f"⚠️  {package}: Not installed")
        except pkg_resources.VersionConflict as e:
            print(f"❌ {package}: Version conflict - {e}")

check_compatibility()
```

```bash
# 3. Upgrade to compatible versions
pip install --upgrade sifaka openai anthropic

# 4. Install specific versions
pip install "openai>=1.76.0" "anthropic>=0.50.0"
```

## IDE and Editor Issues

### VS Code Issues

**Problem:** Imports fail in VS Code but work in terminal.

**Solutions:**

1. **Select correct Python interpreter:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Python: Select Interpreter"
   - Choose the interpreter where Sifaka is installed

2. **Reload VS Code window:**
   - Press `Ctrl+Shift+P`
   - Type "Developer: Reload Window"

3. **Check VS Code settings:**
   ```json
   // settings.json
   {
       "python.defaultInterpreterPath": "/path/to/your/python",
       "python.terminal.activateEnvironment": true
   }
   ```

### PyCharm Issues

**Problem:** Imports fail in PyCharm.

**Solutions:**

1. **Configure Project Interpreter:**
   - Go to File → Settings → Project → Python Interpreter
   - Select the interpreter where Sifaka is installed

2. **Mark directories as source roots:**
   - Right-click project directory
   - Mark Directory as → Sources Root

3. **Invalidate caches:**
   - File → Invalidate Caches and Restart

### Jupyter Notebook Issues

**Problem:** Imports fail in Jupyter notebooks.

**Solutions:**

```python
# 1. Check kernel
import sys
print(sys.executable)

# 2. Install in notebook environment
!pip install sifaka

# 3. Restart kernel after installation
# Kernel → Restart

# 4. Install ipykernel if needed
!pip install ipykernel
!python -m ipykernel install --user --name=sifaka-env
```

## Advanced Debugging

### Import Hook Debugging

```python
# Debug import process
import sys
import importlib.util

def debug_import(module_name):
    """Debug module import process."""
    print(f"Debugging import of '{module_name}'")

    # Check if module is already imported
    if module_name in sys.modules:
        print(f"✅ {module_name} already in sys.modules")
        return sys.modules[module_name]

    # Find module spec
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ Module spec not found for {module_name}")
        return None

    print(f"✅ Module spec found: {spec.origin}")

    # Try to import
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        print(f"✅ Successfully imported {module_name}")
        return module
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return None

# Usage
debug_import("sifaka")
debug_import("sifaka.models")
```

### Dependency Tree Analysis

```bash
# Check dependency tree
pip install pipdeptree
pipdeptree -p sifaka

# Check for conflicts
pip check

# Show outdated packages
pip list --outdated
```

## Prevention Strategies

### 1. Use Dependency Management

```bash
# Install with specific optional dependencies
pip install sifaka[openai,anthropic]

# Install all features
pip install sifaka[all]

# For development
pip install -e .[full]

# Create environment file for reproducibility
pip freeze > environment.txt
pip install -r environment.txt
```

### 2. Use Virtual Environments

```bash
# Using uv (recommended - handles virtual environments automatically)
uv init sifaka-project
cd sifaka-project
uv add sifaka[all]

# Or using traditional venv
python -m venv sifaka-project
source sifaka-project/bin/activate
pip install sifaka[all]

# Or use conda
conda create -n sifaka-project python=3.11
conda activate sifaka-project
pip install sifaka[all]
```

### 3. Pin Dependencies

```bash
# Pin exact versions for reproducibility
pip install sifaka==0.1.0 openai==1.76.0 anthropic==0.50.0

# Or use pip-tools
pip install pip-tools
echo "sifaka[all]" > requirements.in
pip-compile requirements.in
pip install -r requirements.txt
```

### 4. Test Imports in CI/CD

```yaml
# .github/workflows/test.yml
name: Test Imports
on: [push, pull_request]

jobs:
  test-imports:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11"]  # Sifaka supports Python 3.11 only

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install sifaka[all]

    - name: Test imports
      run: |
        python -c "import sifaka; print('✅ Sifaka imported')"
        python -c "from sifaka import Thought; print('✅ Core imports work')"
        python -c "from sifaka.agents import create_pydantic_chain; print('✅ PydanticAI Chain imports work')"
```

## Getting Help

If you continue to have import issues:

1. **Run the diagnostic script** at the top of this page
2. **Check the [common issues guide](common-issues.md)** for related problems
3. **Report the issue** with:
   - Full error traceback
   - Output of `pip list`
   - Python version and OS
   - Virtual environment details

Most import issues can be resolved by ensuring you have the correct Python environment activated and the right packages installed.
