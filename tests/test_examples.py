"""Test suite to verify all examples work correctly."""

import subprocess
import sys
from pathlib import Path
import pytest


class TestExamples:
    """Test that all examples run successfully."""

    def test_basic_chain_example(self):
        """Test that basic_chain.py runs without errors."""
        result = subprocess.run(
            [sys.executable, "examples/mock/basic_chain.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Example failed with error: {result.stderr}"
        assert "Example completed successfully!" in result.stdout

    def test_critics_example(self):
        """Test that critics_example.py runs without errors."""
        result = subprocess.run(
            [sys.executable, "examples/mock/critics_example.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Example failed with error: {result.stderr}"
        assert "Examples completed!" in result.stdout

    def test_validators_example(self):
        """Test that validators_example.py runs without errors."""
        result = subprocess.run(
            [sys.executable, "examples/mock/validators_example.py"],
            capture_output=True,
            text=True,
            timeout=60,  # Longer timeout for GuardrailsAI
        )
        assert result.returncode == 0, f"Example failed with error: {result.stderr}"
        assert "Examples completed!" in result.stdout

    def test_storage_example(self):
        """Test that storage_example.py runs without errors."""
        result = subprocess.run(
            [sys.executable, "examples/mock/storage_example.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Example failed with error: {result.stderr}"
        assert "completed successfully!" in result.stdout

    def test_redis_retriever_example(self):
        """Test that redis_retriever_example.py runs without errors."""
        result = subprocess.run(
            [sys.executable, "examples/mock/redis_retriever_example.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Example failed with error: {result.stderr}"
        assert "Example completed!" in result.stdout or "completed successfully!" in result.stdout

    def test_vector_db_retriever_example(self):
        """Test that vector_db_retriever_example.py runs without errors."""
        result = subprocess.run(
            [sys.executable, "examples/mock/vector_db_retriever_example.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Example failed with error: {result.stderr}"
        assert "Example completed!" in result.stdout or "completed!" in result.stdout

    def test_self_refine_example(self):
        """Test that self_refine_example.py runs without errors."""
        result = subprocess.run(
            [sys.executable, "examples/mock/self_refine_example.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Example failed with error: {result.stderr}"
        assert "Example completed!" in result.stdout or "completed successfully!" in result.stdout

    @pytest.mark.skipif(
        not Path("examples/openai").exists(), reason="OpenAI examples directory not found"
    )
    def test_openai_examples_exist(self):
        """Test that OpenAI examples exist and are properly structured."""
        openai_dir = Path("examples/openai")
        assert openai_dir.exists()

        # Check for at least one OpenAI example
        python_files = list(openai_dir.glob("*.py"))
        assert len(python_files) > 0, "No Python files found in examples/openai"

    @pytest.mark.skipif(
        not Path("examples/mixed").exists(), reason="Mixed examples directory not found"
    )
    def test_mixed_examples_exist(self):
        """Test that mixed examples exist and are properly structured."""
        mixed_dir = Path("examples/mixed")
        assert mixed_dir.exists()

        # Check for at least one mixed example
        python_files = list(mixed_dir.glob("*.py"))
        assert len(python_files) > 0, "No Python files found in examples/mixed"


class TestExampleStructure:
    """Test the structure and content of example files."""

    def test_examples_directory_exists(self):
        """Test that examples directory exists."""
        examples_dir = Path("examples")
        assert examples_dir.exists(), "Examples directory not found"
        assert examples_dir.is_dir(), "Examples path is not a directory"

    def test_mock_examples_exist(self):
        """Test that mock examples directory exists with expected files."""
        mock_dir = Path("examples/mock")
        assert mock_dir.exists(), "Mock examples directory not found"

        expected_files = [
            "basic_chain.py",
            "critics_example.py",
            "validators_example.py",
            "storage_example.py",
            "redis_retriever_example.py",
            "vector_db_retriever_example.py",
            "self_refine_example.py",
        ]

        for filename in expected_files:
            file_path = mock_dir / filename
            assert file_path.exists(), f"Expected example file {filename} not found"
            assert file_path.is_file(), f"{filename} is not a file"

    def test_example_files_have_content(self):
        """Test that example files are not empty."""
        mock_dir = Path("examples/mock")

        for py_file in mock_dir.glob("*.py"):
            content = py_file.read_text()
            assert len(content.strip()) > 0, f"Example file {py_file.name} is empty"
            assert "import" in content, f"Example file {py_file.name} has no imports"

    def test_example_files_have_docstrings(self):
        """Test that example files have proper documentation."""
        mock_dir = Path("examples/mock")

        for py_file in mock_dir.glob("*.py"):
            content = py_file.read_text()
            # Check for either module docstring or significant comments
            has_docstring = '"""' in content or "'''" in content
            has_comments = content.count("#") >= 3  # At least 3 comment lines

            assert has_docstring or has_comments, f"Example file {py_file.name} lacks documentation"


class TestDocumentation:
    """Test that documentation files exist and are properly structured."""

    def test_docs_directory_exists(self):
        """Test that docs directory exists."""
        docs_dir = Path("docs")
        assert docs_dir.exists(), "Documentation directory not found"
        assert docs_dir.is_dir(), "Docs path is not a directory"

    def test_api_reference_exists(self):
        """Test that API reference documentation exists."""
        api_ref = Path("docs/API_REFERENCE.md")
        assert api_ref.exists(), "API_REFERENCE.md not found"

        content = api_ref.read_text()
        assert len(content) > 1000, "API reference seems too short"
        assert "# Sifaka API Reference" in content
        assert "## Core Components" in content

    def test_architecture_docs_exist(self):
        """Test that architecture documentation exists."""
        arch_doc = Path("docs/ARCHITECTURE.md")
        assert arch_doc.exists(), "ARCHITECTURE.md not found"

        content = arch_doc.read_text()
        assert len(content) > 1000, "Architecture doc seems too short"
        assert "# Sifaka Architecture" in content
        assert "## Core Components" in content

    def test_examples_docs_exist(self):
        """Test that examples documentation exists."""
        examples_doc = Path("docs/EXAMPLES.md")
        assert examples_doc.exists(), "EXAMPLES.md not found"

        content = examples_doc.read_text()
        assert len(content) > 1000, "Examples doc seems too short"
        assert "# Sifaka Examples Documentation" in content
        assert "## Quick Start" in content


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
