"""
Setup script for the new Sifaka.
"""

from setuptools import find_packages, setup

# Core dependencies (required)
requirements = [
    "pydantic>=2.0.0",
    "typing-extensions>=4.0.0",
    "openai>=1.0.0",
    "anthropic>=0.5.0",
]

# Optional dependencies
extras_require = {
    # Model providers
    "openai": ["tiktoken>=0.9.0"],
    # Retrievers
    "elasticsearch": ["elasticsearch>=8.0.0"],
    "milvus": ["pymilvus>=2.0.0"],
    # Dev tools
    "dev": [
        "pytest>=8.0.2",
        "black>=24.2.0",
        "isort>=5.13.2",
        "mypy>=1.9.0",
        "ruff>=0.3.0",
        "pytest-cov>=4.1.0",
        "flake8>=5.0.0",
        "types-jsonschema>=4.21.0",
        "types-PyYAML>=6.0.0",
    ],
}

# Add a retrievers group that includes all retriever dependencies
extras_require["retrievers"] = [
    dep
    for name, deps in extras_require.items()
    if name in ["elasticsearch", "milvus"]
    for dep in deps
]

# Add an 'all' extra that includes everything
extras_require["all"] = [dep for name, deps in extras_require.items() for dep in deps]

setup(
    name="sifaka",
    version="0.1.0",
    description="Sifaka - Simplified AI Text Processing Framework",
    author="Evan Volgas",
    author_email="evan.volgas@gmail.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)
