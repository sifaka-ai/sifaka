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
    # Dev tools
    "dev": [
        "pytest>=8.0.0",
        "black>=24.0.0",
        "isort>=5.12.0",
        "mypy>=1.5.0",
        "ruff>=0.1.0",
    ],
}

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
