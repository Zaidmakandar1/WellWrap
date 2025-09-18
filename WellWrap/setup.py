#!/usr/bin/env python3
"""
Medical Report Simplifier - Clean Architecture Setup
Package setup and installation script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="medical-report-simplifier",
    version="1.0.0",
    description="AI-powered medical report analysis platform with clean architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Medical Report Simplifier Team",
    author_email="team@medicalreportsimplifier.com",
    url="https://github.com/your-username/medical-report-simplifier",
    
    # Package configuration
    packages=find_packages(
        exclude=["tests*", "docs*", "scripts*"]
    ),
    python_requires=">=3.8",
    install_requires=requirements,
    
    # Optional dependencies for different use cases
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "ml": [
            "torch>=1.9.0",
            "transformers>=4.20.0",
            "scikit-learn>=1.0.0",
            "optuna>=3.0.0",
            "shap>=0.41.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "monitoring": [
            "prometheus-client>=0.14.0",
            "grafana-api>=1.0.3",
        ],
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "medical-simplifier=scripts.setup.start_development:main",
            "mrs-migrate=scripts.setup.migrate_to_clean_structure:main",
            "mrs-setup=scripts.setup.setup_database:main",
        ]
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for package discovery
    keywords=[
        "medical",
        "healthcare",
        "ai",
        "machine learning",
        "report analysis",
        "nlp",
        "medical reports",
        "health analytics",
    ],
    
    # Include additional files
    include_package_data=True,
    package_data={
        "data": ["samples/*.txt"],
        "config": ["environments/*.yaml", "deployment/*.yml"],
        "ml": ["models/trained/.gitkeep"],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-username/medical-report-simplifier/issues",
        "Source": "https://github.com/your-username/medical-report-simplifier",
        "Documentation": "https://medical-report-simplifier.readthedocs.io/",
    },
)
