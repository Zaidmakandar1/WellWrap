#!/usr/bin/env python3
"""
WellWrap Setup Script
Automated setup for the WellWrap healthcare application
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wellwrap",
    version="1.0.0",
    author="WellWrap Team",
    author_email="team@wellwrap.com",
    description="AI-powered healthcare application that transforms medical reports into user-friendly insights",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wellwrap",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-flask>=1.2.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "ml": [
            "torch>=1.9.0",
            "transformers>=4.20.0",
            "pytesseract>=0.3.10",
            "pdf2image>=1.16.0",
        ],
        "production": [
            "gunicorn>=21.2.0",
            "psycopg2-binary>=2.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wellwrap=run_app:main",
            "wellwrap-init=init_database:main",
        ],
    },
    include_package_data=True,
    package_data={
        "wellwrap": [
            "frontend/templates/**/*.html",
            "frontend/static/**/*",
            "data/samples/*",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/wellwrap/issues",
        "Source": "https://github.com/yourusername/wellwrap",
        "Documentation": "https://github.com/yourusername/wellwrap/docs",
    },
)