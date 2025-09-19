# WellWrap Makefile
# Convenient commands for development and deployment

.PHONY: help install dev test clean run init-db lint format docker-build docker-run

# Default target
help:
	@echo "WellWrap - AI-Powered Healthcare Application"
	@echo "============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Install development dependencies"
	@echo "  init-db     - Initialize database with demo data"
	@echo "  run         - Run the application"
	@echo "  test        - Run tests"
	@echo "  test-ml     - Run ML comprehensive tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run Docker container"
	@echo ""

# Installation
install:
	@echo "📦 Installing WellWrap dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete!"

dev:
	@echo "🛠️ Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest pytest-flask pytest-cov black flake8 mypy
	@echo "✅ Development setup complete!"

# Database
init-db:
	@echo "📂 Initializing WellWrap database..."
	python init_database.py --init
	@echo "✅ Database initialized!"

# Running
run:
	@echo "🚀 Starting WellWrap application..."
	python run_app.py

run-enhanced:
	@echo "🚀 Starting WellWrap (Enhanced version)..."
	python enhanced_app.py

run-minimal:
	@echo "🚀 Starting WellWrap (Minimal version)..."
	python minimal_app.py

# Testing
test:
	@echo "🧪 Running WellWrap tests..."
	python -m pytest tests/ -v

test-ml:
	@echo "🧪 Running ML comprehensive tests..."
	python test_ml_comprehensive.py

test-upload:
	@echo "🧪 Testing upload functionality..."
	python test_upload_functionality.py

# Code quality
lint:
	@echo "🔍 Running code linting..."
	flake8 backend/ frontend/ ml/ --max-line-length=88 --ignore=E203,W503
	mypy backend/ --ignore-missing-imports

format:
	@echo "🎨 Formatting code with black..."
	black backend/ ml/ *.py --line-length=88

# Cleanup
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "✅ Cleanup complete!"

# Docker
docker-build:
	@echo "🐳 Building WellWrap Docker image..."
	docker build -t wellwrap:latest .

docker-run:
	@echo "🐳 Running WellWrap Docker container..."
	docker run -p 5000:5000 wellwrap:latest

# Development helpers
setup-dev: install dev init-db
	@echo "🎉 WellWrap development environment ready!"
	@echo "Run 'make run' to start the application"

# Production deployment
deploy-prep: clean lint test
	@echo "🚀 WellWrap ready for deployment!"

# Quick start
quick-start: install init-db run

# Show project status
status:
	@echo "📊 WellWrap Project Status"
	@echo "========================="
	@echo "Python version: $(shell python --version)"
	@echo "Pip packages: $(shell pip list | wc -l) installed"
	@echo "Database: $(shell test -f healthcare.db && echo 'Present' || echo 'Missing')"
	@echo "Upload folder: $(shell test -d uploads && echo 'Present' || echo 'Missing')"
	@echo "Tests: $(shell find tests/ -name "*.py" 2>/dev/null | wc -l) test files"