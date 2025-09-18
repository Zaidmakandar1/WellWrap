# Medical Report Simplifier - Build Automation
# Makefile for common development tasks

.PHONY: help install install-dev install-backend install-ml setup test test-backend test-ml lint format clean run run-backend run-ml docker-build docker-run docs

# Default target
help: ## Show this help message
	@echo "Medical Report Simplifier - Available Commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ===== INSTALLATION =====
install: ## Install all dependencies
	pip install -r requirements.txt
	@echo "✅ All dependencies installed successfully"

install-dev: install ## Install development dependencies
	pip install -r tests/requirements.txt
	@echo "✅ Development dependencies installed successfully"

install-backend: ## Install only backend dependencies
	pip install -r backend/requirements.txt
	@echo "✅ Backend dependencies installed successfully"

install-ml: ## Install only ML dependencies
	pip install -r ml/requirements.txt
	@echo "✅ ML dependencies installed successfully"

# ===== PROJECT SETUP =====
setup: install ## Complete project setup
	python scripts/setup/setup_database.py
	python scripts/setup/create_demo_data.py
	@echo "✅ Project setup completed successfully"

# ===== TESTING =====
test: ## Run all tests
	pytest tests/ -v --cov=backend --cov=ml --cov-report=html --cov-report=term
	@echo "✅ All tests completed"

test-backend: ## Run backend tests only
	pytest tests/backend/ -v --cov=backend --cov-report=term
	@echo "✅ Backend tests completed"

test-ml: ## Run ML tests only
	pytest tests/ml/ -v --cov=ml --cov-report=term
	@echo "✅ ML tests completed"

test-integration: ## Run integration tests
	pytest tests/integration/ -v
	@echo "✅ Integration tests completed"

# ===== CODE QUALITY =====
lint: ## Run code linting
	flake8 backend/ ml/ shared/ scripts/
	mypy backend/ ml/ shared/ scripts/
	@echo "✅ Linting completed"

format: ## Format code
	black backend/ ml/ shared/ scripts/ tests/
	@echo "✅ Code formatting completed"

# ===== DEVELOPMENT =====
run: ## Run all services (Flask + Streamlit)
	@echo "🚀 Starting Medical Report Simplifier..."
	@echo "📚 Backend will be available at: http://localhost:5000"
	@echo "🤖 ML Interface will be available at: http://localhost:8501"
	@echo ""
	python scripts/start_development.py

run-backend: ## Run only Flask backend
	@echo "🚀 Starting Flask backend..."
	@echo "📚 Backend will be available at: http://localhost:5000"
	cd backend && python app.py

run-ml: ## Run only Streamlit ML interface
	@echo "🚀 Starting Streamlit ML interface..."
	@echo "🤖 ML Interface will be available at: http://localhost:8501"
	streamlit run ml/streamlit_app/main.py

run-dev: ## Run in development mode with auto-reload
	@echo "🔄 Starting in development mode with auto-reload..."
	export FLASK_ENV=development && make run

# ===== DATABASE =====
db-setup: ## Setup database
	python scripts/setup/setup_database.py
	@echo "✅ Database setup completed"

db-migrate: ## Run database migrations
	cd backend && python -m flask db migrate -m "Auto migration"
	@echo "✅ Database migration created"

db-upgrade: ## Upgrade database
	cd backend && python -m flask db upgrade
	@echo "✅ Database upgraded"

db-seed: ## Seed database with demo data
	python scripts/setup/create_demo_data.py
	@echo "✅ Database seeded with demo data"

# ===== DOCKER =====
docker-build: ## Build Docker image
	docker build -t medical-report-simplifier .
	@echo "✅ Docker image built successfully"

docker-run: ## Run Docker container
	docker run -p 5000:5000 -p 8501:8501 medical-report-simplifier
	@echo "🚀 Docker container started"

docker-compose-up: ## Start with docker-compose
	docker-compose up -d
	@echo "🚀 Docker services started"

docker-compose-down: ## Stop docker-compose services
	docker-compose down
	@echo "🛑 Docker services stopped"

# ===== DOCUMENTATION =====
docs: ## Generate documentation
	cd docs && sphinx-build -b html . _build/html
	@echo "✅ Documentation generated"

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8080
	@echo "📖 Documentation available at: http://localhost:8080"

# ===== DEPLOYMENT =====
deploy-local: setup ## Deploy locally
	@echo "🚀 Deploying locally..."
	make run

deploy-staging: ## Deploy to staging
	python scripts/deployment/deploy.py --env staging
	@echo "✅ Deployed to staging"

deploy-production: ## Deploy to production
	python scripts/deployment/deploy.py --env production
	@echo "✅ Deployed to production"

# ===== MAINTENANCE =====
clean: ## Clean up temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	@echo "✅ Cleanup completed"

clean-data: ## Clean up data files (use with caution)
	rm -rf data/processed/*
	rm -rf logs/*.log
	rm -rf uploads/*
	@echo "⚠️  Data cleanup completed"

# ===== MONITORING =====
health-check: ## Run health check
	python scripts/deployment/health_check.py
	@echo "✅ Health check completed"

logs: ## View application logs
	tail -f logs/app.log

# ===== BACKUP =====
backup: ## Create backup
	python scripts/data_migration/backup_data.py
	@echo "✅ Backup created"

restore: ## Restore from backup (requires BACKUP_FILE variable)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "❌ Please provide BACKUP_FILE variable"; \
		echo "Usage: make restore BACKUP_FILE=path/to/backup.zip"; \
		exit 1; \
	fi
	python scripts/data_migration/restore_data.py $(BACKUP_FILE)
	@echo "✅ Restore completed"

# ===== DEVELOPMENT HELPERS =====
requirements: ## Update requirements files
	pip freeze > requirements.txt
	@echo "✅ Requirements updated"

check-security: ## Run security checks
	bandit -r backend/ ml/ shared/ scripts/
	@echo "✅ Security check completed"

profile: ## Profile the application
	python scripts/performance/profile_app.py
	@echo "✅ Profiling completed"
