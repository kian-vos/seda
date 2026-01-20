.PHONY: install dev test lint format clean run-dashboard collect analyze seeds

# Install production dependencies
install:
	pip install -e .

# Install with development dependencies
dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=seda --cov-report=html

# Lint code
lint:
	ruff check seda scripts dashboard tests
	mypy seda scripts

# Format code
format:
	black seda scripts dashboard tests
	ruff check --fix seda scripts dashboard tests

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run Streamlit dashboard
run-dashboard:
	streamlit run dashboard/app.py

# Initialize seed accounts
seeds-init:
	python -m scripts.seed_accounts init

# List seed accounts
seeds-list:
	python -m scripts.seed_accounts list

# Collect data from seed accounts
collect-seeds:
	python -m scripts.collect seeds

# Collect single account
collect-account:
	@echo "Usage: make collect-account USERNAME=username"
	python -m scripts.collect account $(USERNAME)

# Run full analysis pipeline
analyze-all:
	python -m scripts.analyze all

# Run specific analysis
analyze-features:
	python -m scripts.analyze features

analyze-bot:
	python -m scripts.analyze bot

analyze-stance:
	python -m scripts.analyze stance

analyze-coordination:
	python -m scripts.analyze coordination

# Export results
export:
	@echo "Usage: make export FILE=output.csv"
	python -m scripts.analyze export $(FILE)

# Create database backup
backup:
	cp data/seda.db data/seda_backup_$$(date +%Y%m%d_%H%M%S).db

# Setup new project
setup: dev seeds-init
	@echo "SEDA setup complete. Configure .env with your API keys."
