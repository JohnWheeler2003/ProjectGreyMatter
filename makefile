
# Makefile for CNN Project - ProjectGreyMatter

.PHONY: all deps run clean clean-env rebuild help

# Virtual environment name
VENV := brain
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Python dependencies
REQUIREMENTS := torch torchvision torchaudio matplotlib seaborn scikit-learn numpy pandas


# Default target

all: $(VENV)
	@echo "Checking dependencies in '$(VENV)'..."
	@$(MAKE) deps

# 1. Create virtual environment (if it doesn't exist)

$(VENV):
	@echo "Creating virtual environment: $(VENV)"
	python3 -m venv $(VENV)
	@echo "Virtual environment created."
	@$(PIP) install --upgrade pip

# 2. Check/install missing dependencies

deps: $(VENV)
	@echo "Checking for missing dependencies..."
	@missing_packages=$$($(PYTHON) -m pip freeze | cut -d '=' -f 1 | tr '\n' ' '); \
	for pkg in $(REQUIREMENTS); do \
		if ! echo "$$missing_packages" | grep -w $$pkg >/dev/null 2>&1; then \
			echo "Installing missing package: $$pkg"; \
			$(PIP) install $$pkg; \
		fi; \
	done
	@echo "Dependencies verified."

# 3. Run the classification script
#    - Automatically checks/install missing dependencies

run: all
	@echo "Running classification.py..."
	@$(PYTHON) classification.py

# 4. Clean up generated files
#    - Removes caches, compiled files, and .png outputs

clean:
	@echo "Cleaning up temporary and generated files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@find . -name ".DS_Store" -delete
	@find . -name "*.png" -delete
	@find . -name "*.pth" -delete
	@echo " Cleanup complete."

# 5. Remove the virtual environment completely

clean-env:
	@echo "Removing virtual environment '$(VENV)'..."
	rm -rf $(VENV)
	@echo "Virtual environment removed."

# 6. Full rebuild (clean + recreate environment)

rebuild: clean-env all
	@echo "Full rebuild complete."

# 7. Help command

help:
	@echo " Makefile commands for ProjectGreyMatter"
	@echo ""
	@echo "  make           : Create virtual environment '$(VENV)' and install dependencies"
	@echo "  make run       : Run classification.py (auto-checks missing dependencies)"
	@echo "  make clean     : Remove __pycache__, .pyc/.pyo files, .DS_Store, and generated .png files"
	@echo "  make clean-env : Remove the entire virtual environment '$(VENV)'"
	@echo "  make rebuild   : Clean environment + re-create virtual environment + install dependencies"
	@echo "  make help      : Show this help message"