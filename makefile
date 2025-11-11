# Makefile for CNN Project - ProjectGreyMatter
# Cross-platform version (Windows + Linux/macOS)

.PHONY: all deps run clean clean-env rebuild help

# Detect OS
ifeq ($(OS),Windows_NT)
    VENV := brain
    PYTHON := $(VENV)/Scripts/python.exe
    PIP := $(VENV)/Scripts/pip.exe
    RM := del /Q /F
    RMDIR := rmdir /S /Q
    FIND := powershell -Command "Get-ChildItem -Recurse -Include __pycache__,*.pyc,*.pyo,.DS_Store,*.png,*.pth | Remove-Item -Force -Recurse"
else
    VENV := brain
    PYTHON := $(VENV)/bin/python
    PIP := $(VENV)/bin/pip
    RM := rm -f
    RMDIR := rm -rf
    FIND := find . \( -name "__pycache__" -o -name "*.pyc" -o -name "*.pyo" -o -name ".DS_Store" -o -name "*.png" -o -name "*.pth" \) -exec rm -rf {} +
endif

# Python dependencies
REQUIREMENTS := torch torchvision torchaudio matplotlib seaborn scikit-learn numpy pandas

# Default target
all: $(VENV)
	@echo "Checking dependencies in '$(VENV)'..."
	@$(MAKE) deps

# 1. Create virtual environment (if it doesn't exist)
$(VENV):
	@echo "Creating virtual environment: $(VENV)"
	python -m venv $(VENV)
	@echo "Virtual environment created."
	@$(PIP) install --upgrade pip

# 2. Check/install missing dependencies
deps: $(VENV)
	@echo "Checking for missing dependencies..."
	@missing_packages=$$($(PYTHON) -m pip freeze | cut -d '=' -f 1 | tr '\n' ' '); \
	for pkg in $(REQUIREMENTS); do \
		echo "$$missing_packages" | grep -w $$pkg >/dev/null 2>&1 || (echo "Installing missing package: $$pkg" && $(PIP) install $$pkg); \
	done
	@echo "Dependencies verified."

# 3. Run the classification script
run: all
	@echo "Running classification.py..."
	@$(PYTHON) classification.py

# 4. Clean up generated files
clean:
	@echo "Cleaning up temporary and generated files..."
	@$(FIND)
	@echo "Cleanup complete."

# 5. Remove the virtual environment completely
clean-env:
	@echo "Removing virtual environment '$(VENV)'..."
	@$(RMDIR) $(VENV)
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
	@echo "  make clean     : Remove caches, .pyc/.pyo files, .DS_Store, .png, and .pth files"
	@echo "  make clean-env : Remove the entire virtual environment '$(VENV)'"
	@echo "  make rebuild   : Clean + recreate virtual environment + install dependencies"
	@echo "  make help      : Show this help message"
