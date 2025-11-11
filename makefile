.PHONY: all deps run clean clean-env rebuild help

UNAME_S := $(shell uname -s 2>/dev/null)
VENV := brain

ifeq ($(UNAME_S),Linux)
    PYTHON := $(VENV)/bin/python
    PIP := $(VENV)/bin/pip
else ifeq ($(UNAME_S),Darwin)
    PYTHON := $(VENV)/bin/python
    PIP := $(VENV)/bin/pip
else
    PYTHON := $(VENV)/Scripts/python.exe
    PIP := $(VENV)/Scripts/pip.exe
endif

REQUIREMENTS := torch torchvision torchaudio matplotlib seaborn scikit-learn numpy pandas

all: $(VENV)
	@echo "Checking dependencies in '$(VENV)'..."
	@$(MAKE) deps

$(VENV):
	@echo "Creating virtual environment: $(VENV)"
	python -m venv $(VENV)
	@echo "Virtual environment created."
	@$(PIP) install --upgrade pip

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

run: all
	@echo "Running classification.py..."
	@$(PYTHON) classification.py

clean:
	@echo "Cleaning up temporary and generated files..."
	@$(PYTHON) -c "import pathlib, shutil; \
	for p in pathlib.Path('.').rglob('__pycache__'): shutil.rmtree(p, ignore_errors=True); \
	for ext in ['*.pyc','*.pyo','*.DS_Store','*.png','*.pth']: \
	    [f.unlink() for f in pathlib.Path('.').rglob(ext)]"
	@echo "Cleanup complete."

clean-env:
	@echo "Removing virtual environment '$(VENV)'..."
	@$(PYTHON) -c "import shutil; shutil.rmtree('$(VENV)', ignore_errors=True)"
	@echo "Virtual environment removed."

rebuild: clean-env all
	@echo "Full rebuild complete."

help:
	@echo " Makefile commands for ProjectGreyMatter"
	@echo ""
	@echo "  make           : Create virtual environment '$(VENV)' and install dependencies"
	@echo "  make run       : Run classification.py (auto-checks missing dependencies)"
	@echo "  make clean     : Remove caches and generated files"
	@echo "  make clean-env : Remove the entire virtual environment '$(VENV)'"
	@echo "  make rebuild   : Clean environment + re-create virtual environment + install dependencies"
	@echo "  make help      : Show this help message"