param (
    [string]$Target = "all"
)

$Venv = "brain"
$Python = "$Venv\Scripts\python.exe"
$Pip = "$Venv\Scripts\pip.exe"

# General packages (Excluding PyTorch, which needs a special URL)
$GeneralReqs = @("matplotlib", "seaborn", "scikit-learn", "numpy", "pandas")

function Show-Help {
    Write-Host "Windows PowerShell Script for ProjectGreyMatter" -ForegroundColor Cyan
    Write-Host "Usage: .\manage.ps1 [command]"
    Write-Host "  all        : Create virtual environment and install dependencies"
    Write-Host "  run        : Run classification.py (auto-checks missing dependencies)"
    Write-Host "  clean      : Remove __pycache__, .pyc files, and generated outputs"
    Write-Host "  clean-env  : Remove the entire virtual environment '$Venv'"
    Write-Host "  rebuild    : Clean environment + re-create virtual environment + install dependencies"
}

function Setup-Venv {
    if (-not (Test-Path $Venv)) {
        Write-Host "Creating virtual environment: $Venv..." -ForegroundColor Yellow
        python -m venv $Venv
        & $Python -m pip install --upgrade pip
        Write-Host "Virtual environment created." -ForegroundColor Green
    }
}

function Install-Deps {
    Setup-Venv
    
    Write-Host "Installing/Verifying Intel XPU PyTorch..." -ForegroundColor Yellow
    # We install this separately to ensure you get the XPU wheels, not the default ones!
    & $Pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

    Write-Host "Installing/Verifying general dependencies..." -ForegroundColor Yellow
    foreach ($pkg in $GeneralReqs) {
        & $Pip install $pkg
    }
    Write-Host "Dependencies verified." -ForegroundColor Green
}

function Run-Project {
    Install-Deps
    Write-Host "Running train.py..." -ForegroundColor Yellow
    & $Python train.py
}

function Clean-Files {
    Write-Host "Cleaning up temporary and generated files..." -ForegroundColor Yellow
    
    # Remove pycache directories
    Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    
    # Remove specific file extensions
    Get-ChildItem -Path . -Include *.pyc, *.pyo, .DS_Store, *.png, *.pth -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force
    
    Write-Host "Cleanup complete." -ForegroundColor Green
}

function Clean-Env {
    Write-Host "Removing virtual environment '$Venv'..." -ForegroundColor Yellow
    if (Test-Path $Venv) {
        Remove-Item -Path $Venv -Recurse -Force
    }
    Write-Host "Virtual environment removed." -ForegroundColor Green
}

# Makefile-like target routing
switch ($Target) {
    "all"       { Install-Deps }
    "run"       { Run-Project }
    "clean"     { Clean-Files }
    "clean-env" { Clean-Env }
    "rebuild"   { Clean-Env; Install-Deps }
    "help"      { Show-Help }
    default     { Write-Host "Unknown target: $Target" -ForegroundColor Red; Show-Help }
}