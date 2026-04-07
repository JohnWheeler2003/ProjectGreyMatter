param (
    [string]$Target = "all"
)

# --- Configuration ---
$Venv        = "brain"
$Python      = "$Venv\Scripts\python.exe"
$Pip         = "$Venv\Scripts\pip.exe"
$GeneralReqs = @("matplotlib", "seaborn", "scikit-learn", "numpy", "pandas", "kagglehub")
$DataFolder  = "BrainTumorImages" # Change this to your actual dataset directory name

# --- Core Functions ---

function Show-Help {
    Write-Host "`nWindows PowerShell Script for ProjectGreyMatter" -ForegroundColor Cyan
    $Table = @(
        @{ Target = "all";       Desc = "Full setup: Venv, Deps, and Dataset" }
        @{ Target = "setup";     Desc = "Download and prepare the dataset" }
        @{ Target = "run";       Desc = "Run train.py (auto-checks deps and data)" }
        @{ Target = "clean";     Desc = "Remove temp files and outputs" }
        @{ Target = "rebuild";   Desc = "Fresh wipe and reinstall of everything" }
    )
    $Table | Format-Table -AutoSize
}

function Setup-Venv {
    if (-not (Test-Path $Venv)) {
        Write-Host "--> Creating virtual environment: $Venv..." -ForegroundColor Yellow
        python -m venv $Venv
        & $Python -m pip install --upgrade pip
    }
}

function Install-Deps {
    Setup-Venv
    
    # Check for Intel Hardware
    $intelMatch = Get-CimInstance Win32_VideoController, Win32_PnPEntity | 
                  Where-Object { $_.Name -match "Arc|Core.*Ultra|Data Center GPU|NPU|AI Boost" }

    if ($intelMatch) {
        Write-Host "--> Intel XPU detected. Installing optimized PyTorch..." -ForegroundColor Cyan
        & $Pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
    } else {
        Write-Host "--> No Intel XPU detected. Installing standard PyTorch..." -ForegroundColor Cyan
        & $Pip install torch torchvision torchaudio
    }

    Write-Host "--> Installing general dependencies..." -ForegroundColor Yellow
    foreach ($pkg in $GeneralReqs) { & $Pip install $pkg }
}

function Invoke-SetupDataset {
    if (-not (Test-Path $Python)) { Install-Deps }
    
    Write-Host "--> Preparing dataset via setup_dataset.py..." -ForegroundColor Yellow
    & $Python setup_dataset.py
    Write-Host "--> Dataset ready." -ForegroundColor Green
}

function Run-Project {
    # Sequence Check: Env -> Deps -> Data -> Run
    if (-not (Test-Path $Python)) { Install-Deps }
    
    if (-not (Test-Path $DataFolder)) {
        Write-Host "--> Dataset missing!" -ForegroundColor Red
        Invoke-SetupDataset
    }

    Write-Host "--> Launching Training..." -ForegroundColor Cyan
    & $Python train.py
}

# --- Cleanup Logic ---

function Clean-Files {
    Write-Host "--> Cleaning temporary files..." -ForegroundColor Yellow
    $Targets = @("__pycache__", "*.pyc", "*.pyo", "*.pth", "*.png")
    foreach ($T in $Targets) {
        Get-ChildItem -Path . -Include $T -Recurse | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
    }
}

function Clean-Env {
    if (Test-Path $Venv) { 
        Write-Host "--> Removing $Venv..." -ForegroundColor Red
        Remove-Item -Path $Venv -Recurse -Force 
    }
}

# --- Target Routing ---

switch ($Target) {
    "all"       { Install-Deps; Invoke-SetupDataset }
    "setup"     { Invoke-SetupDataset }
    "run"       { Run-Project }
    "clean"     { Clean-Files }
    "clean-env" { Clean-Env }
    "rebuild"   { Clean-Env; Install-Deps; Invoke-SetupDataset }
    "help"      { Show-Help }
    default     { Write-Host "Unknown target: $Target" -ForegroundColor Red; Show-Help }
}