param (
    [string]$Target = "all",
    [string]$Model = "" # Options: custom_cnn, resnet, vit, all
)

# CONFIGURATION 
$Venv        = "brain"
$Python      = "$Venv\Scripts\python.exe"
$Pip         = "$Venv\Scripts\pip.exe"
$GeneralReqs = @("matplotlib", "seaborn", "scikit-learn", "numpy", "pandas", "kagglehub", "imagehash")
$DataFolder  = "BrainTumorImages"



# CORE FUNCTIONS
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
    
    Write-Host "--> Checking hardware architecture..." -ForegroundColor Yellow

    # 1. Grab all video controllers
    $videoCards = @(Get-CimInstance Win32_VideoController)

    # 2. Check for NVIDIA first (Highest Priority)
    $nvidiaMatch = $videoCards | Where-Object { $_.Name -match "NVIDIA" }

    if ($nvidiaMatch) {
        $deviceName = ($nvidiaMatch.Name | Select-Object -Unique) -join ', '
        Write-Host "--> NVIDIA GPU detected: $deviceName" -ForegroundColor Green
        Write-Host "--> Installing standard PyTorch (Includes CUDA support)..." -ForegroundColor Cyan
        & $Pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    } 
    else {
        # 3. If no NVIDIA, check for Intel XPU hardware
        $pnpEntities = @(Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match "Intel|NPU|AI Boost" })
        $combinedHardware = $videoCards + $pnpEntities
        
        # Using word boundaries (\b) around Arc to prevent matching things like "Architecture" or "Audio Return Channel (ARC)"
        $intelMatch = $combinedHardware | Where-Object { $_.Name -match "\bArc\b|Core.*Ultra|Data Center GPU|NPU|AI Boost" }

        if ($intelMatch) {
            $deviceName = ($intelMatch.Name | Select-Object -Unique) -join ', '
            Write-Host "--> Intel XPU detected: $deviceName" -ForegroundColor Cyan
            Write-Host "--> Installing Intel-optimized PyTorch..." -ForegroundColor Cyan
            & $Pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
        } else {
            Write-Host "--> No specialized GPU detected. Installing standard CPU PyTorch..." -ForegroundColor Cyan
            & $Pip install torch torchvision torchaudio
        }
    }

    Write-Host "--> Installing general dependencies..." -ForegroundColor Yellow
    foreach ($pkg in $GeneralReqs) { & $Pip install $pkg }
}

function Invoke-SetupDataset {
    if (-not (Test-Path $Python)) { Install-Deps }
    
    # Check if folder exists AND has files in it
    if (Test-Path $DataFolder) {
        $files = Get-ChildItem -Path $DataFolder -Recurse -File | Select-Object -First 1
        if ($null -ne $files) {
            Write-Host "--> Dataset already exists in $DataFolder. Skipping download." -ForegroundColor Green
            return
        }
    }    

    Write-Host "--> Dataset missing or empty. Preparing dataset via ultimate_setup_dataset.py..." -ForegroundColor Yellow
    & $Python ultimate_setup_dataset.py
    Write-Host "--> Dataset ready." -ForegroundColor Green
}

function Run-Project {
    param([string]$TargetModel)

    # Prompt the user if no model was provided in the command line
    if ([string]::IsNullOrWhiteSpace($TargetModel)) {
        Write-Host "`nNo model specified." -ForegroundColor Yellow
        $TargetModel = Read-Host "Which model would you like to run? [custom_cnn, resnet, vit, all]"
    }

    # Sequence Check: Env -> Deps -> Data -> Run
    if (-not (Test-Path $Python)) { Install-Deps }
    
    if (-not (Test-Path $DataFolder)) {
        Write-Host "--> Dataset missing!" -ForegroundColor Red
        Invoke-SetupDataset
    }

    # Determine which models to run
    $modelsToRun = @()
    if ($TargetModel -eq "all") {
        $modelsToRun = @("custom_cnn", "resnet", "vit")
    } else {
        $modelsToRun = @($TargetModel)
    }

    # Loop through and execute
    foreach ($m in $modelsToRun) {
        Write-Host "--> Launching Pipeline for Model: $m" -ForegroundColor Magenta
        Write-Host "--> Launching Training..." -ForegroundColor Cyan
        & $Python train.py --model $m

        # Error Check
        if ($LASTEXITCODE -ne 0) {
            Write-Host " [!] Training failed for $m with exit code $LASTEXITCODE. Skipping evaluation." -ForegroundColor Red
            continue 
        }

        Write-Host "--> Launching Evaluation..." -ForegroundColor Cyan
        & $Python evaluate.py --model $m
    }
}



# CLEANUP LOGIC

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

# TARGET ROUTING

switch ($Target) {
    "all"       { Install-Deps; Invoke-SetupDataset }
    "setup"     { Invoke-SetupDataset }
    "run"       { Run-Project -TargetModel $Model } 
    "clean"     { Clean-Files }
    "clean-env" { Clean-Env }
    "rebuild"   { Clean-Env; Install-Deps; Invoke-SetupDataset }
    "help"      { Show-Help }
    default     { Write-Host "Unknown target: $Target" -ForegroundColor Red; Show-Help }
}