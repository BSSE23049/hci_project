# =============================================================
#  HCI Project - Single-File Windows Installer
#  Works on ANY Windows laptop with Miniconda or Anaconda.
#  Safe to re-run: already-installed packages are skipped,
#  only missing or failed packages are downloaded again.
#
#  HOW TO RUN:
#    Option A: Right-click this file -> "Run with PowerShell"
#    Option B: Open PowerShell in this folder and type:
#              Set-ExecutionPolicy -Scope Process Bypass
#              .\install.ps1
# =============================================================

$ErrorActionPreference = "Continue"
$ENV_NAME   = "hci_env"
$SCRIPT_DIR = $PSScriptRoot
$ENV_YML    = Join-Path $SCRIPT_DIR "environment.yml"
$SETUP_PY   = Join-Path $SCRIPT_DIR "setup_env.py"

# --- Colour helpers -----------------------------------------------------------
function Write-Step ([int]$n, [string]$msg) {
    Write-Host ""
    Write-Host "[$n/7] $msg" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor DarkGray
}
function OK   ([string]$m) { Write-Host "  [OK]   $m" -ForegroundColor Green  }
function FAIL ([string]$m) { Write-Host "  [FAIL] $m" -ForegroundColor Red    }
function INFO ([string]$m) { Write-Host "  [INFO] $m" -ForegroundColor Yellow }

# Helper: pip install with binary preference + retries
function Pip-Install ([string[]]$packages) {
    $pkgList = $packages -join " "
    & $condaExe run -n $ENV_NAME pip install `
        --prefer-binary `
        --retries 5 `
        --timeout 120 `
        $packages 2>&1
}

# =============================================================================
# STEP 1 - Locate conda executable
# =============================================================================
Write-Step 1 "Locating conda"

$condaExe = $null

try {
    $found = Get-Command conda -ErrorAction Stop
    $condaExe = $found.Source
    OK "conda found in PATH: $condaExe"
} catch {
    INFO "conda not in PATH - searching common install locations..."
}

if (-not $condaExe) {
    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\Miniconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe",
        "C:\miniconda3\Scripts\conda.exe",
        "C:\Miniconda3\Scripts\conda.exe",
        "C:\anaconda3\Scripts\conda.exe",
        "C:\Anaconda3\Scripts\conda.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $condaExe = $c
            OK "conda found at: $condaExe"
            break
        }
    }
}

if (-not $condaExe) {
    FAIL "Conda not found on this machine."
    Write-Host ""
    Write-Host "  Install Miniconda from:" -ForegroundColor Red
    Write-Host "  https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor White
    Write-Host ""
    Read-Host "Press ENTER to exit"
    exit 1
}

# =============================================================================
# STEP 2 - Configure pip for resilient downloads (retries + binary preference)
#          Done before conda env create/update so all pip calls inherit it.
# =============================================================================
Write-Step 2 "Configuring pip (retries + binary wheels)"

# These settings apply inside hci_env's pip only (not global).
# prefer-binary  -> use pre-built wheels, avoids slow source compilation
# retries        -> retry each chunk up to 5 times on network glitches
# timeout        -> wait up to 120 s per response (default is 15 s)
& $condaExe run -n $ENV_NAME pip config set global.prefer-binary true  2>&1 | Out-Null
& $condaExe run -n $ENV_NAME pip config set global.retries 5            2>&1 | Out-Null
& $condaExe run -n $ENV_NAME pip config set global.timeout 120          2>&1 | Out-Null
OK "pip configured: prefer-binary=true, retries=5, timeout=120s"

# =============================================================================
# STEP 3 - Create or update the hci_env conda environment
#          Re-run safe: conda env update skips packages that are already at
#          the correct version ("Requirement already satisfied").
# =============================================================================
Write-Step 3 "Creating / updating conda environment '$ENV_NAME'"

if (-not (Test-Path $ENV_YML)) {
    FAIL "environment.yml not found at: $ENV_YML"
    exit 1
}

$envExists = & $condaExe env list 2>&1 | Select-String -Pattern "^\s*$ENV_NAME\s"

if ($envExists) {
    INFO "Environment '$ENV_NAME' already exists - updating (skips installed packages)..."
    & $condaExe env update -n $ENV_NAME -f $ENV_YML --prune 2>&1
} else {
    INFO "Creating new environment '$ENV_NAME' from environment.yml ..."
    INFO "(First install: ~2-4 GB download. Re-runs are much faster.)"
    & $condaExe env create -f $ENV_YML 2>&1
}

if ($LASTEXITCODE -ne 0) {
    FAIL "Conda step failed (exit code $LASTEXITCODE)."
    Write-Host ""
    Write-Host "  If the error was a network drop (IncompleteRead / Connection broken):" -ForegroundColor Yellow
    Write-Host "  -> Just re-run this script. Already-installed packages will be skipped." -ForegroundColor Yellow
    Write-Host "  -> Make sure you have a stable internet connection." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press ENTER to exit"
    exit 1
}
OK "Conda environment '$ENV_NAME' is ready."

# =============================================================================
# STEP 4 - Install large pip packages with explicit retry settings
#          This catches any package that failed in the conda pip step above
#          (e.g. tensorflow interrupted mid-download) and retries it cleanly.
#          Already-installed packages are skipped instantly.
# =============================================================================
Write-Step 4 "Ensuring large pip packages (tensorflow / torch / whisper)"

INFO "Checking tensorflow (350 MB - most common interruption point)..."
$tfOK = & $condaExe run -n $ENV_NAME python -c "import tensorflow; print('ok')" 2>&1
if ($tfOK -match "ok") {
    OK "tensorflow already installed - skipping download."
} else {
    INFO "tensorflow missing - installing with retry settings (5 retries, 120s timeout)..."
    Pip-Install @("tensorflow", "tf-keras", "deepface")
    $tfOK2 = & $condaExe run -n $ENV_NAME python -c "import tensorflow; print('ok')" 2>&1
    if ($tfOK2 -match "ok") {
        OK "tensorflow installed successfully."
    } else {
        FAIL "tensorflow still failed. Check your internet and re-run."
    }
}

INFO "Checking torch (used by Whisper)..."
$torchOK = & $condaExe run -n $ENV_NAME python -c "import torch; print('ok')" 2>&1
if ($torchOK -match "ok") {
    OK "torch already installed - skipping download."
} else {
    INFO "torch missing - installing..."
    Pip-Install @("openai-whisper")
}

# =============================================================================
# STEP 5 - PyAudio Windows safety net
# =============================================================================
Write-Step 5 "Checking PyAudio (Windows binary fix)"

$pyaudioOK = & $condaExe run -n $ENV_NAME python -c "import pyaudio; print('ok')" 2>&1
if ($pyaudioOK -match "ok") {
    OK "PyAudio imports correctly."
} else {
    INFO "PyAudio conda build failed - trying pipwin fallback..."
    & $condaExe run -n $ENV_NAME pip install pipwin 2>&1 | Out-Null
    & $condaExe run -n $ENV_NAME pipwin install pyaudio 2>&1
    $pyaudioOK2 = & $condaExe run -n $ENV_NAME python -c "import pyaudio; print('ok')" 2>&1
    if ($pyaudioOK2 -match "ok") {
        OK "PyAudio installed via pipwin."
    } else {
        INFO "PyAudio still failing - voice input will be unavailable, but everything else will run."
    }
}

# =============================================================================
# STEP 6 - Download NLTK corpora + Whisper + MediaPipe models + DeepFace
# =============================================================================
Write-Step 6 "Running setup_env.py  (NLTK / TextBlob / Whisper / MediaPipe / DeepFace)"

if (-not (Test-Path $SETUP_PY)) {
    FAIL "setup_env.py not found at: $SETUP_PY"
    exit 1
}

& $condaExe run -n $ENV_NAME python $SETUP_PY 2>&1
if ($LASTEXITCODE -ne 0) {
    INFO "setup_env.py exited with code $LASTEXITCODE - check output above for details."
} else {
    OK "setup_env.py completed."
}

# =============================================================================
# STEP 7 - Verify all critical imports
# =============================================================================
Write-Step 7 "Verifying package imports"

$checks = @(
    @{ pkg = "cv2";         label = "OpenCV"             },
    @{ pkg = "mediapipe";   label = "MediaPipe"          },
    @{ pkg = "numpy";       label = "NumPy"              },
    @{ pkg = "whisper";     label = "Whisper"            },
    @{ pkg = "nltk";        label = "NLTK"               },
    @{ pkg = "textblob";    label = "TextBlob"           },
    @{ pkg = "sklearn";     label = "scikit-learn"       },
    @{ pkg = "sounddevice"; label = "sounddevice"        },
    @{ pkg = "pyttsx3";     label = "pyttsx3"            },
    @{ pkg = "requests";    label = "requests"           },
    @{ pkg = "PIL";         label = "Pillow"             },
    @{ pkg = "deepface";    label = "DeepFace"           },
    @{ pkg = "tf_keras";    label = "tf-keras (emotion)" },
    @{ pkg = "pyaudio";     label = "PyAudio"            },
    @{ pkg = "ollama";      label = "Ollama client"      }
)

$failCount = 0
foreach ($c in $checks) {
    $result = & $condaExe run -n $ENV_NAME python -c "import $($c.pkg); print('ok')" 2>&1
    if ($result -match "ok") {
        OK "$($c.label)"
    } else {
        FAIL "$($c.label)  <- import failed"
        $failCount++
    }
}

# =============================================================================
# Summary
# =============================================================================
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor DarkGray
Write-Host ""
if ($failCount -eq 0) {
    Write-Host "  ALL PACKAGES INSTALLED SUCCESSFULLY" -ForegroundColor Green
} else {
    Write-Host "  $failCount package(s) failed - see [FAIL] lines above." -ForegroundColor Yellow
    Write-Host "  Re-run this script to retry only the failed packages." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  To run the project:" -ForegroundColor White
Write-Host "    conda activate $ENV_NAME" -ForegroundColor Cyan
Write-Host "    cd `"$SCRIPT_DIR`"" -ForegroundColor Cyan
Write-Host "    python run_chatbot.py   # chatbot (offline replay needs no hardware)" -ForegroundColor Cyan
Write-Host "    python run_vision.py    # vision  (needs a webcam)" -ForegroundColor Cyan
Write-Host ""
Write-Host "  MANUAL STEP - Download Ollama LLM (fast internet, ~4.7 GB):" -ForegroundColor Yellow
Write-Host "    1. Install Ollama: https://ollama.com/download" -ForegroundColor White
Write-Host "    2. Run: ollama pull llama3" -ForegroundColor White
Write-Host "    3. Before chatbot: ollama serve" -ForegroundColor White
Write-Host ""
Write-Host "  (Without Ollama the chatbot works with rule-based responses.)" -ForegroundColor DarkGray
Write-Host ""

if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
    Read-Host "Press ENTER to close"
}
