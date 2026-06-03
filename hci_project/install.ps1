# =============================================================
#  HCI Project - Single-File Windows Installer
#  Works on ANY Windows laptop with Miniconda or Anaconda.
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
    Write-Host "[$n/6] $msg" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor DarkGray
}
function OK   ([string]$m) { Write-Host "  [OK]   $m" -ForegroundColor Green  }
function FAIL ([string]$m) { Write-Host "  [FAIL] $m" -ForegroundColor Red    }
function INFO ([string]$m) { Write-Host "  [INFO] $m" -ForegroundColor Yellow }

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
# STEP 2 - Create or update the hci_env conda environment
# =============================================================================
Write-Step 2 "Creating / updating conda environment '$ENV_NAME'"

if (-not (Test-Path $ENV_YML)) {
    FAIL "environment.yml not found at: $ENV_YML"
    exit 1
}

$envExists = & $condaExe env list 2>&1 | Select-String -Pattern "^\s*$ENV_NAME\s"

if ($envExists) {
    INFO "Environment '$ENV_NAME' already exists - updating packages..."
    & $condaExe env update -n $ENV_NAME -f $ENV_YML --prune 2>&1
} else {
    INFO "Creating new environment '$ENV_NAME' from environment.yml ..."
    INFO "(This downloads ~2-4 GB. Do not close this window.)"
    & $condaExe env create -f $ENV_YML 2>&1
}

if ($LASTEXITCODE -ne 0) {
    FAIL "Conda environment setup failed (exit code $LASTEXITCODE)."
    FAIL "Check your internet connection and try again."
    exit 1
}
OK "Conda environment '$ENV_NAME' is ready."

# =============================================================================
# STEP 3 - PyAudio Windows safety net
# =============================================================================
Write-Step 3 "Checking PyAudio (Windows binary fix)"

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
# STEP 4 - Download NLTK corpora + Whisper + MediaPipe models + DeepFace
# =============================================================================
Write-Step 4 "Running setup_env.py  (NLTK / TextBlob / Whisper / MediaPipe / DeepFace)"

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
# STEP 5 - Verify all critical imports
# =============================================================================
Write-Step 5 "Verifying package imports"

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
# STEP 6 - Summary
# =============================================================================
Write-Step 6 "Summary"

Write-Host ""
if ($failCount -eq 0) {
    Write-Host "  ALL PACKAGES INSTALLED SUCCESSFULLY" -ForegroundColor Green
} else {
    Write-Host "  $failCount package(s) failed to import - see [FAIL] lines above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  To run the project:" -ForegroundColor White
Write-Host "    conda activate $ENV_NAME" -ForegroundColor Cyan
Write-Host "    cd `"$SCRIPT_DIR`"" -ForegroundColor Cyan
Write-Host "    python run_chatbot.py   # chatbot (offline replay works with no hardware)" -ForegroundColor Cyan
Write-Host "    python run_vision.py    # vision  (needs a webcam)" -ForegroundColor Cyan
Write-Host ""
Write-Host "  MANUAL STEP - Download Ollama LLM (do this on fast internet):" -ForegroundColor Yellow
Write-Host "    1. Install Ollama from https://ollama.com/download" -ForegroundColor White
Write-Host "    2. Run: ollama pull llama3   (~4.7 GB)" -ForegroundColor White
Write-Host "    3. Before running chatbot: ollama serve" -ForegroundColor White
Write-Host ""
Write-Host "  (Without Ollama the chatbot works using rule-based responses.)" -ForegroundColor DarkGray
Write-Host ""

# Pause only when run interactively (not when piped or redirected)
if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
    Read-Host "Press ENTER to close"
}