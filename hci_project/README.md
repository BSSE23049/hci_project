# HCI Project — Vision & Chatbot System

A complete Human-Computer Interaction project containing **two independent applications**
that share a small set of utilities:

| App | Launcher | What it does |
|-----|----------|--------------|
| **Vision App** | `python run_vision.py` | Real-time computer vision: lips, eyes, face emotion, hand gestures |
| **Chatbot App** | `python run_chatbot.py` | University info chatbot **+** NEXUS wellbeing advisor (text/voice) |

Everything runs on **Windows** inside a **conda environment** named `hci_env` using **Python 3.10**.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Installation (detailed)](#2-installation-detailed)
3. [How to Run Each App](#3-how-to-run-each-app)
4. [Project Structure — What Every File Does](#4-project-structure--what-every-file-does)
5. [How the Code Flows](#5-how-the-code-flows)
6. [Where to Make Changes (Config-Driven Design)](#6-where-to-make-changes-config-driven-design)
7. [Verified Working Status](#7-verified-working-status)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Quick Start

If conda is already installed, the fastest path is the **one-file installer**:

```powershell
# In PowerShell, from the hci_project folder:
Set-ExecutionPolicy -Scope Process Bypass -Force
.\install.ps1
```

This creates `hci_env`, installs every dependency, downloads the NLTK/Whisper models,
and verifies all 14 critical imports. Then:

```powershell
conda activate hci_env
python run_chatbot.py     # works immediately (offline replay needs no hardware)
python run_vision.py      # needs a webcam
```

---

## 2. Installation (detailed)

### Prerequisites
- **Miniconda or Anaconda** — https://docs.conda.io/en/latest/miniconda.html
- **Ollama** (optional, for LLM chatbot replies) — https://ollama.com/download
- A webcam (for the vision app) and microphone (for voice chat input)

### Option A — Automatic (recommended)

`install.ps1` is generic and works on any Windows laptop. It:
- finds conda automatically in 14 common locations (even if not on PATH),
- creates `hci_env` (or updates it if it already exists),
- applies the Windows PyAudio binary fix if needed,
- runs `setup_env.py` to fetch NLTK corpora + the Whisper `base` model,
- verifies every package import and prints a pass/fail summary.

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\install.ps1
```

> Or simply **right-click `install.ps1` → Run with PowerShell**.

### Option B — Manual (step by step)

```powershell
# 1. Create the environment from the spec file
conda env create -f environment.yml

# 2. Activate it (do this in EVERY new terminal)
conda activate hci_env

# 3. Verify core imports
python -c "import cv2, mediapipe, whisper, nltk; print('Core imports OK')"

# 4. Download models/corpora (NLTK + TextBlob + Whisper base)
python setup_env.py

# 5. (Optional) Download the LLM for the chatbot — needs internet, ~4.7 GB
ollama pull llama3
```

> A full command reference (including every troubleshooting command) lives in
> **`INSTALL_COMMANDS.txt`**.

### Activating the environment

Every time you open a new terminal you must activate the env before running anything:

```powershell
conda activate hci_env
```

Your prompt will change to show `(hci_env)`. To leave it later: `conda deactivate`.

---

## 3. How to Run Each App

### Vision App

```powershell
conda activate hci_env
python run_vision.py
```

You will see a **console menu**:

```
Select Input Source:        Select Detection Mode:
  1. Live Webcam              1. Lips Detection
  2. Video File              2. Eyes Detection
  3. Image File              3. Face Detection
                             4. Hand Detection (asks: enable Gesture Game? y/n)
                             0. Exit
```

A separate OpenCV window opens for the live video. **Press `ESC` inside that window**
to stop detection and return to the menu.

### Chatbot App

The chatbot has two modes selected by `CHATBOT_MODE` in `chatbot/chatbot_config.py`.

```powershell
conda activate hci_env

# (Optional, for AI replies) start Ollama in a SECOND terminal first:
ollama serve

# Then run:
python run_chatbot.py
```

Menu:

```
  1. Start conversation
  2. Switch mode  (university / nexus)
  3. Switch input (text / voice / hybrid)
  4. Exit
  5. Run offline session replay     <- only shown in NEXUS mode
```

**No hardware needed to demo NEXUS:** choose `2` to switch to NEXUS, then `5` to run the
offline replay — it processes a built-in 10-message student conversation and prints the
full intelligence report. (This is the exact path verified in section 7.)

---

## 4. Project Structure — What Every File Does

```
hci_project/
│
├── environment.yml          conda environment definition (all dependencies)
├── install.ps1              one-file Windows installer + verifier
├── setup_env.py             downloads NLTK corpora, TextBlob data, Whisper model
├── INSTALL_COMMANDS.txt     copy-paste command reference & troubleshooting
├── README.md               this file
│
├── run_vision.py            launcher → vision.vision_main.main()  (pure entry point)
├── run_chatbot.py           launcher → chatbot.chatbot_main.main() (pure entry point)
│
├── shared/                  utilities used by BOTH apps
│   ├── audio_utils.py       capture_audio() from mic + transcribe() via Whisper
│   └── llm_utils.py         query_ollama() — the ONLY place that talks to the LLM
│
├── vision/                  the Vision App
│   ├── vision_config.py     ⚙ ALL vision flags, thresholds, gesture/letter maps
│   ├── landmarks.py         central MediaPipe landmark indices (lips/eyes/nose/fingers)
│   ├── mp_tasks.py          model download + bgr_to_mp_image + landmarker factories
│   ├── vision_utils.py      geometry + drawing helpers + shared detection_loop
│   ├── models/              auto-downloaded .task model bundles (created on first run)
│   ├── lips_module.py       MAR, smile detection, lip-sync counter
│   ├── eyes_module.py       EAR, blink counter, drowsiness detector
│   ├── face_module.py       landmark bbox + head pose + DeepFace emotion + mood history
│   ├── hand_module.py       finger counting, gesture/letter lookup, gesture game
│   └── vision_main.py       menu + source selection + dispatch to each module
│
└── chatbot/                 the Chatbot App
    ├── chatbot_config.py     ⚙ ALL chatbot + NEXUS flags, intents, keywords, rules
    ├── spam_module.py        is_spam() — keyword + domain-relevance filter
    ├── intent_module.py      classify_intent() — keyword-scored intent detection
    ├── response_module.py    generate_university_response() (static or LLM) + TTS
    ├── nexus_wellbeing.py    assess_wellbeing() via VADER, trajectory, crisis alert
    ├── nexus_support.py      classify_support_need(), transition log, nexus_respond()
    ├── nexus_report.py       generate_intelligence_report() — risk score + summary
    └── chatbot_main.py       menu + university loop + NEXUS loop + offline replay
```

The `⚙` files are the **only files you normally edit** to change behaviour (see section 6).

### Module responsibilities in one line each

**shared/**
- `audio_utils.py` — `capture_audio(seconds, rate)` records the mic; `transcribe(audio)` returns `{text, language, confidence}` using Whisper. Degrades gracefully if libraries are missing.
- `llm_utils.py` — `query_ollama(prompt, system_prompt, model, base_url)` posts to the local Ollama server. The **system_prompt** is where the "scenario/role" lives. Returns `None` (never crashes) if Ollama is down, triggering rule-based fallback everywhere.

**vision/**
- `vision_config.py` — feature on/off flags, all numeric thresholds, landmarker confidences, `GESTURE_MAP` (ASCII tags), `LETTER_GESTURE_MAP`, game settings.
- `landmarks.py` — the single source of truth for landmark index numbers (lips, eyes, nose, finger tips/PIPs). Tweak which points a feature uses by editing only this file.
- `mp_tasks.py` — the **MediaPipe Tasks API** infrastructure: model paths, first-run `.task` download into `models/`, `bgr_to_mp_image()`, and the `make_face_landmarker()` / `make_hand_landmarker()` factories (read confidences/counts from config).
- `vision_utils.py` — shared geometry (`euclidean`, `lm_to_px`, `compute_aspect_ratio`), drawing helpers (`draw_text` with drop-shadow, `draw_red_border`, `draw_red_banner`, `draw_esc_hint`), source management, and the master `detection_loop()` used by every module (ESC = menu, 'q' = quit, videos replay).
- `lips_module.py` — `compute_mar`, `detect_smile`, and `run_lips(source)`.
- `eyes_module.py` — `compute_ear` and `run_eyes(source)` (blink counter + drowsiness banner).
- `face_module.py` — `classify_head_pose`, `bbox_from_landmarks` (face box straight from landmarks, no Haar), and `run_face(source)` (DeepFace emotion, throttled, + rolling recent-mood).
- `hand_module.py` — `count_fingers`, `classify_gesture` (reads `GESTURE_MAP` only), `detect_letter` (reads `LETTER_GESTURE_MAP`), and `run_hands(source, game_mode)`.
- `vision_main.py` — console menu; `_choose_source()` returns the right type (int/str/ndarray); each module is dispatched through a `MODULES` registry after its `ENABLE_*` flag is checked. Uses the official Tasks drawing API for landmark overlays.

**chatbot/**
- `chatbot_config.py` — mode switch, LLM/Whisper flags, `UNIVERSITY_INTENTS`, spam/domain keywords, `NEXUS_WELLBEING_SCALE`, `NEXUS_SUPPORT_KEYWORDS`, `NEXUS_RESPONSE_RULES`, risk thresholds.
- `spam_module.py` — `is_spam(text)` → `(bool, reason)`. Two stages: spam keywords, then domain-relevance check.
- `intent_module.py` — `classify_intent(text)` → `{intent, confidence, pattern, all_scores}` by keyword counting.
- `response_module.py` — `generate_university_response(intent, text)`: static response if `USE_LLM=False` or Ollama down, otherwise LLM. `speak_response(text)` does TTS (or prints if unavailable).
- `nexus_wellbeing.py` — `assess_wellbeing(text)` (VADER → tier), `compute_trajectory(log)`, `check_and_alert(result, turn)` (crisis banner).
- `nexus_support.py` — `classify_support_need(text)`, `log_support_transition(...)` (tracks topic shifts/escalations), `nexus_respond(...)` (matches `NEXUS_RESPONSE_RULES` top-to-bottom).
- `nexus_report.py` — `generate_intelligence_report(...)` computes the risk score and prints the counsellor report.
- `chatbot_main.py` — the menu and the two conversation loops, plus `run_offline_replay()`.

---

## 5. How the Code Flows

### Vision App flow

```
run_vision.py
   └─ vision_main.main()                # console menu loop
        ├─ ensure_models()              # download .task bundles on first run
        ├─ _choose_source()             # webcam(int) / video(str) / image(ndarray)
        └─ run_lips/eyes/face/hands()   # picked from MODULES registry (ENABLE_* checked)
              ├─ make_face/hand_landmarker(running_mode)   # mp_tasks factory (reads config)
              └─ vision_utils.detection_loop(source, process):
                    read → process(frame): detect → compute_* → draw_* (Tasks drawing API)
                    → imshow → ESC = menu / 'q' = quit / video auto-replays
```

### University chatbot flow (per turn)

```
input (text / voice→Whisper)
   → spam_module.is_spam()         # blocked? show message, next turn
   → intent_module.classify_intent()   # prints intent + score + keywords
   → response_module.generate_university_response()
            ├─ USE_LLM=False or Ollama down → static answer from UNIVERSITY_INTENTS
            └─ USE_LLM=True → query_ollama(system_prompt=role+intent)
   → response_module.speak_response()   # TTS
```

### NEXUS wellbeing flow (per turn, with session memory)

```
input (text / voice→Whisper)
   → nexus_wellbeing.assess_wellbeing()   # VADER → tier + emoji + score
   → nexus_wellbeing.check_and_alert()    # crisis banner if CRISIS tier
   → nexus_support.classify_support_need()# primary + all detected categories
   → nexus_support.log_support_transition()# records topic shifts / escalations
   → nexus_support.nexus_respond()        # first matching rule in NEXUS_RESPONSE_RULES
   → append to session_log / wellbeing_log / support_log / transition_log
on exit:
   → nexus_report.generate_intelligence_report()  # trajectory + risk score + action
```

### Risk score formula (in `nexus_report.py`)

```
base_risk          = 20
wellbeing_penalty  = abs(avg_wellbeing_score) * 40
at_risk_penalty    = at_risk_count * 15
escalation_penalty = escalation_count * 10
word_count_factor  = 5 if avg_words_per_turn > 20 else 0
risk_score         = clamp(round(sum), 0, 100)

>= 70 → URGENT_REFERRAL   |   >= 40 → FOLLOW_UP   |   else → NO_ACTION
```

---

## 6. Where to Make Changes (Config-Driven Design)

**The core design rule: behaviour changes only require editing a config file.**
No thresholds, keywords, gesture names, or response strings are hardcoded in logic files.

| I want to… | Edit this file | Change this |
|------------|----------------|-------------|
| Turn a vision feature on/off | `vision/vision_config.py` | `ENABLE_LIPS/EYES/FACE/HAND_DETECTION = True/False` |
| Remap a gesture (e.g. 2 fingers) | `vision/vision_config.py` | `GESTURE_MAP` dict entry (label + ASCII tag) |
| Add a new ASL letter | `vision/vision_config.py` | add to `LETTER_GESTURE_MAP` |
| Turn the gesture game on/off | `vision/vision_config.py` | `ENABLE_GESTURE_GAME = True/False` |
| Make drowsiness trigger faster | `vision/vision_config.py` | `DROWSY_FRAMES_TRIGGER` |
| Tune detection sensitivity | `vision/vision_config.py` | `FACE/HAND_DETECTION_CONFIDENCE` |
| Change which landmark a feature uses | `vision/landmarks.py` | the relevant index constant |
| Fix "webcam not found" | `vision/vision_config.py` | `WEBCAM_INDEX = 0 → 1` |
| Switch chatbot ↔ NEXUS | `chatbot/chatbot_config.py` | `CHATBOT_MODE = "university" / "nexus"` |
| Make chatbot fully rule-based | `chatbot/chatbot_config.py` | `USE_LLM = False` |
| Change text/voice/hybrid input | `chatbot/chatbot_config.py` | `DEFAULT_INPUT_MODE` |
| Add a new university intent | `chatbot/chatbot_config.py` | add entry to `UNIVERSITY_INTENTS` |
| Edit a static answer | `chatbot/chatbot_config.py` | that intent's `response_static` |
| Add/adjust spam words | `chatbot/chatbot_config.py` | `SPAM_KEYWORDS` / `DOMAIN_KEYWORDS` |
| Re-tune wellbeing tiers | `chatbot/chatbot_config.py` | `NEXUS_WELLBEING_SCALE` |
| Add a support category | `chatbot/chatbot_config.py` | add to `NEXUS_SUPPORT_KEYWORDS` |
| Change a NEXUS reply | `chatbot/chatbot_config.py` | edit a tuple in `NEXUS_RESPONSE_RULES` |
| Change crisis line / risk cutoffs | `chatbot/chatbot_config.py` | `NEXUS_CRISIS_LINE`, `NEXUS_RISK_*` |

Adding a new intent, gesture, letter, support category, or response rule is **add one
line/entry** — the classifiers and lookups rebuild from config automatically.

---

## 7. Verified Working Status

Last verified on this machine (Python 3.10.20, Windows 11):

**All 20 dependencies installed** — opencv-python 4.13.0, mediapipe 0.10.35, deepface 0.0.100,
numpy 2.2.6, openai-whisper, nltk 3.9.4, textblob 0.20.0, scikit-learn 1.7.2, sounddevice,
soundfile, PyAudio 0.2.11, SpeechRecognition, pydub, ollama, anthropic, python-dotenv,
requests, Pillow, gradio, pyttsx3.

**All 16 project modules import cleanly** (shared, vision, chatbot).

**University pipeline tested:** admission/fee intents classify correctly, spam + off-topic
queries are blocked.

**NEXUS pipeline tested:** wellbeing tiers (THRIVING→CRISIS) map correctly, support
categories detected, escalations logged, and the full intelligence report generates
(example output: trajectory IMPROVING, lowest tier DISTRESSED, 2 escalations,
risk score 41 → FOLLOW_UP).

**Vision config maps verified:** `(2,False)→Peace`, `(1,True)→Thumbs Up`, `(4,False)→letter B`.

Run the offline replay yourself to reproduce:
```powershell
conda activate hci_env
python run_chatbot.py    # → 2 (NEXUS) → 5 (offline replay)
```

---

## 8. Troubleshooting

**PyAudio fails to install**
```powershell
conda install -n hci_env -c conda-forge pyaudio
# or
pip install pipwin && pipwin install pyaudio
```

**Ollama not running / chatbot uses rule-based replies**
The app falls back automatically — this is by design, not an error. For AI replies:
1. Install Ollama, 2. run `ollama pull llama3`, 3. run `ollama serve` before the chatbot.
Verify: `curl http://localhost:11434/api/tags`

**Emotion detection fails: "DeepFace analysis failed: ... requires tf-keras package"**
On TensorFlow ≥ 2.16 DeepFace needs the `tf-keras` compatibility package. Install it:
```
pip install tf-keras
```
It is already listed in `environment.yml`, so a fresh `conda env create` includes it.

**DeepFace slow on first run**
It downloads the facial-expression model (~5 MB) the first time face detection runs.
`python setup_env.py` pre-caches it (step 5/5). Normal; later runs are fast.

**Webcam not found**
Close other apps using the camera, or set `WEBCAM_INDEX = 1` in `vision/vision_config.py`.
Test: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

**Microphone not detected (voice input)**
List devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`
Set your mic as the default Windows recording device.

**`conda activate` does nothing / not recognized**
Run `conda init powershell`, close and reopen the terminal, then `conda activate hci_env`.
