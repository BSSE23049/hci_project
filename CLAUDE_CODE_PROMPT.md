# HCI Project — Master Claude Code Prompt
# Paste everything below this line into Claude Code

---

## PROJECT OVERVIEW

Build a complete, modular, production-quality HCI (Human-Computer Interaction) project for a university course. The project has TWO independent applications that share some utilities:

1. **Vision App** — Real-time computer vision system (Assignment 4)
2. **Chatbot App** — Multi-modal university chatbot + NEXUS wellbeing advisor (Assignment 5 + Text/Voice Assessment)

Both apps run on **Windows**, inside a **conda environment** named `hci_env` using **Python 3.10**.

---

## ABSOLUTE RULES (never break these)

- **Every feature must be toggled by a True/False flag in its config file.** If the flag is False, that feature is completely skipped — no errors, no crashes.
- **Every tunable value (thresholds, model names, keywords, gesture maps, response rules) lives ONLY in the config file.** No magic numbers or hardcoded strings anywhere else in the code.
- **Changing a scenario means editing only the config file.** Example: to remap "2 fingers = Peace" to something else, you change one line in `vision_config.py`. To change chatbot rules, you edit one list in `chatbot_config.py`. Nothing else should need touching.
- **All functions must have docstrings** describing parameters and return values.
- **No global mutable state.** Pass data as function arguments or return values.
- **Each module lives in its own file.** The main runner files only import and orchestrate.
- **Code must run without modification** on a standard Windows laptop with a working webcam and microphone.
- **Graceful degradation:** if an optional library (DeepFace, Ollama, sounddevice) is unavailable, print a clear warning and continue with a fallback — never crash.

---

## DIRECTORY STRUCTURE TO CREATE

```
hci_project/
│
├── environment.yml                  # conda env definition
├── setup_env.py                     # post-install script (downloads models/corpora)
├── README.md                        # setup and run instructions
│
├── shared/
│   ├── __init__.py
│   ├── audio_utils.py               # capture_audio() + transcribe() — used by both apps
│   └── llm_utils.py                 # query_ollama() wrapper
│
├── vision/
│   ├── __init__.py
│   ├── vision_config.py             # ALL vision flags and constants
│   ├── lips_module.py               # Module 1: smile, MAR, lip-sync counter
│   ├── eyes_module.py               # Module 2: EAR, blink counter, drowsiness
│   ├── face_module.py               # Module 3: emotion, head pose, emotion history
│   ├── hand_module.py               # Module 4: finger count, gestures, letters, game
│   └── vision_main.py               # entry point for the vision app
│
├── chatbot/
│   ├── __init__.py
│   ├── chatbot_config.py            # ALL chatbot + NEXUS flags and constants
│   ├── spam_module.py               # spam detection (rule-based + domain filter)
│   ├── intent_module.py             # intent classification with confidence
│   ├── response_module.py           # response generation (static + LLM)
│   ├── nexus_wellbeing.py           # NEXUS 6-tier wellbeing engine
│   ├── nexus_support.py             # NEXUS support classifier + transition logger
│   ├── nexus_report.py              # NEXUS intelligence report generator
│   └── chatbot_main.py              # entry point for the chatbot app
│
└── run_vision.py                    # top-level launcher: python run_vision.py
└── run_chatbot.py                   # top-level launcher: python run_chatbot.py
```

---

## FILE 1 — `environment.yml`

Create a conda environment file that:

- Sets `name: hci_env`
- Uses channels: `conda-forge`, `defaults`
- Installs **Python 3.10** via conda
- Installs these via conda (they need system-level binaries):
  - `numpy`
  - `ffmpeg` (system binary, not pip)
  - `portaudio` (required by PyAudio)
  - `libsndfile` (required by sounddevice/soundfile)
- Installs ALL of the following via the `pip:` section inside the conda env (so they land inside the env, not globally):
  - `opencv-python`
  - `mediapipe`
  - `deepface`
  - `sounddevice`
  - `soundfile`
  - `pyaudio`
  - `SpeechRecognition`
  - `pydub`
  - `openai-whisper`
  - `nltk`
  - `textblob`
  - `scikit-learn`
  - `ollama` (Python client for local Ollama server)
  - `anthropic` (kept for future use, not actively used)
  - `python-dotenv`
  - `requests`
  - `Pillow`
  - `gradio` (kept for compatibility)
- Add a comment block at the bottom listing the 4 manual post-install steps:
  1. `python setup_env.py` to download NLTK corpora, TextBlob corpora, Whisper base model
  2. `ollama pull llama3` (done at home with internet, ~4.7 GB)
  3. Verify: `python -c "import cv2, mediapipe, whisper, nltk; print('OK')"`
  4. Start Ollama before running chatbot: `ollama serve`

---

## FILE 2 — `setup_env.py`

A script the user runs ONCE after activating the env. It must:
- Download NLTK corpora: `vader_lexicon`, `punkt`, `stopwords`, `averaged_perceptron_tagger`
- Download TextBlob corpora
- Pre-cache the Whisper `base` model
- Print clear success/failure messages for each step
- NOT download the Ollama llama3 model (too large, user does this manually)

---

## FILE 3 — `README.md`

Write clear setup and usage instructions:
- Step 1: Clone/download the project
- Step 2: `conda env create -f environment.yml`
- Step 3: `conda activate hci_env`
- Step 4: `python setup_env.py`
- Step 5: `ollama pull llama3` (internet required, do this at home)
- Step 6: How to run the vision app: `python run_vision.py`
- Step 7: How to run the chatbot app: `python run_chatbot.py`
- Step 8: How to switch chatbot mode (university vs NEXUS): edit `CHATBOT_MODE` in `chatbot/chatbot_config.py`
- Step 9: How to enable/disable any feature: edit the True/False flags in the relevant config file
- Include a troubleshooting section for: PyAudio install failure on Windows, Ollama not running, DeepFace slow first run, webcam not found

---

## FILE 4 — `vision/vision_config.py`

This file contains ALL flags and constants for the vision app. No other vision file should contain hardcoded values. Include:

### Feature enable/disable flags
```python
ENABLE_LIPS_DETECTION  = True
ENABLE_EYES_DETECTION  = True
ENABLE_FACE_DETECTION  = True
ENABLE_HAND_DETECTION  = True
```

### Input source
```python
DEFAULT_INPUT_SOURCE = "webcam"   # "webcam" | "video" | "image"
WEBCAM_INDEX = 0
```

### Lips constants
```python
MAR_OPEN_THRESHOLD  = 0.55    # mouth open / yawning
MAR_CLOSE_THRESHOLD = 0.35    # hysteresis close threshold
SMILE_CORNER_THRESHOLD = 0.02
```

### Eyes constants
```python
EAR_CLOSED_THRESHOLD  = 0.25
DROWSY_FRAMES_TRIGGER = 20    # consecutive closed frames to fire alert
DROWSY_CLEAR_FRAMES   = 5     # consecutive open frames to clear alert
```

### Face constants
```python
EMOTION_INFERENCE_EVERY  = 5    # run DeepFace every N frames
EMOTION_HISTORY_SECONDS  = 5    # rolling window for recent-mood
```

### Hand gesture map
```python
# Key: (non_thumb_finger_count: int, thumb_extended: bool)
# Value: (label: str, emoji: str)
# TO CHANGE A GESTURE: edit only this dict. Nothing else changes.
GESTURE_MAP = {
    (0, False): ("Fist",      "✊"),
    (1, False): ("One",       "☝️"),
    (2, False): ("Peace",     "✌️"),
    (3, False): ("Three",     "🤟"),
    (4, False): ("Four",      "🖖"),
    (5, False): ("Open Hand", "🖐"),
    (1, True):  ("Thumbs Up", "👍"),
    (5, True):  ("High Five", "🙌"),
}
```

### Letter / hand-sign detection map
```python
# TO ADD LETTERS: add entries here only. Set ENABLE_LETTER_DETECTION = True.
# Key: letter string. Value: (non_thumb_count, thumb_extended) — same key format as GESTURE_MAP
ENABLE_LETTER_DETECTION = True
LETTER_GESTURE_MAP = {
    "A": (0, True),
    "B": (4, False),
    "I": (1, False),
    "L": (1, True),
    "U": (2, False),
    "W": (3, False),
    "Y": (5, True),
    # user can add more here
}
```

### Game
```python
GAME_HOLD_SECONDS = 1.0
MAX_HANDS = 2
```

---

## FILE 5 — `vision/lips_module.py`

Implement using MediaPipe FaceMesh (468 landmarks). All threshold values must be imported from `vision_config.py`.

### Functions to implement:

**`compute_mar(landmarks, frame_width, frame_height) -> float`**
- Compute Mouth Aspect Ratio: vertical mouth opening divided by horizontal mouth width
- Use landmarks: top-lip centre (13), bottom-lip centre (14), left corner (61), right corner (291)
- Return float

**`detect_smile(landmarks, frame_width, frame_height) -> bool`**
- Compare average y-position of lip corners vs mouth centre
- If corners are higher (smaller y) than centre by more than `SMILE_CORNER_THRESHOLD * frame_height` → smiling
- Return bool

**`class LipSyncCounter`**
- Tracks open→close mouth cycles using hysteresis (`MAR_OPEN_THRESHOLD`, `MAR_CLOSE_THRESHOLD`)
- Method: `update(mar: float) -> int` — increments and returns count when a full cycle completes
- State: `count`, `_is_open`

**`draw_lips_overlay(frame, landmarks, w, h, mar, smiling, sync_count) -> None`**
- Draw: smile/neutral label, MAR value (2 decimal places), lip-sync count
- If `mar > MAR_OPEN_THRESHOLD`: draw a thick RED rectangle border around entire frame (yawn/open mouth alert)

---

## FILE 6 — `vision/eyes_module.py`

Implement using MediaPipe FaceMesh. Import thresholds from `vision_config.py`.

Left eye landmark indices: `[362, 385, 387, 263, 373, 380]`
Right eye landmark indices: `[33, 160, 158, 133, 153, 144]`

**`compute_ear(landmarks, eye_indices, frame_w, frame_h) -> float`**
- EAR formula: `(||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)` where p1..p6 are the 6 eye landmark points
- Return float

**`compute_both_ears(landmarks, frame_w, frame_h) -> tuple[float, float]`**
- Return `(left_ear, right_ear)` using the two index lists above

**`class BlinkTracker`**
- Detects complete open→close→open blink cycles
- Method: `update(avg_ear: float) -> int`
- Increments count when: was closed (avg_ear < threshold) and now open (avg_ear >= threshold)

**`class DrowsinessDetector`**
- Fires alert after `DROWSY_FRAMES_TRIGGER` consecutive frames with avg_ear < threshold
- Clears after `DROWSY_CLEAR_FRAMES` consecutive open frames
- Method: `update(avg_ear: float) -> bool` — returns True if drowsy

**`draw_eyes_overlay(frame, left_ear, right_ear, blink_count, is_drowsy) -> None`**
- Display: blink count, left EAR (3 decimal places), right EAR (3 decimal places)
- If `is_drowsy`: overlay a large red filled rectangle in the centre of the frame with white text `"DROWSY! Wake Up!"`

---

## FILE 7 — `vision/face_module.py`

Combine Haar cascade (bounding box), MediaPipe FaceMesh (head pose), DeepFace (emotion). Import constants from `vision_config.py`.

**`detect_emotion(frame, bbox: tuple) -> tuple[str, float]`**
- Crop face ROI using bbox `(x, y, w, h)`
- Run `DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False, silent=True)`
- Run only every `EMOTION_INFERENCE_EVERY` frames (caller passes frame count, function decides whether to run)
- Return `(dominant_emotion: str, confidence: float)`
- If DeepFace not installed or fails: return `("unknown", 0.0)` without crashing

**`estimate_head_pose(landmarks, frame_w, frame_h) -> str`**
- Use nose tip (landmark 1) vs eye midpoint (average of landmarks 33 and 263) geometry
- Calculate dx (horizontal offset) and dy (vertical offset) in pixels
- Rules: if both small → "Forward"; if |dx| > |dy| → "Left" or "Right"; else → "Up" or "Down"
- Return one of: `"Forward"` | `"Left"` | `"Right"` | `"Up"` | `"Down"`

**`class EmotionHistoryTracker`**
- Maintains a rolling deque of `(timestamp, emotion)` tuples
- Method: `add(emotion: str) -> None`
- Method: `recent_mood() -> str` — returns most frequent emotion in last `EMOTION_HISTORY_SECONDS`, or "N/A"
- Automatically prunes old entries

**`draw_face_overlay(frame, bbox, emotion, confidence, pose, recent_mood) -> None`**
- Draw: blue rectangle bounding box, emotion + confidence above the box
- Top-left text: head pose direction, recent mood label

---

## FILE 8 — `vision/hand_module.py`

Use MediaPipe Hands (up to `MAX_HANDS` hands, 21 landmarks each). Import all maps from `vision_config.py`.

Tip landmark IDs: `[4, 8, 12, 16, 20]` (thumb, index, middle, ring, pinky)
PIP landmark IDs: `[3, 6, 10, 14, 18]`

**`count_fingers(hand_landmarks, handedness: str) -> tuple[int, bool]`**
- For fingers 1–4: extended if `tip.y < pip.y`
- For thumb: 
  - If `handedness == "Right"`: extended if `tip.x < pip.x`
  - If `handedness == "Left"`:  extended if `tip.x > pip.x`
- Return `(non_thumb_count: int 0-4, thumb_extended: bool)`

**`classify_gesture(finger_count: int, thumb_extended: bool) -> tuple[str, str]`**
- Look up `(finger_count, thumb_extended)` in `GESTURE_MAP` from config
- Return `(label, emoji)` or `("Unknown", "❓")` if not found
- **This function must ONLY read from GESTURE_MAP — no hardcoded gesture names**

**`detect_letter(finger_count: int, thumb_extended: bool) -> str | None`**
- If `ENABLE_LETTER_DETECTION` is False: return None immediately
- Look up `(finger_count, thumb_extended)` in `LETTER_GESTURE_MAP` from config
- Return the matched letter string or None

**`class GestureGame`**
- Picks a random target gesture from `GESTURE_MAP` keys
- Method: `update(finger_count, thumb_extended) -> bool` — returns True if point just scored
- Method: `hold_progress() -> float` — returns 0.0–1.0 for the progress bar
- Property: `target_label` — returns `"{emoji} {label}"` for the current target
- When a point is scored: increment `self.score` and pick a new random target

**`draw_hand_overlay(frame, hand_results, game=None) -> None`**
- For each detected hand: draw finger count, gesture label+emoji, letter (if detected)
- Position text near the wrist landmark
- If `game` is not None: draw game HUD at top of frame showing target gesture, score, and progress bar

---

## FILE 9 — `vision/vision_main.py`

The vision app entry point. Console menu only (no GUI window for the menu — only the OpenCV frame window for detection).

### Menu flow:

```
=== HCI Vision System ===
Select Input Source:
  1. Live Webcam
  2. Video File
  3. Image File

Select Detection Mode:
  1. Lips Detection
  2. Eyes Detection
  3. Face Detection
  4. Hand Detection  (sub-option: enable Gesture Game? y/n)

Press ESC at any time to return to this menu.
```

### Architecture:

- Use a single `while True` main loop for the menu
- Each detection mode runs its own `run_lips(source)`, `run_eyes(source)`, `run_face(source)`, `run_hands(source, game_mode)` function
- Each run function:
  1. Opens the video source (webcam index, video path, or image path)
  2. Initialises the required MediaPipe model(s)
  3. Initialises stateful objects (BlinkTracker, LipSyncCounter, etc.)
  4. Enters a per-frame loop: read frame → process → draw overlay → `cv2.imshow`
  5. Breaks on ESC key (`cv2.waitKey(1) & 0xFF == 27`)
  6. Releases camera and destroys windows before returning
- Check the corresponding `ENABLE_*` flag before running each mode — if False, print a message and return
- For Face Detection: initialise both `cv2.CascadeClassifier` (for bbox) and MediaPipe FaceMesh (for pose), and `EmotionHistoryTracker`
- For image mode: display the processed image, wait for any key, then return to menu

---

## FILE 10 — `shared/audio_utils.py`

Shared by both apps.

**`capture_audio(seconds: int, sample_rate: int) -> np.ndarray`**
- Record from default microphone using `sounddevice`
- Print a countdown: `"🎙️ Recording... 7s 6s 5s..."` updating every second
- Return float32 numpy array, shape `(seconds * sample_rate,)`
- If sounddevice not available: print error and raise ImportError

**`transcribe(audio_array: np.ndarray, sample_rate: int, model_name: str = "base") -> dict`**
- Load Whisper model by `model_name`
- Call `model.transcribe(audio_array, fp16=False)`
- Return dict: `{"text": str, "language": str, "confidence": "high" | "low"}`
- Confidence is `"high"` if `len(text) > 10`, else `"low"`

---

## FILE 11 — `shared/llm_utils.py`

**`query_ollama(prompt, system_prompt, model, base_url, temperature=0.7) -> str | None`**
- POST to `{base_url}/api/chat` with the messages array: system + user
- Parse `response.json()["message"]["content"]`
- On `ConnectionError`: print `"[WARN] Ollama not running. Using rule-based fallback."` and return `None`
- On any other exception: print warning and return `None`
- **The `system_prompt` parameter is where the scenario/rules live. Changing the scenario = changing what is passed as `system_prompt`. The function itself never changes.**

---

## FILE 12 — `chatbot/chatbot_config.py`

This file contains ALL flags and constants for both the university chatbot and NEXUS. No other chatbot file should contain hardcoded values.

### Top-level mode switch
```python
CHATBOT_MODE = "university"   # "university" | "nexus"
DEFAULT_INPUT_MODE = "hybrid" # "text" | "voice" | "hybrid"
```

### LLM flags
```python
USE_LLM      = True           # False = pure rule-based only
LLM_PROVIDER = "ollama"
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"
```

### Whisper flags
```python
WHISPER_MODEL        = "base"
AUDIO_RECORD_SECONDS = 7
AUDIO_SAMPLE_RATE    = 16000
```

### University chatbot — intent dictionary
```python
# TO ADD A NEW INTENT: add one entry to this dict. Nothing else changes.
UNIVERSITY_INTENTS = {
    "Admission": {
        "keywords": ["admission", "apply", "application", "enroll", ...],
        "response_static": "...",
    },
    "Fee": {
        "keywords": ["fee", "fees", "charges", "tuition", ...],
        "response_static": "...",
    },
    "Courses": {
        "keywords": [...],
        "response_static": "...",
    },
    "Schedule": {
        "keywords": [...],
        "response_static": "...",
    },
    # Commented-out examples for future intents (Library, Hostel, Transport)
}
```

### Spam config
```python
SPAM_KEYWORDS    = [...]   # list of spam trigger phrases
DOMAIN_KEYWORDS  = [...]   # university-relevant words; query must match at least one
SPAM_BLOCKED_RESPONSE = "This query is classified as spam."
UNKNOWN_INTENT_RESPONSE = "Sorry, I could not understand your request. ..."
```

### NEXUS wellbeing scale
```python
# Order matters — checked top to bottom
NEXUS_WELLBEING_SCALE = [
    ("THRIVING",     0.60,  float("inf"), "🌟"),
    ("CONTENT",      0.20,  0.60,         "😊"),
    ("NEUTRAL",     -0.19,  0.20,         "😐"),
    ("STRESSED",    -0.40, -0.19,         "😟"),
    ("DISTRESSED",  -0.60, -0.40,         "😢"),
    ("CRISIS",  float("-inf"), -0.60,     "🆘"),
]
```

### NEXUS support keyword dict
```python
# TO ADD A CATEGORY: add one entry. The classifier rebuilds automatically.
NEXUS_SUPPORT_KEYWORDS = {
    "ACADEMIC":  [...],
    "WELLBEING": [...],
    "FINANCIAL": [...],
    "TECHNICAL": [...],
    "SOCIAL":    [...],
    "ADMIN":     [...],
}
```

### NEXUS response rules list
```python
# Rules are checked top to bottom. Use "*" as wildcard.
# TO CHANGE A RESPONSE: edit the string in this list only.
# TO ADD A NEW RULE: add a tuple. Nothing else changes.
NEXUS_RESPONSE_RULES = [
    ("WELLBEING", "CRISIS",
     "I am very concerned about you. Please contact the university counselling line RIGHT NOW: 0800-XXX-XXXX."),
    ("WELLBEING", "DISTRESSED",
     "It sounds like you are going through a really difficult time. Have you spoken to anyone about how you are feeling?"),
    ("ACADEMIC",  "STRESSED",
     "Exam pressure is real. Let us look at what support your faculty offers — have you spoken to your tutor?"),
    ("FINANCIAL", "*",
     "Financial difficulty is more common than you think. The university bursary office can help — shall I give you their contact?"),
    ("TECHNICAL", "*",
     "Let me help you with that technical issue. Which system are you trying to access?"),
    ("SOCIAL",    "DISTRESSED",
     "Feeling isolated at university is incredibly hard. The student union runs weekly social events — would that help?"),
    ("*",         "*",
     "Thank you for sharing that with me. I am here to help — can you tell me a little more about what you need?"),
]
```

### Risk score constants
```python
NEXUS_RISK_URGENT    = 70
NEXUS_RISK_FOLLOW_UP = 40
NEXUS_CRISIS_LINE    = "0800-XXX-XXXX"
```

---

## FILE 13 — `chatbot/spam_module.py`

Import all keyword lists from `chatbot_config.py`.

**`is_spam(text: str) -> tuple[bool, str]`**
- Step 1 (basic spam): check if any word/phrase from `SPAM_KEYWORDS` appears in lowercase text
- Step 2 (domain filter): if not caught by step 1, check if text contains at least one word from `DOMAIN_KEYWORDS`. If not → also spam
- Return `(True, reason_string)` if spam, `(False, "")` if clean
- reason_string should say WHY it was flagged (e.g. "matched spam keyword: 'buy now'" or "off-topic: no university context detected")

---

## FILE 14 — `chatbot/intent_module.py`

Import `UNIVERSITY_INTENTS` and `UNKNOWN_INTENT_RESPONSE` from `chatbot_config.py`.

**`classify_intent(text: str) -> dict`**
- Lowercase the text
- For each intent in `UNIVERSITY_INTENTS`: count how many of its `keywords` appear in text
- Return dict:
  ```python
  {
      "intent":     str,          # top-scoring intent name or "Unknown"
      "confidence": int,          # raw keyword match count (score)
      "pattern":    str,          # comma-joined list of matched keywords
      "all_scores": dict          # {intent_name: score} for all intents
  }
  ```
- If all scores are 0: intent = "Unknown"
- Handle variations by checking both the keyword and simple stemmed forms (e.g. "fees" matches "fee" entry) — do this by checking if the keyword is a substring of any word in the text

---

## FILE 15 — `chatbot/response_module.py`

Import `UNIVERSITY_INTENTS`, `UNKNOWN_INTENT_RESPONSE`, `USE_LLM`, `OLLAMA_MODEL`, `OLLAMA_BASE_URL` from `chatbot_config.py`. Import `query_ollama` from `shared/llm_utils.py`.

**`generate_university_response(intent: str, user_text: str) -> str`**
- If intent is "Unknown": return `UNKNOWN_INTENT_RESPONSE`
- If `USE_LLM` is False: return `UNIVERSITY_INTENTS[intent]["response_static"]`
- If `USE_LLM` is True:
  - Build a `system_prompt` that describes the university chatbot role and the specific intent rules — this is where the "scenario" lives
  - Call `query_ollama(user_text, system_prompt, OLLAMA_MODEL, OLLAMA_BASE_URL)`
  - If Ollama returns None: fall back to `response_static`
  - Return the LLM response or fallback

**`speak_response(text: str) -> None`**
- Use `pyttsx3` to speak the text
- If pyttsx3 not available: print `"[TTS unavailable] {text}"` and continue

---

## FILE 16 — `chatbot/nexus_wellbeing.py`

Import `NEXUS_WELLBEING_SCALE`, `NEXUS_CRISIS_LINE` from `chatbot_config.py`.

**`assess_wellbeing(text: str) -> dict`**
- Use `nltk.sentiment.SentimentIntensityAnalyzer` (VADER) to get compound score
- Walk `NEXUS_WELLBEING_SCALE` top to bottom; match the first tier where `low <= score < high`
- Return: `{"tier": str, "score": float, "emoji": str, "is_at_risk": bool}`
- `is_at_risk = True` only for "CRISIS" tier

**`compute_trajectory(wellbeing_log: list[dict]) -> dict`**
- Split log into first half and second half by index
- Compute mean `score` for each half
- If `second_avg > first_avg + 0.1` → trend = "improving"
- If `second_avg < first_avg - 0.1` → trend = "declining"
- Else → "fluctuating"
- Return: `{"trend": str, "lowest_tier": str, "at_risk_turns": list[int]}`
- `at_risk_turns`: 0-based turn indices where `is_at_risk == True`
- `lowest_tier`: tier with the lowest compound score across all turns

**`check_and_alert(wellbeing_result: dict, turn_number: int) -> bool`**
- If `wellbeing_result["is_at_risk"]` is True: print a prominent alert block (use `=` borders) recommending the student contact the counselling line, include the turn number
- Return the boolean flag

---

## FILE 17 — `chatbot/nexus_support.py`

Import `NEXUS_SUPPORT_KEYWORDS`, `NEXUS_RESPONSE_RULES` from `chatbot_config.py`.

**`classify_support_need(text: str) -> dict`**
- Lowercase text, score ALL categories by keyword count
- Return:
  ```python
  {
      "primary":      str,         # highest-scoring category
      "all_detected": list[str],   # all categories with score > 0
      "scores":       dict         # {category: score}
  }
  ```
- If all scores are 0: primary = "GENERAL"

**`log_support_transition(support_log: list, new_primary: str, turn_number: int) -> list`**
- If support_log is empty or new_primary == last entry's "curr": return log unchanged
- Otherwise append: `{"prev": str, "curr": str, "turn": int, "is_escalation": bool}`
- `is_escalation = True` if new_primary == "WELLBEING" and prev != "WELLBEING"
- Return updated log

**`nexus_respond(text: str, support_need: str, wellbeing_tier: str) -> str`**
- Walk `NEXUS_RESPONSE_RULES` from config top to bottom
- For each rule `(cat, tier, response)`:
  - Match if `(cat == support_need or cat == "*") and (tier == wellbeing_tier or tier == "*")`
  - Return the first match's response string
- **This function contains NO hardcoded responses — all responses come from the config**

---

## FILE 18 — `chatbot/nexus_report.py`

Import risk constants from `chatbot_config.py`.

**`generate_intelligence_report(session_log, wellbeing_log, support_log, transition_log) -> None`**

Print a formatted report to console. Compute and display every field below:

```
===========================================================
        NEXUS STUDENT INTELLIGENCE REPORT
        Code: NX-2B  —  Counsellor Eyes Only
===========================================================
Session Turns     : {total}  (Voice: {voice_count} | Text: {text_count})
-----------------------------------------------------------
Wellbeing Trajectory  : IMPROVING / DECLINING / FLUCTUATING
Lowest Tier Reached   : {tier with emoji}
At-Risk Alerts        : {count} (turns: {list of turn numbers})
-----------------------------------------------------------
Support Categories (by frequency):
  1. ACADEMIC    — 4 turns
  2. WELLBEING   — 3 turns
  ...
Escalation Events     : {count} transitions into WELLBEING
  (e.g. Turn 4: ACADEMIC → WELLBEING)
-----------------------------------------------------------
Risk Score            : {0-100}
Recommended Action    : NO_ACTION / FOLLOW_UP / URGENT_REFERRAL
===========================================================
```

### Risk score formula (implement exactly):
```python
avg_wellbeing_score = mean of all score values in wellbeing_log
at_risk_count       = count of is_at_risk == True in wellbeing_log
escalation_count    = count of is_escalation == True in transition_log
avg_words_per_turn  = mean word count of all session_log entries

base_risk           = 20
wellbeing_penalty   = abs(avg_wellbeing_score) * 40
at_risk_penalty     = at_risk_count * 15
escalation_penalty  = escalation_count * 10
word_count_factor   = 5 if avg_words_per_turn > 20 else 0

risk_score = base_risk + wellbeing_penalty + at_risk_penalty + escalation_penalty + word_count_factor
risk_score = max(0, min(100, round(risk_score)))

# Action thresholds from config:
# >= NEXUS_RISK_URGENT    → "URGENT_REFERRAL"
# >= NEXUS_RISK_FOLLOW_UP → "FOLLOW_UP"
# else                    → "NO_ACTION"
```

---

## FILE 19 — `chatbot/chatbot_main.py`

The chatbot app entry point. Console-based.

### Menu flow:

```
=== HCI Chatbot System ===
Mode: UNIVERSITY CHATBOT  (or NEXUS WELLBEING ADVISOR)
Input: HYBRID

Options:
  1. Start conversation
  2. Switch mode (university / nexus)
  3. Switch input (text / voice / hybrid)
  4. Exit
```

### Architecture:

**University chatbot conversation loop:**
1. Get input (text, voice, or hybrid based on config)
   - Text: `input("You: ")`
   - Voice: call `capture_audio()` → `transcribe()` from `shared/audio_utils.py`, print recognised text
   - Hybrid: ask user each turn "Press ENTER to type, or V+ENTER for voice:"
2. Check spam → if spam: print blocked message, continue loop
3. Classify intent → print detected intent + confidence + matched keywords
4. Generate response → print response
5. Speak response (if TTS enabled)
6. Repeat until user types "exit" or "quit"

**NEXUS conversation loop:**
Maintain these lists across turns: `session_log`, `wellbeing_log`, `support_log`, `transition_log`
1. Get input (same as above)
2. `assess_wellbeing()` → print emoji + tier
3. `check_and_alert()` → print crisis alert if needed
4. `classify_support_need()` → print primary + all detected categories
5. `log_support_transition()` → update transition log
6. `nexus_respond()` → print NEXUS response
7. Append to all logs
8. After loop ends (user types "exit" or "quit"): call `generate_intelligence_report()`

**Session log entry format:**
```python
{"text": str, "source": "text"|"voice", "turn": int, "word_count": int}
```

---

## FILE 20 — `run_vision.py`

```python
# Top-level launcher for the vision app
# Usage: python run_vision.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from vision.vision_main import main
if __name__ == "__main__":
    main()
```

---

## FILE 21 — `run_chatbot.py`

```python
# Top-level launcher for the chatbot app
# Usage: python run_chatbot.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from chatbot.chatbot_main import main
if __name__ == "__main__":
    main()
```

---

## ADDITIONAL IMPLEMENTATION REQUIREMENTS

### Shared `__init__.py` files
Create empty `__init__.py` in: `shared/`, `vision/`, `chatbot/`

### Error handling throughout
- Wrap every camera open with a check: if `cap.isOpened()` is False, print an error and return to menu
- Wrap every `DeepFace.analyze()` call in try/except — DeepFace can fail on first run due to model download
- Wrap every Whisper transcription in try/except
- Wrap every Ollama call in try/except (already handled in `llm_utils.py`)

### NLTK download guard
In `nexus_wellbeing.py`, before importing VADER, add:
```python
import nltk
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
```

### MediaPipe initialisation pattern
All vision modules that use MediaPipe must initialise inside the `run_*` function (not at module level) to avoid importing MediaPipe before it is needed.

### Frame processing pattern for all vision run functions
```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)   # mirror for natural interaction
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mesh.process(rgb)   # or hands.process(rgb)
    # ... draw overlays ...
    cv2.imshow("NEXUS Vision", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # ESC
        break
cap.release()
cv2.destroyAllWindows()
```

### Offline session replay (Stage 4 of Text+Voice Assessment)
In `chatbot_main.py`, add an option in the NEXUS menu:
```
  5. Run offline session replay (test with hardcoded STUDENT_LOG)
```
This processes the following 10-message log through the full pipeline without any mic/keyboard input, prints all turn results, then prints the intelligence report:
```python
STUDENT_LOG = [
    "Hi, I need some help please.",
    "I have a major assignment due tomorrow and I have not started.",
    "My laptop also broke yesterday so I cannot access my files.",
    "To be honest I have been struggling a lot lately, not just academically.",
    "I have not been sleeping, I feel completely hopeless about everything.",
    "I think I might need to talk to someone but I do not know who.",
    "Also I got an email saying my fees are overdue and I cannot register.",
    "Sorry for dumping all this. I just feel very alone right now.",
    "Actually, my friend just texted. I feel a tiny bit better now.",
    "Thank you for listening. I will try to contact the counsellor.",
]
```
For each message print: `Turn {i+1}: {emoji} {tier} | {primary_category} | all: {all_detected}`
Then print NEXUS response. Then print the full intelligence report at the end.

---

## WHAT NOT TO DO

- Do NOT use `global` mutable variables anywhere
- Do NOT hardcode any threshold, keyword, gesture name, or response string outside a config file
- Do NOT import a module at the top level if it might not be installed — use local imports inside functions with try/except
- Do NOT use `sys.exit()` on missing optional packages — degrade gracefully
- Do NOT implement a Gradio UI — terminal/console only for menus, OpenCV window for video
- Do NOT mix vision and chatbot logic in the same file
- Do NOT implement `run_vision.py` and `run_chatbot.py` with any logic — they are pure launchers

---

## VERIFICATION CHECKLIST (Claude Code must confirm each item is done)

After implementing everything, verify:

- [ ] `environment.yml` creates the env cleanly with `conda env create -f environment.yml`
- [ ] All 21 files exist in the correct locations
- [ ] Every vision feature can be disabled by setting its flag to False in `vision_config.py`
- [ ] Changing a gesture in `GESTURE_MAP` in config changes it everywhere with no other edits
- [ ] Adding a letter to `LETTER_GESTURE_MAP` in config makes it appear in detection with no other edits
- [ ] Changing `USE_LLM = False` makes the chatbot fully rule-based
- [ ] Changing `CHATBOT_MODE` from `"university"` to `"nexus"` switches the entire chatbot flow
- [ ] Changing an intent's `response_static` string in `UNIVERSITY_INTENTS` changes the response
- [ ] Adding a new intent dict entry to `UNIVERSITY_INTENTS` makes it classifiable immediately
- [ ] Editing a rule tuple in `NEXUS_RESPONSE_RULES` changes the NEXUS response
- [ ] The offline NEXUS session replay runs without any hardware (no webcam, no mic needed)
- [ ] All functions have docstrings
- [ ] No hardcoded magic numbers or strings outside config files
