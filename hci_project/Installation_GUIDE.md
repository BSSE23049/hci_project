# HCI Project — Function Reference

Two Python applications built for an HCI university course:

- **Chatbot App** (`run_chatbot.py`) — university information chatbot + NEXUS wellbeing advisor, with text/voice/hybrid input and optional LLM responses via Ollama.
- **Vision App** (`run_vision.py`) — real-time computer vision using MediaPipe (lips, eyes, face, hands) with an optional hybrid mode that runs all features simultaneously.

**Quick start:**
```bash
conda activate hci_env
python run_chatbot.py   # chatbot
python run_vision.py    # vision
```

---

## Table of Contents

### Chatbot App
1. [shared/audio\_utils.py](#1-sharedaudio_utilspy)
2. [shared/llm\_utils.py](#2-sharedllm_utilspy)
3. [chatbot/spam\_module.py](#3-chatbotspam_modulepy)
4. [chatbot/intent\_module.py](#4-chatbotintent_modulepy)
5. [chatbot/response\_module.py](#5-chatbotresponse_modulepy)
6. [chatbot/nexus\_wellbeing.py](#6-chatbotnexus_wellbeingpy)
7. [chatbot/nexus\_support.py](#7-chatbotnexus_supportpy)
8. [chatbot/nexus\_report.py](#8-chatbotnexus_reportpy)
9. [chatbot/chatbot\_main.py](#9-chatbotchatbot_mainpy)

### Vision App
10. [vision/vision\_utils.py](#10-visionvision_utilspy)
11. [vision/mp\_tasks.py](#11-visionmp_taskspy)
12. [vision/lips\_module.py](#12-visionlips_modulepy)
13. [vision/eyes\_module.py](#13-visioneyes_modulepy)
14. [vision/face\_module.py](#14-visionface_modulepy)
15. [vision/hand\_module.py](#15-visionhand_modulepy)
16. [vision/hybrid\_module.py](#16-visionhybrid_modulepy)
17. [vision/vision\_main.py](#17-visionvision_mainpy)

---

## Chatbot App

### 1. `shared/audio_utils.py`

Microphone capture and Whisper transcription shared between apps.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `capture_audio` | `seconds: int = 7`<br>`sample_rate: int = 16000` | `numpy.ndarray` (float32) | Records audio from the default microphone. Recording starts immediately; a countdown shows how long remains. Raises `ImportError` if sounddevice is missing, `RuntimeError` if the mic cannot be accessed. |
| `transcribe` | `audio_array: ndarray`<br>`sample_rate: int = 16000`<br>`model_name: str = "base"` | `dict` — `{"text": str, "language": str, "confidence": "high"\|"low"}` | Transcribes a float32 audio array with OpenAI Whisper. The Whisper model is loaded once and cached for all subsequent calls. Returns an empty-text dict on failure instead of raising. |

#### Example Usage

```python
import sys; sys.path.insert(0, "path/to/hci_project")
from shared.audio_utils import capture_audio, transcribe

# Record 5 seconds from the microphone
audio = capture_audio(seconds=5, sample_rate=16000)

# Transcribe with the 'small' Whisper model
result = transcribe(audio, sample_rate=16000, model_name="small")
print(result["text"])        # "What are the library hours?"
print(result["language"])    # "en"
print(result["confidence"])  # "high"
```

---

### 2. `shared/llm_utils.py`

Thin wrapper around a locally running Ollama server.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `query_ollama` | `prompt: str`<br>`system_prompt: str`<br>`model: str`<br>`base_url: str`<br>`temperature: float = 0.7` | `str \| None` | Sends a chat request to Ollama and returns the model reply. Returns `None` with a printed warning if Ollama is not running, `requests` is not installed, or the HTTP request fails. Never raises. |

#### Example Usage

```python
from shared.llm_utils import query_ollama

reply = query_ollama(
    prompt="What are the library hours?",
    system_prompt="You are a university info assistant. Keep replies under 2 sentences.",
    model="qwen2.5-coder:7b",
    base_url="http://localhost:11434",
)
if reply:
    print(reply)
else:
    print("Ollama unavailable — using static fallback.")
```

---

### 3. `chatbot/spam_module.py`

Two-stage spam and off-topic filter.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `is_spam` | `text: str` | `tuple[bool, str]` — `(flagged, reason)` | Step 1: checks for phrases from `SPAM_KEYWORDS`. Step 2: if no spam phrase found, verifies at least one word from `DOMAIN_KEYWORDS` appears — otherwise flags as off-topic. Returns `(False, "")` for clean input. |

#### Example Usage

```python
from chatbot.spam_module import is_spam

flagged, reason = is_spam("How much are the fees?")
print(flagged, reason)   # False  ""

flagged, reason = is_spam("Buy now — win a prize!")
print(flagged, reason)   # True   "matched spam keyword: 'buy now'"

flagged, reason = is_spam("hello there")
print(flagged, reason)   # True   "off-topic: no university context detected"
```

---

### 4. `chatbot/intent_module.py`

Classifies a student message into one of the university intents defined in `chatbot_config.py`. The strategy is controlled by `INTENT_CLASSIFICATION_MODE`.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `classify_intent` | `text: str` | `dict` — `{"intent": str, "confidence": int, "pattern": str, "all_scores": dict}` | Public entry point. Dispatches to rule-based or AI classifier based on `INTENT_CLASSIFICATION_MODE` in config. |
| `_classify_intent_rule_based` | `text: str` | same `dict` | Scores each intent by counting keyword matches in the lowercased text. Returns `intent="Unknown"` when all scores are zero. |
| `_classify_intent_ai` | `text: str` | same `dict` | Sends the message to Ollama with a constrained prompt listing valid intent names. Validates the returned label and falls back to rule-based if Ollama is down or returns an unrecognised name. |

#### Example Usage

```python
from chatbot.intent_module import classify_intent

result = classify_intent("What courses does the engineering faculty offer?")
print(result["intent"])      # "Courses"
print(result["confidence"])  # 2   (number of matched keywords)
print(result["pattern"])     # "course, courses"

result = classify_intent("I want to go to the moon")
print(result["intent"])      # "Unknown"
print(result["confidence"])  # 0
```

---

### 5. `chatbot/response_module.py`

Generates a text response and optionally speaks it aloud.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `generate_university_response` | `intent: str`<br>`user_text: str` | `str` | Returns the static `response_static` string from `UNIVERSITY_INTENTS` when `USE_LLM = False`. When `USE_LLM = True`, calls Ollama with a context-aware system prompt and falls back to the static string if Ollama is unavailable. Returns `UNKNOWN_INTENT_RESPONSE` when `intent == "Unknown"`. |
| `speak_response` | `text: str` | `None` | Speaks the response aloud using pyttsx3 via a fresh subprocess on every call — prevents the "run loop already started" error that occurs when `pyttsx3.init()` is called multiple times in the same process. Silently returns if pyttsx3 is not installed. |

#### Example Usage

```python
from chatbot.response_module import generate_university_response, speak_response

# USE_LLM = False in config → returns static string immediately
response = generate_university_response("Library", "When does the library close?")
print(response)
# "The university library is open Monday-Friday 08:00-22:00..."

# USE_LLM = True in config → calls Ollama, falls back to static if down
response = generate_university_response("Fee", "How much is the tuition?")
print(response)

# Speak it aloud
speak_response(response)

# Unknown intent
print(generate_university_response("Unknown", "What is the meaning of life?"))
# "Sorry, I could not understand your request..."
```

---

### 6. `chatbot/nexus_wellbeing.py`

VADER sentiment analysis mapped to the NEXUS wellbeing tier scale.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `assess_wellbeing` | `text: str` | `dict` — `{"tier": str, "score": float, "emoji": str, "is_at_risk": bool}` | Runs VADER on the text and maps the compound score to a tier (THRIVING, CONTENT, NEUTRAL, STRESSED, DISTRESSED, CRISIS). `is_at_risk` is `True` only for CRISIS. |
| `compute_trajectory` | `wellbeing_log: list[dict]` | `dict` — `{"trend": str, "lowest_tier": str, "at_risk_turns": list[int]}` | Splits the session into two halves, compares mean scores, and returns `"improving"`, `"declining"`, or `"fluctuating"`. Also returns the lowest tier reached and the turn indices where CRISIS occurred. |
| `check_and_alert` | `wellbeing_result: dict`<br>`turn_number: int` | `bool` | Prints a formatted CRISIS ALERT to the console if `is_at_risk` is `True`. Returns `True` if an alert was printed, `False` otherwise. |

#### Example Usage

```python
from chatbot.nexus_wellbeing import assess_wellbeing, compute_trajectory, check_and_alert

result = assess_wellbeing("I feel completely hopeless and cannot cope anymore.")
print(result["tier"])        # "CRISIS" or "DISTRESSED"
print(result["score"])       # e.g. -0.82
print(result["emoji"])       # "🆘"
print(result["is_at_risk"])  # True

# Print a crisis alert if needed
check_and_alert(result, turn_number=3)

# Trajectory across multiple turns
log = [
    assess_wellbeing("I feel terrible today"),
    assess_wellbeing("Things are a bit better"),
    assess_wellbeing("I am doing well now"),
]
traj = compute_trajectory(log)
print(traj["trend"])        # "improving"
print(traj["lowest_tier"])  # "DISTRESSED"
print(traj["at_risk_turns"]) # []
```

---

### 7. `chatbot/nexus_support.py`

Multi-label support category classifier and topic-transition logger.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `classify_support_need` | `text: str` | `dict` — `{"primary": str, "all_detected": list[str], "scores": dict}` | Counts keyword matches for each category (ACADEMIC, WELLBEING, FINANCIAL, TECHNICAL, SOCIAL, ADMIN). Returns the highest-scoring category as `primary`, all categories with at least one match, and the full score dict. Returns `"GENERAL"` as primary if nothing matches. |
| `log_support_transition` | `support_log: list[dict]`<br>`new_primary: str`<br>`turn_number: int` | `list[dict]` | Appends a transition event when the primary category changes turn-to-turn. Marks `is_escalation=True` when the new category is `WELLBEING`. Returns the log unchanged if the category did not change. |
| `nexus_respond` | `text: str`<br>`support_need: str`<br>`wellbeing_tier: str` | `str` | Walks `NEXUS_RESPONSE_RULES` top-to-bottom and returns the first rule whose category and tier match (`"*"` is a wildcard for either). All response strings live in `chatbot_config.py`. |

#### Example Usage

```python
from chatbot.nexus_support import classify_support_need, log_support_transition, nexus_respond

result = classify_support_need("I am really struggling with my assignment deadline.")
print(result["primary"])       # "ACADEMIC"
print(result["all_detected"])  # ["ACADEMIC"]
print(result["scores"])        # {"ACADEMIC": 2, "WELLBEING": 0, ...}

# Build a transition log across turns
transition_log = []
transition_log = log_support_transition(transition_log, "ACADEMIC",  turn_number=1)
transition_log = log_support_transition(transition_log, "WELLBEING", turn_number=2)
print(transition_log[-1]["is_escalation"])  # True

# Select the right response
response = nexus_respond("", support_need="ACADEMIC", wellbeing_tier="STRESSED")
print(response)
```

---

### 8. `chatbot/nexus_report.py`

Session intelligence report for the counsellor.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `generate_intelligence_report` | `session_log: list[dict]`<br>`wellbeing_log: list[dict]`<br>`support_log: list[dict]`<br>`transition_log: list[dict]` | `None` | Computes a risk score, determines a recommended action (`NO_ACTION`, `FOLLOW_UP`, or `URGENT_REFERRAL`), and prints a formatted counsellor report. Called automatically at the end of `run_nexus_chatbot` and `run_offline_replay`. |

**Risk score formula:**
```
risk = 20
     + abs(avg_wellbeing_score) * 40
     + at_risk_count * 15
     + escalation_count * 10
     + (5 if avg_words_per_turn > 20 else 0)
risk = clamp(risk, 0, 100)
```

#### Example Usage

```python
from chatbot.nexus_report import generate_intelligence_report

generate_intelligence_report(
    session_log=[
        {"text": "I feel hopeless",     "source": "text",  "turn": 1, "word_count": 3},
        {"text": "My fees are overdue", "source": "voice", "turn": 2, "word_count": 4},
    ],
    wellbeing_log=[
        {"tier": "CRISIS",     "score": -0.80, "emoji": "🆘", "is_at_risk": True},
        {"tier": "DISTRESSED", "score": -0.45, "emoji": "😢", "is_at_risk": False},
    ],
    support_log=[
        {"primary": "WELLBEING"},
        {"primary": "FINANCIAL"},
    ],
    transition_log=[
        {"prev": "START",     "curr": "WELLBEING", "turn": 1, "is_escalation": True},
        {"prev": "WELLBEING", "curr": "FINANCIAL",  "turn": 2, "is_escalation": False},
    ],
)
# Prints:
# =====================================================
#         NEXUS STUDENT INTELLIGENCE REPORT
#         Code: NX-2B  —  Counsellor Eyes Only
# =====================================================
# Session Turns     : 2  (Voice: 1 | Text: 1)
# ...
```

---

### 9. `chatbot/chatbot_main.py`

Entry point and conversation loops for both chatbot modes.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `run_university_chatbot` | `input_mode: str` | `None` | Runs the university info chatbot loop. Each turn: collect input → spam check → classify intent → generate response → speak. Exits on `"exit"` or `"quit"`. |
| `run_nexus_chatbot` | `input_mode: str` | `None` | Runs the NEXUS wellbeing advisor loop. Each turn: assess wellbeing, check for crisis alert, classify support need, log transitions, generate and speak response. Prints the intelligence report on exit. |
| `run_offline_replay` | — | `None` | Processes the hardcoded `STUDENT_LOG` through the full NEXUS pipeline with no hardware (no mic, no webcam). Prints per-turn results and the intelligence report. |
| `_get_input_text` | `input_mode: str`<br>`turn: int` | `tuple[str, str]` — `(text, source)` | Collects one turn of input. In voice mode, falls back to typed input when recording returns empty text. `source` is `"text"` or `"voice"`. |
| `_record_and_transcribe` | — | `str` | Records audio via `capture_audio` and transcribes via Whisper. Returns `""` on any failure so callers handle gracefully. |
| `main` | — | `None` | Displays the main menu, reads `ENABLE_*` flags to build visible options, handles mode and input switching, and dispatches to the chosen conversation loop. |

#### Example Usage

```python
import sys; sys.path.insert(0, "path/to/hci_project")
from chatbot.chatbot_main import (
    run_university_chatbot, run_nexus_chatbot,
    run_offline_replay, main
)

# University chatbot — text input only
run_university_chatbot(input_mode="text")

# NEXUS advisor — voice input
run_nexus_chatbot(input_mode="voice")

# Offline replay — no hardware needed, good for testing
run_offline_replay()

# Full interactive menu (normal entry point)
main()
```

---

## Vision App

### 10. `vision/vision_utils.py`

Shared geometry helpers, drawing utilities, and the master frame-processing loop.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `euclidean` | `p1: tuple`<br>`p2: tuple` | `float` | L2 distance between two 2-D points `(x, y)`. |
| `lm_to_px` | `landmark`<br>`w: int`<br>`h: int` | `tuple[int, int]` | Converts a normalised MediaPipe landmark (`.x`/`.y` in `[0, 1]`) to pixel coordinates. |
| `compute_aspect_ratio` | `top: tuple`<br>`bottom: tuple`<br>`left: tuple`<br>`right: tuple` | `float` | `‖top−bottom‖ / (‖left−right‖ + ε)`. Used for both EAR (eyes) and MAR (mouth). |
| `draw_text` | `frame: ndarray`<br>`text: str`<br>`pos: tuple`<br>`color=(0,255,0)`<br>`scale=0.7`<br>`thickness=2` | `None` | Renders anti-aliased text with a black drop-shadow. Mutates `frame` in-place. |
| `draw_red_border` | `frame: ndarray`<br>`thickness=10` | `None` | Draws a red rectangle around the full frame (used as a threshold-crossing warning). |
| `draw_red_banner` | `frame: ndarray`<br>`message: str = "ALERT!"` | `None` | Overlays a semi-transparent red banner with centred white text across the middle of the frame. |
| `draw_esc_hint` | `frame: ndarray` | `None` | Draws a small grey `"ESC = Menu"` hint in the bottom-right corner. |
| `open_source` | `source: int \| str \| VideoCapture` | `cv2.VideoCapture` | Opens a webcam by index or video file by path. Raises `RuntimeError` if the source cannot be opened. |
| `read_frame` | `cap_or_image` | `tuple[bool, ndarray \| None]` | Reads one BGR frame from a `VideoCapture`, or returns a copy of a static image array. |
| `detection_loop` | `source`<br>`process_fn: callable`<br>`window_title: str` | `None` | Master loop: calls `process_fn(frame)` every frame, handles ESC (back to menu) and `q` (quit via `SystemExit`), replays videos from the start when they end. |

#### Example Usage

```python
import cv2
from vision.vision_utils import draw_text, draw_esc_hint, detection_loop

# Custom detection loop
def my_process(frame):
    draw_text(frame, "Live!", (20, 40), color=(0, 255, 0))
    draw_esc_hint(frame)
    return frame

detection_loop(source=0, process_fn=my_process, window_title="My App")

# Geometry helpers
from vision.vision_utils import euclidean, lm_to_px
dist = euclidean((0, 0), (3, 4))   # 5.0
```

---

### 11. `vision/mp_tasks.py`

MediaPipe Tasks API infrastructure: model download and landmarker factories.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `ensure_model` | `filename: str` | `pathlib.Path` | Returns the local path to a `.task` bundle, downloading it from Google CDN on first use. Raises `RuntimeError` if missing and cannot be downloaded. |
| `ensure_models` | — | `None` | Calls `ensure_model` for both `face_landmarker.task` and `hand_landmarker.task`. |
| `bgr_to_mp_image` | `frame: ndarray` | `mediapipe.Image` | Converts an OpenCV BGR frame (uint8, H×W×3) to a MediaPipe SRGB `Image` for the Tasks API. |
| `make_face_landmarker` | `running_mode: RunningMode`<br>`num_faces: int = 1` | `FaceLandmarker` (context manager) | Builds a `FaceLandmarker` using confidences from `vision_config.py`. Use with `with make_face_landmarker(...) as det:`. |
| `make_hand_landmarker` | `running_mode: RunningMode`<br>`num_hands: int = MAX_HANDS` | `HandLandmarker` (context manager) | Builds a `HandLandmarker` using confidences from `vision_config.py`. |

#### Example Usage

```python
import cv2
from mediapipe.tasks.python.vision import RunningMode
from vision.mp_tasks import ensure_models, bgr_to_mp_image, make_face_landmarker

ensure_models()   # download .task files on first run

frame = cv2.imread("photo.jpg")
mp_img = bgr_to_mp_image(frame)

with make_face_landmarker(RunningMode.IMAGE) as detector:
    result = detector.detect(mp_img)
    if result.face_landmarks:
        print(f"Detected {len(result.face_landmarks)} face(s)")
```

---

### 12. `vision/lips_module.py`

Mouth Aspect Ratio, smile detection, and lip-sync counter.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `compute_mar` | `landmarks: list`<br>`w: int`<br>`h: int` | `float` | Mouth Aspect Ratio: `‖top_lip − bottom_lip‖ / ‖left_corner − right_corner‖`. Higher = more open. |
| `detect_smile` | `landmarks: list`<br>`w: int`<br>`h: int` | `bool` | Returns `True` when both lip corners drop below the upper-lip centre by more than `SMILE_CORNER_THRESHOLD × mouth_width`. Scale-invariant. |
| `run_lips` | `source: int \| str \| ndarray` | `None` | Launches Lips Detection: draws lip mesh, shows Mood / MAR / open-close sync counter, and draws a red border when the mouth is above the open threshold. |

#### Example Usage

```python
import cv2
from mediapipe.tasks.python.vision import RunningMode
from vision.mp_tasks import make_face_landmarker, bgr_to_mp_image
from vision.lips_module import compute_mar, detect_smile, run_lips

frame = cv2.imread("face.jpg")
h, w  = frame.shape[:2]

with make_face_landmarker(RunningMode.IMAGE) as det:
    result = det.detect(bgr_to_mp_image(frame))
    if result.face_landmarks:
        lms = result.face_landmarks[0]
        print("MAR:",   compute_mar(lms, w, h))    # e.g. 0.12
        print("Smile:", detect_smile(lms, w, h))   # True / False

# Full live module (webcam)
run_lips(source=0)

# Static image
run_lips(source=frame)
```

---

### 13. `vision/eyes_module.py`

Blink counter, Eye Aspect Ratio, and drowsiness alert with personal calibration.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `compute_ear` | `landmarks: list`<br>`eye_indices: list[int]`<br>`w: int`<br>`h: int` | `float` | Eye Aspect Ratio using 6 landmarks and the Soukupova & Cech (2016) formula. Lower = more closed. |
| `_calibrate_threshold` | `samples: list[float]` | `float` | Derives a personal closed-eye threshold: trims the bottom 10 % of samples, multiplies the baseline mean by `EAR_CLOSED_RATIO`, and clamps to `[EAR_FLOOR, EAR_CEILING]`. |
| `run_eyes` | `source: int \| str \| ndarray` | `None` | Launches Eyes Detection with an automatic calibration phase (progress bar shown). After calibration displays EAR per eye, blink count, and a drowsiness banner when triggered. |

#### Example Usage

```python
import cv2
from mediapipe.tasks.python.vision import RunningMode
from vision.mp_tasks import make_face_landmarker, bgr_to_mp_image
from vision.landmarks import FACE_LEFT_EYE, FACE_RIGHT_EYE
from vision.eyes_module import compute_ear, run_eyes

frame = cv2.imread("face.jpg")
h, w  = frame.shape[:2]

with make_face_landmarker(RunningMode.IMAGE) as det:
    result = det.detect(bgr_to_mp_image(frame))
    if result.face_landmarks:
        lms = result.face_landmarks[0]
        ear_l = compute_ear(lms, FACE_LEFT_EYE,  w, h)
        ear_r = compute_ear(lms, FACE_RIGHT_EYE, w, h)
        print(f"EAR  Left: {ear_l:.3f}   Right: {ear_r:.3f}")

# Full live module
run_eyes(source=0)
```

---

### 14. `vision/face_module.py`

Emotion recognition (DeepFace), head pose estimation, and rolling mood window.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `classify_head_pose` | `landmarks: list`<br>`w: int`<br>`h: int` | `str` — `"Left"\|"Right"\|"Up"\|"Down"\|"Forward"` | Compares nose tip to inter-eye midpoint normalised by inter-eye distance to estimate coarse head orientation. |
| `bbox_from_landmarks` | `landmarks: list`<br>`w: int`<br>`h: int`<br>`pad: float = 0.10` | `tuple[int, int, int, int]` — `(x, y, w, h)` | Padded face bounding box from landmark extents, clamped to the frame. Used to crop the face ROI for DeepFace. |
| `run_face` | `source: int \| str \| ndarray` | `None` | Launches Face Detection: draws bounding box, runs throttled DeepFace emotion inference (`EMOTION_INFERENCE_EVERY` frames), shows rolling recent-mood window, and displays head pose. |

#### Example Usage

```python
import cv2
from mediapipe.tasks.python.vision import RunningMode
from vision.mp_tasks import make_face_landmarker, bgr_to_mp_image
from vision.face_module import classify_head_pose, bbox_from_landmarks, run_face

frame = cv2.imread("face.jpg")
h, w  = frame.shape[:2]

with make_face_landmarker(RunningMode.IMAGE) as det:
    result = det.detect(bgr_to_mp_image(frame))
    if result.face_landmarks:
        lms = result.face_landmarks[0]
        print("Pose:", classify_head_pose(lms, w, h))   # "Forward"
        x, y, bw, bh = bbox_from_landmarks(lms, w, h)
        print(f"Face box: ({x},{y})  {bw}x{bh}")

# Full live module
run_face(source=0)
```

---

### 15. `vision/hand_module.py`

Finger counting, gesture recognition, ASL letter detection, and gesture game.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `count_fingers` | `landmarks: list`<br>`handedness: str`<br>`w: int`<br>`h: int` | `tuple[int, bool]` — `(non_thumb_count, thumb_extended)` | Counts extended fingers (tip Y < PIP Y) and checks thumb using X-axis with chirality correction for palm-facing pose. |
| `classify_gesture` | `fingers_up: int`<br>`thumb_up: bool` | `tuple[str, str]` — `(label, tag)` | Maps `(fingers_up, thumb_up)` to a gesture label + ASCII tag via `GESTURE_MAP` in config. Falls back to `(fingers_up, False)` for unknown combos. |
| `detect_letter` | `fingers_up: int`<br>`thumb_up: bool` | `str \| None` | Looks up the hand pose in `LETTER_GESTURE_MAP` and returns the matched letter, or `None`. Returns `None` immediately if `ENABLE_LETTER_DETECTION = False`. |
| `run_hands` | `source: int \| str \| ndarray` | `None` | Launches Hand Detection: draws hand skeletons, annotates gesture + letter near each wrist, shows the gesture game HUD at the bottom when `ENABLE_GESTURE_GAME = True`. |

#### Example Usage

```python
import cv2
from mediapipe.tasks.python.vision import RunningMode
from vision.mp_tasks import make_hand_landmarker, bgr_to_mp_image
from vision.hand_module import count_fingers, classify_gesture, detect_letter, run_hands

frame = cv2.imread("hand.jpg")
h, w  = frame.shape[:2]

with make_hand_landmarker(RunningMode.IMAGE) as det:
    result = det.detect(bgr_to_mp_image(frame))
    if result.hand_landmarks:
        lms        = result.hand_landmarks[0]
        handedness = result.handedness[0][0].display_name  # "Right" or "Left"

        fingers_up, thumb_up = count_fingers(lms, handedness, w, h)
        label, tag = classify_gesture(fingers_up, thumb_up)
        letter     = detect_letter(fingers_up, thumb_up)

        print(f"Fingers: {fingers_up}   Thumb: {thumb_up}")
        print(f"Gesture: {label} [{tag}]")
        if letter:
            print(f"Letter: {letter}")

# Full live module
run_hands(source=0)
```

---

### 16. `vision/hybrid_module.py`

Runs any combination of the four feature sets simultaneously in one panel.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `run_hybrid` | `source: int \| str \| ndarray` | `None` | Reads `HYBRID_ACTIVE_MODULES` from config and runs all listed features on every frame. The FaceLandmarker is shared across lips/eyes/face so only one inference call is made per frame. Annotations stack top-left with colour-coded section headers. Eye calibration runs in the background while other features keep annotating. `contextlib.ExitStack` cleanly closes both detectors on any exit path. |

#### Example Usage

```python
# Control from vision_config.py only — no code changes needed:
#
#   ENABLE_HYBRID_MODE    = True
#   HYBRID_ACTIVE_MODULES = ["lips", "eyes"]               # two features
#   HYBRID_ACTIVE_MODULES = ["face", "hands"]              # emotion + gestures
#   HYBRID_ACTIVE_MODULES = ["lips", "eyes", "face", "hands"]  # all four

from vision.hybrid_module import run_hybrid

run_hybrid(source=0)                          # webcam
run_hybrid(source="path/to/video.mp4")        # video file

import cv2
run_hybrid(source=cv2.imread("photo.jpg"))    # static image
```

---

### 17. `vision/vision_main.py`

Entry point and main menu for the Vision app.

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `main` | — | `None` | Reads `ENABLE_*` flags at startup to build the active module list (disabled modules hidden, numbering compact). Ensures model files are present. Repeatedly shows the menu, prompts for an input source, and dispatches to the selected module. |

#### Example Usage

```python
# Standard usage — run from hci_project/:
#   python run_vision.py
#
# Or import directly:
import sys; sys.path.insert(0, "path/to/hci_project")
from vision.vision_main import main
main()
```

**Interactive menu flow:**
```
Select Detection Mode:
  1. Lips Detection
  2. Eyes Detection
  3. Face Detection
  4. Hand Detection
  5. Hybrid Mode
  0. Exit

Select Input Source:
  1. Live Webcam
  2. Video File
  3. Image File
  0. Back
```

---

## Configuration Quick Reference

### `chatbot/chatbot_config.py`

| Flag | Default | Effect |
|---|---|---|
| `CHATBOT_MODE` | `"university"` | Starting mode: `"university"` or `"nexus"` |
| `DEFAULT_INPUT_MODE` | `"hybrid"` | Starting input: `"text"`, `"voice"`, or `"hybrid"` |
| `INTENT_CLASSIFICATION_MODE` | `"rule_based"` | `"rule_based"` = keyword match; `"ai"` = LLM classifies intent |
| `USE_LLM` | `True` | `True` = Ollama for responses; `False` = static strings only |
| `OLLAMA_MODEL` | `"qwen2.5-coder:7b"` | Ollama model name |
| `WHISPER_MODEL` | `"base"` | Whisper model size (`"tiny"`, `"base"`, `"small"`, `"medium"`) |
| `AUDIO_RECORD_SECONDS` | `7` | Seconds of audio captured per voice turn |
| `ENABLE_UNIVERSITY_MODE` | `True` | Show/hide University Chatbot in menu |
| `ENABLE_NEXUS_MODE` | `True` | Show/hide NEXUS Advisor in menu |
| `ENABLE_OFFLINE_REPLAY` | `True` | Show/hide offline session replay option |

### `vision/vision_config.py`

| Flag | Default | Effect |
|---|---|---|
| `ENABLE_LIPS_DETECTION` | `True` | Show Lips module in menu |
| `ENABLE_EYES_DETECTION` | `True` | Show Eyes module in menu |
| `ENABLE_FACE_DETECTION` | `True` | Show Face module in menu |
| `ENABLE_HAND_DETECTION` | `True` | Show Hand module in menu |
| `ENABLE_HYBRID_MODE` | `True` | Show Hybrid mode in menu |
| `HYBRID_ACTIVE_MODULES` | `["lips","eyes","face","hands"]` | Features active in hybrid mode (any subset) |
| `ENABLE_GESTURE_GAME` | `True` | Show gesture game HUD in hand/hybrid mode |
| `ENABLE_LETTER_DETECTION` | `True` | Enable ASL letter lookup |
| `CALIBRATION_FRAMES` | `60` | Frames sampled for eye EAR auto-calibration |
| `EMOTION_INFERENCE_EVERY` | `5` | Run DeepFace every N frames |
| `MAR_OPEN_THRESHOLD` | `0.55` | MAR above this triggers open-mouth detection |
| `DROWSY_FRAMES_TRIGGER` | `20` | Consecutive closed-eye frames before drowsiness fires |
