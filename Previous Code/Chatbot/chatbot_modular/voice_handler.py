"""
chatbot_modular/voice_handler.py
==================================
NEXUS Input Layer — Stage 1 of the assessment brief.

This module exposes the three mandated function names from Stage 1
(nexus_capture_audio, nexus_transcribe, nexus_get_input) plus the
older Text-to-Speech helpers (initialize_tts, speak_text) that the
interactive demo continues to rely on for spoken NEXUS replies.

Stage 1 — required functions
----------------------------
  nexus_capture_audio(seconds=7, sample_rate=16000)
      Record from the microphone with sounddevice and return a
      float32 numpy array. Prints a countdown so the student knows
      how long is left.

  nexus_transcribe(audio_array, sample_rate)
      Transcribe with Whisper model 'small' (not 'base').
      Returns {'text': str, 'language': str, 'confidence': str}.
      Confidence is 'high' when len(text) > 10, else 'low'.

  nexus_get_input(turn_number)
      Ask the student to choose voice or text, then capture & return:
        {'text': str, 'source': str, 'turn': int, 'word_count': int}

Output helpers (unchanged behaviour, retained for compatibility)
-----------------------------------------------------------------
  initialize_tts()        — pyttsx3 engine builder
  speak_text(text, eng)   — spoken NEXUS reply
  initialize_voice_components() — legacy SpeechRecognition components

Optional imports
----------------
sounddevice, numpy, and whisper are imported lazily. If they are not
installed, voice capture silently degrades to text-only mode without
raising at import time — useful for grading machines that only run the
text demo.
"""

import re
import sys
import time
import subprocess
from typing import Optional, Tuple, Dict


# --- Optional voice-IN libraries (lazy) ------------------------------------
try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False

try:
    import numpy as np
    NP_AVAILABLE = True
except Exception:
    NP_AVAILABLE = False

_whisper_model = None  # built on first nexus_transcribe() call


# --- Optional legacy SpeechRecognition (older demos) -----------------------
try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except Exception:
    STT_AVAILABLE = False


# --- Optional TTS (pyttsx3) ------------------------------------------------
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False


# ===========================================================================
# Q1.1 — Audio Capture
# ===========================================================================

def nexus_capture_audio(seconds: int = 7, sample_rate: int = 16000):
    """
    Record `seconds` of audio from the system microphone for NEXUS.

    Returns
    -------
    numpy.ndarray, dtype float32, shape (seconds * sample_rate,)
        On success — a 1-D mono recording in the range [-1.0, +1.0].
    None
        If sounddevice / numpy aren't installed, or the recording
        fails (e.g. no microphone). Callers should fall back to
        typed input.

    The countdown is informational; sd.rec returns immediately and
    fills the buffer in the background, so the recording runs for the
    full `seconds` window in parallel with the prints.
    """
    if not (SD_AVAILABLE and NP_AVAILABLE):
        print("[NEXUS] Voice capture unavailable — sounddevice/numpy not installed.")
        print("        Install with: pip install sounddevice numpy")
        return None

    print(f"\n[NEXUS] Recording for {seconds} seconds — please speak now.")
    try:
        recording = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        for remaining in range(seconds, 0, -1):
            print(f"  ... {remaining} second(s) remaining", end="\r", flush=True)
            time.sleep(1)
        sd.wait()
    except Exception as exc:
        print(f"\n[NEXUS] Recording failed: {exc}")
        print("        Is a microphone connected and accessible?")
        return None
    print("\n[NEXUS] Recording complete.            ")
    return np.squeeze(recording)


# ===========================================================================
# Q1.2 — Whisper Transcription
# ===========================================================================

def _get_whisper_model():
    """Cache and return the Whisper 'small' model. None if unavailable."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
        _whisper_model = whisper.load_model("small")
    except Exception as exc:
        print(f"[NEXUS] Whisper unavailable: {exc}")
        _whisper_model = None
    return _whisper_model


def nexus_transcribe(audio_array, sample_rate: int = 16000) -> Dict:
    """
    Transcribe a NEXUS audio array using Whisper model 'small'.

    Args:
        audio_array: float32 numpy array from nexus_capture_audio().
        sample_rate: sample rate of the audio (kept for API symmetry —
                     Whisper resamples internally if needed).

    Returns:
        dict with keys:
          'text'       — transcribed text (str)
          'language'   — detected language code (str)
          'confidence' — 'high' if len(text) > 10, else 'low'
    """
    if audio_array is None:
        return {"text": "", "language": "unknown", "confidence": "low"}

    model = _get_whisper_model()
    if model is None:
        return {"text": "", "language": "unknown", "confidence": "low"}

    try:
        # Whisper expects float32 audio at 16 kHz; we already produce
        # that format in nexus_capture_audio().
        result = model.transcribe(audio_array, fp16=False)
        text     = (result.get("text") or "").strip()
        language = result.get("language") or "unknown"
    except Exception as exc:
        print(f"[NEXUS] Whisper transcription failed: {exc}")
        return {"text": "", "language": "unknown", "confidence": "low"}

    confidence = "high" if len(text) > 10 else "low"
    return {"text": text, "language": language, "confidence": confidence}


# ===========================================================================
# Q1.3 — Unified Input with Source Tracking
# ===========================================================================

def nexus_get_input(turn_number: int) -> Dict:
    """
    Capture a single turn of student input via voice or text.

    The student is shown a prompt that includes the turn number so
    they can orient themselves in the session. They choose 'v' for
    voice or 't' (or just press Enter) for text. If voice is
    selected, nexus_capture_audio() + nexus_transcribe() are called;
    otherwise their typed line is used directly.

    Args:
        turn_number: 1-based turn index, surfaced in the prompt.

    Returns:
        dict with keys:
          'text'       — the student's message
          'source'     — 'voice' or 'text'
          'turn'       — turn_number (echoed back)
          'word_count' — number of whitespace-separated tokens in 'text'
    """
    print(f"\n=== TURN {turn_number} ===")
    try:
        choice = input("Press [v] for voice, [t] (or Enter) for text: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return {"text": "", "source": "text", "turn": turn_number, "word_count": 0}

    if choice == "v" and SD_AVAILABLE and NP_AVAILABLE:
        audio  = nexus_capture_audio()
        result = nexus_transcribe(audio)
        text   = result["text"]
        source = "voice"
        if not text:
            # Whisper returned nothing — fall back to a typed prompt so
            # the session can continue without losing this turn.
            print("[NEXUS] No speech detected. Please type your message instead.")
            try:
                text = input(f"You (turn {turn_number}): ").strip()
            except (KeyboardInterrupt, EOFError):
                text = ""
            source = "text"
    else:
        try:
            text = input(f"You (turn {turn_number}): ").strip()
        except (KeyboardInterrupt, EOFError):
            text = ""
        source = "text"

    return {
        "text":       text,
        "source":     source,
        "turn":       turn_number,
        "word_count": len(text.split()),
    }


# ===========================================================================
# TTS / legacy STT helpers — unchanged behaviour
# ===========================================================================

def initialize_tts():
    """Build and return a configured pyttsx3 engine, or None if unavailable."""
    if not TTS_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate",   160)
        engine.setProperty("volume", 0.9)
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)
        return engine
    except Exception as exc:
        print(f"[TTS] Initialisation failed: {exc}")
        return None


_TTS_SUBPROCESS_SCRIPT = (
    "import sys, pyttsx3\n"
    "e = pyttsx3.init()\n"
    "e.setProperty('rate', 160)\n"
    "e.setProperty('volume', 0.9)\n"
    "v = e.getProperty('voices')\n"
    "if v: e.setProperty('voice', v[0].id)\n"
    "e.say(sys.argv[1])\n"
    "e.runAndWait()\n"
)


def speak_text(text: str, tts_engine) -> None:
    """Speak a NEXUS response aloud via a fresh pyttsx3 subprocess."""
    if tts_engine is None or not TTS_AVAILABLE:
        return

    time.sleep(0.3)

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    spoken_lines = []
    for ln in lines:
        if re.match(r"^[-|:=\s]+$", ln):
            continue
        spoken_lines.append(ln)
        if len(spoken_lines) >= 3:
            break
    spoken = ". ".join(spoken_lines)[:300]
    clean  = re.sub(r"[*_`#|]", " ", spoken)
    clean  = re.sub(r":{1,}",   ",", clean)
    clean  = re.sub(r"-{2,}",   " ", clean)
    clean  = re.sub(r"\s+",     " ", clean).strip()

    if not clean:
        return

    print("  [TTS] Speaking response...")

    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", _TTS_SUBPROCESS_SCRIPT, clean],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            _, stderr = proc.communicate(timeout=90)
            if proc.returncode != 0 and stderr:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                if err_msg:
                    print(f"  [TTS] Speech error: {err_msg.splitlines()[-1]}")
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.communicate(timeout=2)
            except Exception:
                pass
            print("  [TTS] Speech subprocess timed out – continuing.")
    except Exception as exc:
        print(f"  [TTS] Subprocess launch error: {exc}")
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    time.sleep(0.2)


def speech_to_text(recognizer, microphone) -> Optional[str]:
    """Legacy Google-API-based STT. Kept so older demos still run."""
    if recognizer is None or microphone is None:
        return None
    print("\n  [STT] Adjusting for ambient noise - please wait...")
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("  [STT] Listening... (speak now, up to 10 seconds)")
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
        print("  [STT] Recognising speech...")
        text = recognizer.recognize_google(audio)
        print(f'  [STT] Recognised: "{text}"')
        return text
    except sr.WaitTimeoutError:
        print("  [STT] Timeout - no speech detected within 8 seconds.")
    except sr.UnknownValueError:
        print("  [STT] Could not understand audio. Please speak more clearly.")
    except sr.RequestError as exc:
        print(f"  [STT] Google API error: {exc}. Check internet connection.")
    except Exception as exc:
        print(f"  [STT] Unexpected error: {exc}")
    return None


def initialize_voice_components() -> Tuple:
    """Build SpeechRecognition Recognizer + Microphone, or (None, None)."""
    if not STT_AVAILABLE:
        return None, None
    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold        = 300
        recognizer.dynamic_energy_threshold = True
        microphone = sr.Microphone()
        return recognizer, microphone
    except Exception as exc:
        print(f"  [Voice] Setup error: {exc}")
        return None, None
