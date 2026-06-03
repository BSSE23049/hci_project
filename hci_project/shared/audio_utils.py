"""
Shared audio utilities: microphone capture and Whisper transcription.
Used by both the vision app (future) and the chatbot app.
"""

import time


def capture_audio(seconds: int = 7, sample_rate: int = 16000):
    """
    Record audio from the default microphone.

    Parameters
    ----------
    seconds : int
        Duration to record in seconds.
    sample_rate : int
        Samples per second (Hz).

    Returns
    -------
    numpy.ndarray
        Float32 array of shape (seconds * sample_rate,).

    Raises
    ------
    ImportError
        If sounddevice is not installed.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as exc:
        print("[ERROR] sounddevice not installed. Run: pip install sounddevice")
        raise ImportError("sounddevice unavailable") from exc

    print(f"\nGet ready to speak ({seconds}s)...")
    for remaining in range(seconds, 0, -1):
        print(f"  Recording... {remaining}s", end="\r", flush=True)
        time.sleep(1)
    print()

    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def transcribe(audio_array, sample_rate: int = 16000, model_name: str = "base") -> dict:
    """
    Transcribe a float32 audio array using OpenAI Whisper.

    Parameters
    ----------
    audio_array : numpy.ndarray
        Float32 audio samples, shape (N,).
    sample_rate : int
        Sample rate of the audio (must be 16000 for Whisper).
    model_name : str
        Whisper model size: 'tiny', 'base', 'small', 'medium', 'large'.

    Returns
    -------
    dict
        Keys: 'text' (str), 'language' (str), 'confidence' ('high'|'low').
    """
    try:
        import whisper
        import numpy as np
    except ImportError as exc:
        print("[ERROR] openai-whisper not installed. Run: pip install openai-whisper")
        return {"text": "", "language": "unknown", "confidence": "low"}

    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_array, fp16=False)
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        confidence = "high" if len(text) > 10 else "low"
        return {"text": text, "language": language, "confidence": confidence}
    except Exception as exc:
        print(f"[WARN] Whisper transcription failed: {exc}")
        return {"text": "", "language": "unknown", "confidence": "low"}
