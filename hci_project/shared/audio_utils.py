"""
Shared audio utilities: microphone capture and Whisper transcription.
Used by both the vision app (future) and the chatbot app.
"""

import time

_whisper_model_cache: dict = {}


def capture_audio(seconds: int = 7, sample_rate: int = 16000):
    """
    Record audio from the default microphone.

    Recording starts immediately; a countdown is displayed while the
    buffer fills so the user knows how long is left.

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
    RuntimeError
        If the microphone cannot be accessed.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as exc:
        print("[ERROR] sounddevice not installed. Run: pip install sounddevice")
        raise ImportError("sounddevice unavailable") from exc

    print(f"\n[Voice] Recording for {seconds} seconds — please speak now.")
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
        print(f"\n[ERROR] Recording failed: {exc}")
        raise RuntimeError("Microphone recording failed") from exc

    print("\n[Voice] Recording complete.            ")
    import numpy as np
    return np.squeeze(recording)


def transcribe(audio_array, sample_rate: int = 16000, model_name: str = "base") -> dict:
    """
    Transcribe a float32 audio array using OpenAI Whisper.

    The Whisper model is loaded once and cached for subsequent calls.

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
    except ImportError:
        print("[ERROR] openai-whisper not installed. Run: pip install openai-whisper")
        return {"text": "", "language": "unknown", "confidence": "low"}

    try:
        if model_name not in _whisper_model_cache:
            print(f"[Voice] Loading Whisper model '{model_name}'...")
            _whisper_model_cache[model_name] = whisper.load_model(model_name)
        model = _whisper_model_cache[model_name]
        result = model.transcribe(audio_array, fp16=False)
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        confidence = "high" if len(text) > 10 else "low"
        return {"text": text, "language": language, "confidence": confidence}
    except Exception as exc:
        print(f"[WARN] Whisper transcription failed: {exc}")
        return {"text": "", "language": "unknown", "confidence": "low"}