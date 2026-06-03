"""
Post-install setup script.
Run once after activating hci_env:  python setup_env.py
Downloads NLTK corpora, TextBlob corpora, and pre-caches the Whisper base model.
"""

import sys


def download_nltk_corpora() -> None:
    """Download required NLTK corpora for sentiment analysis and text processing."""
    try:
        import nltk
        corpora = [
            "vader_lexicon",
            "punkt",
            "punkt_tab",
            "stopwords",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
        ]
        for corpus in corpora:
            print(f"  Downloading NLTK: {corpus} ...", end=" ", flush=True)
            nltk.download(corpus, quiet=True)
            print("OK")
        print("[NLTK] All corpora downloaded.\n")
    except ImportError:
        print("[ERROR] nltk not installed. Run: pip install nltk")
        sys.exit(1)


def download_textblob_corpora() -> None:
    """Download TextBlob corpora required for sentiment and NLP features."""
    try:
        import subprocess
        print("  Downloading TextBlob corpora ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "textblob.download_corpora"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("OK")
        else:
            print(f"WARN (may already be installed): {result.stderr.strip()[:120]}")
        print("[TextBlob] Corpora step complete.\n")
    except Exception as exc:
        print(f"[WARN] TextBlob corpora download failed: {exc}")


def cache_whisper_model() -> None:
    """Pre-download and cache the Whisper 'base' model (~145 MB)."""
    try:
        import whisper
        print("  Downloading Whisper 'base' model (~145 MB) ...", end=" ", flush=True)
        whisper.load_model("base")
        print("OK")
        print("[Whisper] base model cached.\n")
    except ImportError:
        print("[ERROR] openai-whisper not installed. Run: pip install openai-whisper")
    except Exception as exc:
        print(f"[WARN] Whisper model download failed: {exc}")


def download_mediapipe_models() -> None:
    """
    Pre-download the MediaPipe Tasks model bundles used by the vision app
    (face_landmarker.task and hand_landmarker.task) into vision/models/.
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from vision.mp_tasks import ensure_model, MODEL_URLS
        for name in MODEL_URLS:
            print(f"  Fetching MediaPipe model: {name} ...", end=" ", flush=True)
            ensure_model(name)
            print("OK")
        print("[MediaPipe] Task model bundles ready.\n")
    except Exception as exc:
        print(f"[WARN] MediaPipe model download failed: {exc}")
        print("       (They will be downloaded automatically on first vision run.)\n")


def cache_deepface_model() -> None:
    """
    Warm up DeepFace's emotion model so the first face-detection run is fast.

    DeepFace downloads its facial-expression weights (~5 MB) on first use.
    Running one analyze() call here pre-caches them. Requires the tf-keras
    package (installed via environment.yml) on TensorFlow >= 2.16.
    """
    try:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("GLOG_minloglevel", "3")
        import numpy as np
        from deepface import DeepFace
        print("  Warming up DeepFace emotion model (~5 MB) ...", end=" ", flush=True)
        dummy = (np.random.rand(160, 160, 3) * 255).astype("uint8")
        DeepFace.analyze(dummy, actions=["emotion"],
                         enforce_detection=False, silent=True)
        print("OK")
        print("[DeepFace] Emotion model cached.\n")
    except Exception as exc:
        print(f"[WARN] DeepFace warm-up failed: {exc}")
        print("       Fix: pip install tf-keras   (DeepFace needs it on TF >= 2.16)\n")


def main() -> None:
    """Run all setup steps sequentially."""
    print("=" * 60)
    print("  HCI Project — Environment Setup")
    print("=" * 60)
    print()

    print("[1/5] NLTK corpora")
    download_nltk_corpora()

    print("[2/5] TextBlob corpora")
    download_textblob_corpora()

    print("[3/5] Whisper model")
    cache_whisper_model()

    print("[4/5] MediaPipe Tasks model bundles")
    download_mediapipe_models()

    print("[5/5] DeepFace emotion model")
    cache_deepface_model()

    print("=" * 60)
    print("  Setup complete!")
    print()
    print("  MANUAL STEP REMAINING:")
    print("  ollama pull llama3   (~4.7 GB — do this on fast internet)")
    print("=" * 60)


if __name__ == "__main__":
    main()
