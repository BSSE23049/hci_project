# Top-level launcher for the vision app
# Usage: python run_vision.py
import sys, os

# Quieten TensorFlow / MediaPipe / absl startup logs (must be set before import)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

sys.path.insert(0, os.path.dirname(__file__))
from vision.vision_main import main
if __name__ == "__main__":
    main()
