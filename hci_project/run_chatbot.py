# Top-level launcher for the chatbot app
# Usage: python run_chatbot.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from chatbot.chatbot_main import main
if __name__ == "__main__":
    main()
