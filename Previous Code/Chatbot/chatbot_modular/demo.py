"""
chatbot_modular/demo.py
=========================
NEXUS offline session replay — Stage 4 of the assessment brief.

The brief supplies a 10-line STUDENT_LOG that exercises every part of
the pipeline (academic, technical, wellbeing-distress, crisis, social,
recovery). run_demo_tests() feeds this log into NEXUS turn by turn,
prints each turn's emoji + tier + primary category + response, and
then asks the report generator for the full counsellor briefing.

No microphone or internet needed — every turn is treated as a text
input with a synthesised input dict.

Public API
----------
    run_demo_tests(classifier=None, tts_engine=None) -> dict
"""

from chatbot_modular.pipeline         import new_session, process_turn
from chatbot_modular.report_generator import generate_intelligence_report


# The exact 10-line conversation log from the brief.
STUDENT_LOG = (
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
)


def run_demo_tests(classifier=None, tts_engine=None):
    """
    Replay the brief's STUDENT_LOG through the NEXUS pipeline.

    The first argument is accepted but unused — it is kept so the
    older menu (main_modular.py) can pass its `classifier` value
    without an error.

    Args:
        classifier: ignored (legacy positional argument).
        tts_engine: optional pyttsx3 engine — None during the demo so
                    the replay runs at console speed.

    Returns:
        The dict produced by generate_intelligence_report().
    """
    border = "=" * 60
    print(f"\n{border}")
    print("  NEXUS — OFFLINE SESSION REPLAY (Stage 4)")
    print(f"  Replaying {len(STUDENT_LOG)} student turns")
    print(border)

    state = new_session()

    for i, msg in enumerate(STUDENT_LOG, start=1):
        input_dict = {
            "text":       msg,
            "source":     "text",
            "turn":       i,
            "word_count": len(msg.split()),
        }
        print(f"\nStudent: {msg}")
        process_turn(input_dict, state, tts_engine=tts_engine)

    print(f"\n{border}")
    print("  Generating counsellor intelligence report...")
    print(border)
    report = generate_intelligence_report(
        state["session_log"],
        state["wellbeing_log"],
        state["support_log"],
        state["transition_log"],
    )
    return report


if __name__ == "__main__":
    run_demo_tests()
