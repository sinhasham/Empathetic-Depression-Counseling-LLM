# agents/safety.py

CRISIS_KEYWORDS = [
    "suicide", "suicidal",
    "kill myself", "kill my self",
    "end my life", "end my self",
    "want to die", "wanna die",
    "self harm", "self-harm",
    "cut myself", "cut my self",
    "hurt myself", "hurt my self",
    "no reason to live",
    "better off dead",
    "can't go on", "cant go on",
    "overdose",
    "jump off",
    "hang myself", "hang my self",
    "nobody cares",
    "disappear forever",
    "i give up",
    "life is pointless",
    "don't want to live", "dont want to live",
    "not worth living",
    "end it all"
]

ESCALATION_RESOURCES = """
🆘 IMMEDIATE SUPPORT AVAILABLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📞 iCall (India):        9152987821
📞 Vandrevala Foundation: 1860-2662-345 (24/7)
📞 AASRA:               91-22-27546669
💬 iCall Chat: https://icallhelpline.org
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are not alone. Help is available right now.
"""

def check_risk(user_message, emotion_json=None):
    """
    Returns:
        risk_level: 'low' / 'medium' / 'high' / 'crisis'
        escalate: True/False
        message: escalation message if needed
    """
    message_lower = user_message.lower()

    # Crisis keyword check
    for keyword in CRISIS_KEYWORDS:
        if keyword in message_lower:
            return {
                "risk_level": "crisis",
                "escalate": True,
                "trigger": keyword
            }

    # Emotion JSON risk check
    if emotion_json:
        risk = emotion_json.get("risk", "low")
        if risk == "high":
            return {
                "risk_level": "high",
                "escalate": False,
                "trigger": None
            }

    return {
        "risk_level": "low",
        "escalate": False,
        "trigger": None
    }

def get_crisis_response(agent):
    return f"""
I hear you, and I'm really glad you're talking to me right now.

What you're feeling matters, and you don't have to face this alone.

Before we continue, I want to make sure you're safe.
Please reach out to a crisis counselor who can support you right now:

{ESCALATION_RESOURCES}

I'm still here with you. Can you tell me — are you safe right now?
"""