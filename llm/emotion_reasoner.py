import httpx
from groq import Groq

client = Groq(
    api_key="api_key = "gsk_your_actual_key"7",
    http_client=httpx.Client(verify=False)
)

PROMPT = """You are a mental health AI assistant analyzing a single message in an ongoing conversation.

PHQ-9 Profile: {phq_profile}
Assigned Severity Agent: {agent} (this reflects overall background risk level, NOT this specific message)
User Message: {user_message}

Important: The user may write in Hindi, Hinglish, or English. Understand fully regardless of language.

First, judge THIS SPECIFIC MESSAGE on its own — not the background severity:
- If the user is sharing something casual (a movie, show, sport, hobby, random fact, light comment), 
  do NOT invent a "problem" or assign a negative emotion. Reflect the actual tone (curious, happy, neutral, excited).
- Only identify a real Problem/Cognitive Distortion if THIS message genuinely contains distress, 
  negative self-talk, or a concern the user raised.
- Risk level should reflect this message's content, but you may keep it aligned with the background 
  severity if the message itself doesn't change anything.

Respond ONLY in this exact format, in English:
Emotion: <emotion>
Problem: <main problem, or "None - casual conversation">
Risk: <low/medium/high>
Topic: <topic of this message, casual or clinical as appropriate>
Cognitive Distortion: <distortion if any, or "None">"""


def analyze_emotion(user_message, phq_profile, agent="Agent_M"):
    prompt = PROMPT.format(
        phq_profile=phq_profile,
        agent=agent,
        user_message=user_message
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content