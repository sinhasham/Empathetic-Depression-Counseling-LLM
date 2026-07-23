from groq import Groq

client = Groq(api_key = "gsk_your_actual_key")

def generate_cot(emotion_analysis, agent):
    prompt = f"""You are a CBT counselor ({agent}).

Note: The Emotion Analysis below may reference content originally in Hindi, Hinglish, or 
English. Understand it fully regardless of language, but write your reasoning in English - 
this is internal reasoning, not shown directly to the user.

Emotion Analysis: {emotion_analysis}

Look at the Emotion Analysis above carefully. If it indicates "None - casual conversation" 
or no real problem/distress (the person was just chatting about something casual like a 
movie, show, hobby, or random topic), do NOT force a clinical CBT structure onto it.

In that case, respond with:
Problem: None - casual conversation
Cause: N/A
Goal: Maintain natural rapport and engagement
Intervention: Continue the casual conversation naturally, showing genuine interest

If the Emotion Analysis DOES indicate a real problem, distress, or negative pattern, 
then respond with full clinical CBT reasoning in this exact format:
Problem: <problem>
Cause: <cause>
Goal: <ents <therapeutic goal>
Intervention: <CBT intervention>"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content