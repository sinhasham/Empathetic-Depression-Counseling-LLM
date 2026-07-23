from groq import Groq

client = Groq(api_key = "gsk_your_actual_key"")

def generate_empathy(emotion_analysis):
    prompt = f"""You are an empathetic mental health counselor.

Note: The Emotion Analysis below may reference content originally in Hindi, Hinglish, or 
English. Understand it fully regardless of language, but write your reflection in English - 
this is internal reasoning, not shown directly to the user.

Emotion Analysis: {emotion_analysis}

Look at the Emotion Analysis above. If it indicates "None - casual conversation" or no 
real problem/distress, do NOT manufacture deep empathy or imply the person is struggling. 
Instead, write ONE short, warm, genuinely interested reflection that matches a casual 
conversation tone (like a friend showing interest), 1-2 lines.

If the Emotion Analysis DOES indicate real distress or a negative emotional state, write 
ONE short empathetic reflection (2-3 lines max) that validates the feeling without giving advice.

Do not give advice in either case. Just reflect understanding appropriate to the tone."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content