import httpx
from groq import Groq

client = Groq(
    api_key="api_key = "gsk_your_actual_key"7",
    http_client=httpx.Client(verify=False)
)

PHQ_QUESTIONS = [
    "To start, how have your days been feeling lately - like, do things still feel interesting or has everything started feeling a bit flat?",
    "How would you describe your mood over the past two weeks? Like, has there been a heaviness or sadness sitting with you?",
    "Tell me about your sleep - are you someone who crashes easily, or does your mind keep you up at night?",
    "How's your energy been? Do you feel like doing things, or does everything feel like a lot of effort lately?",
    "What about eating - have you noticed any changes, like eating a lot more or a lot less than usual?",
    "How do you feel about yourself these days - like, do you ever feel like you're letting people down or not good enough?",
    "Has it been hard to focus on things lately - reading, conversations, work, anything like that?",
    "Have you noticed yourself moving or talking slower than usual, or maybe feeling really restless and fidgety?",
    "This is important, so take your time - have you had any thoughts lately about not wanting to be here, or hurting yourself?",
    "What's your motivation been like for things you used to care about - studies, hobbies, goals?",
    "How connected do you feel to the people around you these days - friends, family, classmates?",
    "When you think about the next few months, does the future feel hopeful, uncertain, or heavy?",
    "How do you usually handle stress when it builds up - do you talk to someone, push through alone, or avoid it?",
    "Is there anything specific that's been weighing on your mind a lot lately - work, relationships, health, anything?",
    "Last one - overall, if you had to describe how you've been doing in one or two words, what would it be?"
]

def score_response(question, user_answer):
    prompt = (
        "You are a clinical psychologist assistant scoring a mental health screening response.\n\n"
        f"Question asked: {question}\n"
        f"User's answer: {user_answer}\n\n"
        "The user may have responded in English, Hindi, or Hinglish (mix of both).\n"
        "Understand the full meaning and emotional context of the answer, not just keywords.\n\n"
        "Score on a 0-3 scale based on the severity and frequency implied:\n"
        "0 = Clearly positive or no concern at all\n"
        "    Examples: 'I feel great', 'bilkul theek hoon', 'no issues', 'hopeful', 'amazing'\n\n"
        "1 = Mild, vague, or uncertain - slight concern but nothing significant\n"
        "    Examples: 'ok', 'normal', 'thoda alag lag raha hai', 'sometimes', 'not sure'\n\n"
        "2 = Moderate concern - clearly present but manageable\n"
        "    Examples: 'thoda boring lag raha hai', 'work mein focus nahi', 'stressed sometimes',\n"
        "               'sleep thodi kharab hai', 'I push through alone', 'workk'\n\n"
        "3 = Severe concern - clearly serious, frequent, or overwhelming\n"
        "    Examples: 'bahut bura feel hota hai', 'raat ko neend nahi aati', 'bahut thaka rehta hoon',\n"
        "               'kuch achha nahi lagta', 'sab bekar lag raha hai', 'bahut akela hoon'\n\n"
        "Important rules:\n"
        "- Do NOT default everything to 1. Read the actual meaning.\n"
        "- Positive answers like 'hopeful', 'good', 'everything is fine' should score 0.\n"
        "- Negative or effortful answers like 'workk', 'thoda boring', 'push through alone' should score 2.\n"
        "- Clearly distressed answers should score 3.\n\n"
        "Respond with ONLY a single digit: 0, 1, 2, or 3. Nothing else."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        raw = response.choices[0].message.content.strip()
        score = int(raw[0])
        if 0 <= score <= 3:
            return score
    except Exception:
        pass
    return 1


def run_text_phq_assessment():
    print("\n=== Let's start with a few questions ===")
    print("Just answer naturally, in your own words.\n")

    raw_scores = []
    for q in PHQ_QUESTIONS:
        print(f"Counselor: {q}")
        answer = input("You: ").strip()
        score = score_response(q, answer)
        raw_scores.append(score)

    phq9 = raw_scores[:9].copy()

    phq9[0] = round((phq9[0] + raw_scores[9]) / 2)
    phq9[1] = round((phq9[1] + raw_scores[11]) / 2)
    phq9[3] = round((phq9[3] + raw_scores[12]) / 2)
    phq9[5] = round((phq9[5] + raw_scores[10]) / 2)
    phq9[6] = round((phq9[6] + raw_scores[13]) / 2)
    phq9[8] = round((phq9[8] + raw_scores[14]) / 2)

    phq9 = [min(3, max(0, s)) for s in phq9]
    return phq9