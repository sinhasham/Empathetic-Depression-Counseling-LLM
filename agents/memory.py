# agents/memory.py

conversation_history = []

def add_user_message(text):
    conversation_history.append({"role": "user",      "content": text})

def add_assistant_message(text):
    conversation_history.append({"role": "assistant", "content": text})

def get_history():
    return conversation_history

def clear_history():
    conversation_history.clear()