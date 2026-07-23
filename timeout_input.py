import threading
import random

FUN_NUDGES = [
    "Koi jaldi nahi hai yaar, jitna time chahiye lo. Btw ek random sawaal — abhi tak ka favorite movie kaunsa hai?",
    "Sab theek? Bas chill karo, jab ready ho tab batao. Ek halka sawaal — koi favorite game hai jo tum aksar khelte ho?",
    "Lo ek chhota joke ho jaaye jab tak tum soch rahe ho — Teacher: 'Tum late kyun aaye?' Student: 'Sir, board pe likha tha SLOW' 😄 Anyway, jab ready ho batao.",
    "Itna sochne ki zaroorat nahi, jaise feel ho waise bolo. Suna hai Mumbai ke dabbawale bina kisi app ke perfect delivery karte hain — random fact, par sahi hai.",
    "Take your time bilkul. Acha bata, koi favorite cricket team ya player hai?",
]

CHECKIN_NUDGES = [
    "Sab theek hai na? Bas check kar raha hoon, koi pressure nahi hai.",
    "Thoda time ho gaya quiet rehte — sab okay hai? Jaldi nahi hai batane ki.",
    "Bas ek baar dekh liya, sab fine hai na tumhari taraf se?",
]

def get_input_with_timeout(prompt, first_timeout=20, second_timeout=20):
    """
    Waits for user input. If no input arrives within `first_timeout` seconds,
    prints a light/fun Hinglish nudge. If still no input after
    `second_timeout` more seconds, prints a gentle Hinglish check-in message.
    Keeps waiting for the actual answer the whole time — nudges are just
    printed while waiting, they don't discard or replace whatever the user
    eventually types.
    """
    result = {"value": None}

    def collect():
        result["value"] = input(prompt)

    thread = threading.Thread(target=collect)
    thread.daemon = True
    thread.start()

    thread.join(timeout=first_timeout)
    if thread.is_alive():
        print(f"\nCounselor: {random.choice(FUN_NUDGES)}\n")

        thread.join(timeout=second_timeout)
        if thread.is_alive():
            print(f"\nCounselor: {random.choice(CHECKIN_NUDGES)}\n")
            thread.join()  # keep waiting indefinitely for the real answer

    return result["value"].strip() if result["value"] else ""