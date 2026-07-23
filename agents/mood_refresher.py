import time
import random

session_start_time = None
last_refresh_time = None

REFRESH_INTERVAL = 15 * 60  # 15 minutes in seconds

LIGHT_MOMENTS = [
    "Quick pause — did you know otters hold hands while sleeping so they don't drift apart? Random, but I thought it might bring a tiny smile. Anyway, I'm here with you.",
    "Okay, small detour — apparently a group of flamingos is called a 'flamboyance.' Felt like you needed something light for a second. Let's keep going whenever you're ready.",
    "Just a little breather — I read once that the smell of rain has a name, 'petrichor.' No big reason, just wanted to share something nice before we continue.",
    "Tiny pause here — penguins propose to their partners with a pebble. Sometimes the small things matter most, right? Okay, I'm still with you, take your time.",
]

def start_session():
    global session_start_time, last_refresh_time
    session_start_time = time.time()
    last_refresh_time = time.time()

def should_refresh():
    global last_refresh_time
    if last_refresh_time is None:
        return False
    elapsed = time.time() - last_refresh_time
    if elapsed >= REFRESH_INTERVAL:
        last_refresh_time = time.time()
        return True
    return False

def get_light_moment():
    return random.choice(LIGHT_MOMENTS)