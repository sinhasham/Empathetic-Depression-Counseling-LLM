# agents/router.py

def route_agent(lmh):
    if lmh == "Low":    return "Agent_L"
    elif lmh == "Medium": return "Agent_M"
    else:               return "Agent_H"