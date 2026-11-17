import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.intent_router import IntentRouter
r = IntentRouter()
tests = [
    ("What is the procedure for a helmet violation?", "sop"),
    ("Show me the dashboard", "stats"),
    ("Generate a safety report", "report"),
    ("Explain recent incidents", "semantic"),
    ("Show helmet violations", "sql"),
    ("Compare today vs yesterday", "comparison"),
    ("What are the high risk zones?", "risk"),
    ("What is PPE?", "general"),
]
all_ok = True
for q, expected in tests:
    got = r.route(q)
    status = "✅" if got == expected else f"❌ FAIL (got {got})"
    print(f"{status}: \"{q}\" -> {got}")
    if got != expected:
        all_ok = False
print()
print("All routing tests passed!" if all_ok else "Some routing tests FAILED.")
