import sys
sys.path.append("/mnt/user-data/outputs")
from conversation_manager import ConversationManager

# Suppress internal debug prints
import unittest.mock as mock

manager = ConversationManager(window_size=5, memory_strategy="sliding_window")

turns = [
    ("user",      "What genes are involved in Hirschsprung disease?"),
    ("assistant", "Key genes include RET, GDNF, EDNRB, EDN3, and SOX10. RET mutations account for approximately half of familial cases."),
    ("user",      "What mutations are associated with that gene?"),
    ("assistant", "RET mutations include both coding sequence variants and non-coding variations, contributing to long-segment and syndromic forms of HSCR."),
    ("user",      "Is it dominant or recessive?"),
]

CYAN  = "\033[96m"
GREEN = "\033[92m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"

print()
print(f"{BOLD}{'─'*62}{RESET}")
print(f"{BOLD}  ConversationManager — Multi-Turn Clinical Dialogue Demo{RESET}")
print(f"{BOLD}{'─'*62}{RESET}")

for role, content in turns:
    with mock.patch("builtins.print"):  # suppress internal logs
        manager.add_turn(role, content)

    label = f"{CYAN}USER{RESET}" if role == "user" else f"{GREEN}ASSISTANT{RESET}"
    turn  = manager.get_full_history()[-1]

    print(f"\n  {label}  {DIM}turn {turn['turn_id']}{RESET}")
    print(f"  {content}")

    if role == "user":
        with mock.patch("builtins.print"):
            resolved = manager.resolve_anaphora(content)
            ctx_query = manager.build_contextualized_query(content)
        flag = turn["requires_context"]

        print(f"\n  {DIM}anaphora resolved →{RESET}  {resolved}")
        print(f"  {DIM}retrieval query   →{RESET}  {ctx_query}")
        print(f"  {DIM}requires_context  →{RESET}  {BOLD}{flag}{RESET}")

print()
print(f"{BOLD}{'─'*62}{RESET}")
print(f"{BOLD}  Context Window (passed to LLM){RESET}")
print(f"{BOLD}{'─'*62}{RESET}")
for t in manager.get_context_window():
    label = f"{CYAN}user{RESET}" if t["role"] == "user" else f"{GREEN}assistant{RESET}"
    print(f"\n  [{label}]  {t['content']}")
print()