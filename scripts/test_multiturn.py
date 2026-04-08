# scripts/test_multiturn.py
#
# H2 evaluation — multi-turn context-awareness.
#
# Runs the 100 synthetic dialogues through the RAG system in continuous
# sessions (one session_id per dialogue), so ConversationManager accumulates
# context across turns.
#
# Key fix vs previous version:
#   Gold answers from synthetic dialogues are free-text strings.
#   evaluation.py expects BioASQ structured format:
#     - yesno   : "yes" | "no"
#     - factoid : [["answer"]]  (nested list)
#     - list    : [["item1", "item2", ...]]  (nested list)
#     - summary : str
#   parse_gold_answer() converts free-text to the correct format per type.
#   Without this, yesno/factoid/list F1 scores are near-zero due to format
#   mismatch, not actual model failure.
#
# H2 metric:
#   _turn_f1() computes per-turn answer quality using the appropriate metric
#   for each question type. The delta between requires_context=True and
#   requires_context=False mean F1 is the H2 evidence.

import json
import sys
from pathlib import Path

sys.path.append("src/utils")
sys.path.append("src")

from rag_system import BioASQRAGSystem
from evaluation import run_full_evaluation
from data_utils import load_bioasq_dataset, parse_question
from evaluation_utils import compute_list_f1, compute_rouge_l


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH      = "data/BioASQ-training14b/trainining14b.json"
DIALOGUES_PATH = "data/synthetic/dialogues.json"
OUTPUT_DIR     = "output/evaluation/multiturn"


# ---------------------------------------------------------------------------
# Gold answer parser
# Converts free-text synthetic answers into BioASQ structured format
# so evaluation.py computes meaningful metrics.
# ---------------------------------------------------------------------------

def parse_gold_answer(text: str, qtype: str):
    """
    Convert a free-text synthetic gold answer into the format evaluation.py
    expects for each question type.

    yesno   → "yes" or "no" (scans text for first occurrence)
    factoid → [["answer"]]  (single-candidate nested list)
    list    → [["item1", "item2", ...]]  (split on comma/semicolon)
    summary → str  (unchanged)
    """
    if not text:
        return "" if qtype == "summary" else []

    text = text.strip()

    if qtype == "yesno":
        lower = text.lower()
        # Check for explicit yes/no at start of answer
        if lower.startswith("yes"):
            return "yes"
        if lower.startswith("no"):
            return "no"
        # Scan for standalone yes/no
        for word in lower.split():
            w = word.strip(".,;:")
            if w in ("yes", "no"):
                return w
        # Default: treat absence of "no" as yes (yesno questions in BioASQ
        # skew heavily positive)
        return "yes"

    elif qtype == "factoid":
        # Wrap as single-candidate nested list
        return [[text]]

    elif qtype == "list":
        # Split on common delimiters — synthetic answers often use comma or
        # semicolon separated items
        import re
        items = re.split(r"[,;]|\band\b", text)
        items = [i.strip() for i in items if i.strip()]
        if not items:
            items = [text]
        return [items]

    else:  # summary
        return text


# ---------------------------------------------------------------------------
# Per-turn F1 computation
# Uses the appropriate metric for each question type.
# ---------------------------------------------------------------------------

def _turn_f1(pred: dict) -> float:
    """
    Compute answer quality F1 for a single prediction vs its gold answer.

    summary  → ROUGE-L
    list     → token-level list F1
    factoid  → token-level list F1 (treating as single-item list)
    yesno    → exact match (1.0 or 0.0)

    Returns 0.0 if either prediction or gold is missing.
    """
    gold   = pred.get("gold_answer", "")
    answer = pred.get("answer", "")
    qtype  = pred.get("question_type", "summary")

    if not gold or not answer:
        return 0.0

    if qtype == "summary":
        return compute_rouge_l([str(answer)], [str(gold)])

    elif qtype in ("list", "factoid"):
        # Flatten nested list format for comparison
        if isinstance(gold, list):
            gold_flat = gold[0] if gold and isinstance(gold[0], list) else gold
        else:
            gold_flat = [str(gold)]
        preds = answer if isinstance(answer, list) else [str(answer)]
        return compute_list_f1([preds], [gold_flat])

    elif qtype == "yesno":
        pred_str = str(answer).strip().lower()
        gold_str = str(gold).strip().lower()
        return 1.0 if pred_str == gold_str else 0.0

    return compute_rouge_l([str(answer)], [str(gold)])


# ---------------------------------------------------------------------------
# Load system and indices
# ---------------------------------------------------------------------------

system = BioASQRAGSystem(retriever="hybrid", generator="gemini", k=5)
system.load_generator()

print(f"[INFO] Loading dataset for index building: {DATA_PATH}")
raw_questions = load_bioasq_dataset(DATA_PATH)
questions     = [parse_question(q) for q in raw_questions]
system.index_corpus(questions)


# ---------------------------------------------------------------------------
# Load synthetic dialogues
# ---------------------------------------------------------------------------

print(f"[INFO] Loading synthetic dialogues from {DIALOGUES_PATH}")
with open(DIALOGUES_PATH) as f:
    dialogues = json.load(f)

if isinstance(dialogues, dict):
    dialogues = dialogues.get("dialogues", dialogues.get("data", []))

print(f"[INFO] Loaded {len(dialogues)} dialogues")


# ---------------------------------------------------------------------------
# Multi-turn inference
# ---------------------------------------------------------------------------

predictions  = []
ground_truth = []

for d_idx, dialogue in enumerate(dialogues, 1):
    session_id    = dialogue.get("source_id", f"dialogue_{d_idx}")
    question_type = dialogue.get("question_type", "summary")
    snippets      = dialogue.get("snippets", [])
    turns         = dialogue.get("turns", [])

    print(
        f"[INFO] Dialogue {d_idx}/{len(dialogues)} — "
        f"id={session_id}, type={question_type}, turns={len(turns)}"
    )

    for turn in turns:
        question = {
            "id":       f"{session_id}_turn{turn['turn_id']}",
            "body":     turn["query"],
            "type":     question_type,
            "snippets": snippets,
        }

        raw_gold     = turn.get("answer", "")
        structured   = parse_gold_answer(raw_gold, question_type)

        pred = system.answer(
            question,
            session_id=session_id,
            gold_answer=structured,
        )

        pred["requires_context"] = turn.get("requires_context", False)
        predictions.append(pred)

        # Ground truth in BioASQ-compatible format
        ground_truth.append({
            "id":           question["id"],
            "body":         turn["query"],
            "type":         question_type,
            "ideal_answer": raw_gold,       # free-text for ROUGE-L
            "exact_answer": structured,     # structured for F1/MRR
            "snippets":     snippets,
        })


# ---------------------------------------------------------------------------
# Standard Phase B evaluation
# ---------------------------------------------------------------------------

print(f"\n[INFO] Running Phase B evaluation on {len(predictions)} predictions...")
run_full_evaluation(
    predictions,
    ground_truth,
    judge_model=None,
    output_dir=OUTPUT_DIR,
    retriever="hybrid",
    generator="gemini_multiturn",
)


# ---------------------------------------------------------------------------
# H2 metric — per-turn F1 split by requires_context
# ---------------------------------------------------------------------------

context_turns    = [p for p in predictions if p.get("requires_context", False)]
no_context_turns = [p for p in predictions if not p.get("requires_context", False)]

context_f1    = (
    sum(_turn_f1(p) for p in context_turns) / len(context_turns)
    if context_turns else 0.0
)
no_context_f1 = (
    sum(_turn_f1(p) for p in no_context_turns) / len(no_context_turns)
    if no_context_turns else 0.0
)
delta = context_f1 - no_context_f1

print("\n" + "=" * 60)
print("H2 — Context Retention F1 (strict evaluation)")
print("=" * 60)
print(f"  requires_context=True  turns : {len(context_turns):4d}  |  mean F1 = {context_f1:.3f}")
print(f"  requires_context=False turns : {len(no_context_turns):4d}  |  mean F1 = {no_context_f1:.3f}")
print(f"  Delta (context - no_context) : {delta:+.3f}")
if delta >= 0.10:
    print(f"  → H2 SUPPORTED: ≥10% F1 improvement on context-dependent turns")
elif delta > 0:
    print(f"  → H2 PARTIALLY SUPPORTED: positive but below 10% threshold")
else:
    print(f"  → H2 NOT SUPPORTED: no F1 improvement from context-awareness")
print("=" * 60)

# Save H2 summary
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
h2_summary = {
    "context_turns":      len(context_turns),
    "no_context_turns":   len(no_context_turns),
    "total_turns":        len(predictions),
    "context_f1":         round(context_f1, 4),
    "no_context_f1":      round(no_context_f1, 4),
    "delta":              round(delta, 4),
    "h2_supported":       delta >= 0.10,
    "h2_partial":         0 < delta < 0.10,
}
with open(f"{OUTPUT_DIR}/h2_context_retention.json", "w") as f:
    json.dump(h2_summary, f, indent=2)

print(f"[INFO] H2 summary saved to {OUTPUT_DIR}/h2_context_retention.json")