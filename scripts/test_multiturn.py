# scripts/test_multiturn.py
#
# H2 evaluation — multi-turn context-awareness.
#
# Runs the 100 synthetic dialogues through the RAG system in continuous
# sessions (one session_id per dialogue), so ConversationManager accumulates
# context across turns. Then computes:
#
#   - Standard Phase B metrics (F1, MRR, ROUGE-L) via run_full_evaluation
#   - H2 metric: mean answer F1 on requires_context=True turns vs
#     requires_context=False turns. The delta between the two is the
#     evidence for H2 — if context-aware turns score higher, multi-turn
#     memory is helping.
#
# Why F1 and not a proxy "non-empty" check:
#   A model that outputs confident nonsense would score 100% on a
#   "non-empty answer" proxy. F1 against the gold answer is the only
#   meaningful measure of whether context retention actually improves
#   answer quality.
#
# Gemini is used (not MedGemma) because:
#   - H2 tests context-awareness, not domain specificity (that is H3)
#   - Gemini is API-based — no GPU/SLURM required
#   - Using the best-performing generator isolates the context effect
#
# Run directly on Babel login node (no SLURM needed):
#   python scripts/test_multiturn.py

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

DATA_PATH      = "data/BioASQ-training14b/training14b.json"
DIALOGUES_PATH = "data/synthetic/dialogues.json"
OUTPUT_DIR     = "output/evaluation/multiturn"


# ---------------------------------------------------------------------------
# Load system and build/load indices
# index_corpus checks for cached indices automatically (FIX-5) — if the
# full-dataset indices already exist on disk they are loaded, not rebuilt.
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

# Handle both top-level list and {"dialogues": [...]} formats
if isinstance(dialogues, dict):
    dialogues = dialogues.get("dialogues", dialogues.get("data", []))

print(f"[INFO] Loaded {len(dialogues)} dialogues")


# ---------------------------------------------------------------------------
# Run multi-turn inference
# One session_id per dialogue — ConversationManager accumulates context
# across all turns within a dialogue, simulating real clinical inquiry.
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

        gold = turn.get("answer", "")

        pred = system.answer(
            question,
            session_id=session_id,
            gold_answer=gold,
        )

        # Use synthetic ground-truth label — more reliable than the
        # ConversationManager's own anaphora detection for H2 measurement
        pred["requires_context"] = turn.get("requires_context", False)
        predictions.append(pred)

        # Ground truth dict compatible with evaluation.py
        ground_truth.append({
            "id":           question["id"],
            "body":         turn["query"],
            "type":         question_type,
            "ideal_answer": gold,
            "exact_answer": gold,
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
#
# For each turn, compute answer F1 against the gold answer using the
# appropriate metric for that question type. Then take the mean F1
# separately for requires_context=True and requires_context=False turns.
#
# The delta (context_F1 - no_context_F1) is the H2 evidence:
#   - Positive delta → multi-turn context helps answer quality
#   - Near-zero delta → context-awareness has no measurable effect
#   - Negative delta → context is introducing noise (bad ConversationManager)
# ---------------------------------------------------------------------------

def _turn_f1(pred: dict) -> float:
    """
    Compute answer quality F1 for a single prediction vs its gold answer.
    Uses the metric appropriate for each question type:
      - summary  → ROUGE-L
      - list     → token-level F1 (compute_list_f1)
      - factoid  → token-level F1 (compute_list_f1, treating as single-item list)
      - yesno    → exact match (1.0 or 0.0)
    Returns 0.0 if either prediction or gold is empty.
    """
    gold   = pred.get("gold_answer", "")
    answer = pred.get("answer", "")
    qtype  = pred.get("question_type", "summary")

    if not gold or not answer:
        return 0.0

    if qtype == "summary":
        return compute_rouge_l([str(answer)], [str(gold)])

    elif qtype in ("list", "factoid"):
        preds = answer if isinstance(answer, list) else [str(answer)]
        golds = gold   if isinstance(gold,   list) else [str(gold)]
        return compute_list_f1([preds], [golds])

    elif qtype == "yesno":
        return 1.0 if str(answer).strip().lower() == str(gold).strip().lower() else 0.0

    # Fallback for unknown types
    return compute_rouge_l([str(answer)], [str(gold)])


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
print("H2 — Context Retention F1")
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
    "context_turns":       len(context_turns),
    "no_context_turns":    len(no_context_turns),
    "total_turns":         len(predictions),
    "context_f1":          round(context_f1, 4),
    "no_context_f1":       round(no_context_f1, 4),
    "delta":               round(delta, 4),
    "h2_supported":        delta >= 0.10,
    "h2_partial":          0 < delta < 0.10,
}
with open(f"{OUTPUT_DIR}/h2_context_retention.json", "w") as f:
    json.dump(h2_summary, f, indent=2)

print(f"[INFO] H2 summary saved to {OUTPUT_DIR}/h2_context_retention.json")