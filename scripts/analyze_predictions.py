"""
Success & Error Analysis Script
Extracts best and worst predictions per question type for presentation slides.

Run from project root:
    python scripts/analyze_predictions.py --retriever hybrid --generator gemini
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "utils"))

from data_utils import load_bioasq_dataset, parse_question


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever",   type=str, default="hybrid")
    parser.add_argument("--generator",   type=str, default="gemini")
    parser.add_argument("--data",        type=str,
                        default=str(PROJECT_ROOT / "data" / "BioASQ-training14b" / "trainining14b.json"))
    parser.add_argument("--output_dir",  type=str,
                        default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--n_examples",  type=int, default=3,
                        help="Number of success/failure examples per question type")
    parser.add_argument("--show_slowest", action="store_true",
                        help="Show the slowest questions regardless of type")
    return parser.parse_args()


def load_eval_tsv(filepath: str) -> list[dict]:
    """Load per-question eval TSV into list of dicts."""
    rows = []
    with open(filepath) as f:
        headers = f.readline().strip().split("\t")
        for line in f:
            values = line.strip().split("\t")
            row = dict(zip(headers, values))
            rows.append(row)
    return rows


def load_predictions_tsv(filepath: str) -> list[str]:
    """Load raw prediction TSV — one answer per line."""
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()

    eval_path = Path(args.output_dir) / "evaluation" / f"{args.retriever}_{args.generator}_eval.tsv"
    pred_path = Path(args.output_dir) / "prediction"  / f"{args.retriever}_{args.generator}.tsv"

    if not eval_path.exists():
        print(f"[ERROR] Eval file not found: {eval_path}")
        return
    if not pred_path.exists():
        print(f"[ERROR] Prediction file not found: {pred_path}")
        return

    # Load eval rows
    eval_rows = load_eval_tsv(str(eval_path))

    # Load ground truth indexed by question id
    print(f"[INFO] Loading dataset from {args.data}")
    raw_qs = load_bioasq_dataset(args.data)
    gt_lookup = {}
    for q in raw_qs:
        parsed = parse_question(q)
        gt_lookup[parsed["id"]] = parsed
    raw_lookup = {q["id"]: q for q in raw_qs}

    # Build prediction lookup: question_id → answer
    # The prediction TSV has one answer per line in the same order as run_batch.
    # We rebuild the mapping via the eval TSV question_id column.
    predictions_list = load_predictions_tsv(str(pred_path))
    pred_lookup = {}
    for i, row in enumerate(eval_rows):
        qid = row.get("question_id", "")
        if qid and i < len(predictions_list):
            pred_lookup[qid] = predictions_list[i]

    # Match eval rows with predictions and ground truth
    records = []
    for row in eval_rows:
        qid   = row.get("question_id", "")
        qtype = row.get("question_type", "")
        gt    = gt_lookup.get(qid, {})
        raw   = raw_lookup.get(qid, {})

        prediction = pred_lookup.get(qid, "")

        # Get the right gold answer per type
        if qtype in ("factoid", "list", "yesno"):
            gold = raw.get("exact_answer", gt.get("ideal_answer", ""))
        else:
            gold = gt.get("ideal_answer", "")

        # Recompute scores from scratch using prediction vs gold
        # (don't trust TSV scores — they may have index drift issues)
        score = None
        pred_str = str(prediction).lower().strip()

        if qtype == "yesno":
            gold_str = gold[0].lower().strip() if isinstance(gold, list) and gold else str(gold).lower().strip()
            score = 1.0 if pred_str == gold_str else 0.0

        elif qtype == "factoid":
            candidates = [p.strip().lower() for p in prediction.split("|")] if "|" in prediction else [pred_str]
            gold_list  = gold if isinstance(gold, list) else [str(gold)]
            # Flatten nested lists
            if gold_list and isinstance(gold_list[0], list):
                gold_list = [item for sub in gold_list for item in sub]
            gold_set = {g.strip().lower() for g in gold_list}

            def _fuzzy_match(candidate, gold_set):
                """Match if candidate contains or is contained in any gold string."""
                c = candidate.strip().lower()
                for g in gold_set:
                    g = g.strip().lower()
                    if c == g or c in g or g in c:
                        return True
                    # Strip punctuation and compare
                    import re
                    c_clean = re.sub(r"[^a-z0-9]", "", c)
                    g_clean = re.sub(r"[^a-z0-9]", "", g)
                    if c_clean and g_clean and (c_clean == g_clean or c_clean in g_clean or g_clean in c_clean):
                        return True
                return False

            score = 0.0
            for rank, c in enumerate(candidates, 1):
                if _fuzzy_match(c, gold_set):
                    score = 1.0 / rank
                    break

        elif qtype == "list":
            pred_items = [p.strip().lower() for p in prediction.split("|")] if "|" in prediction else [pred_str]
            gold_list  = gold if isinstance(gold, list) else [str(gold)]
            if gold_list and isinstance(gold_list[0], list):
                gold_list = [item for sub in gold_list for item in sub]
            ps = {p for p in pred_items if p}
            gs = {g.strip().lower() for g in gold_list if g}
            if ps and gs:
                tp    = len(ps & gs)
                prec  = tp / len(ps)
                rec   = tp / len(gs)
                score = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
            else:
                score = 0.0

        elif qtype == "summary":
            try:
                from rouge_score import rouge_scorer as _rs
                _scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
                gold_str = gold[0] if isinstance(gold, list) and gold else str(gold)
                score = round(_scorer.score(gold_str, prediction)["rougeL"].fmeasure, 4)
            except Exception:
                score = 0.0

        if score is None:
            continue

        records.append({
            "question_id":   qid,
            "question_type": qtype,
            "question":      gt.get("body", ""),
            "prediction":    prediction,
            "gold":          gold,
            "score":         score,
            "latency_s":     float(row.get("latency_s", 0) or 0),
        })

    # Sort by score and extract top/bottom per type
    qtypes = ["yesno", "factoid", "list", "summary"]

    print("\n" + "=" * 70)
    print(f"  ANALYSIS: {args.retriever} + {args.generator}  ({len(records)} questions)")
    print("=" * 70)

    all_examples = {"successes": {}, "failures": {}}

    for qtype in qtypes:
        type_records = [r for r in records if r["question_type"] == qtype]
        if not type_records:
            continue

        sorted_records = sorted(type_records, key=lambda x: x["score"], reverse=True)
        successes = sorted_records[:args.n_examples]
        failures  = sorted_records[-args.n_examples:]

        all_examples["successes"][qtype] = successes
        all_examples["failures"][qtype]  = failures

        metric_name = {
            "yesno":   "Accuracy",
            "factoid": "MRR",
            "list":    "F1",
            "summary": "ROUGE-L",
        }[qtype]

        print(f"\n{'─'*70}")
        print(f"  {qtype.upper()}  (metric: {metric_name}, n={len(type_records)})")
        print(f"{'─'*70}")

        print(f"\n  ✅ SUCCESSES (score = 1.0)")
        for r in successes:
            print(f"\n    Q:    {r['question']}")
            print(f"    Pred: {r['prediction'][:120]}")
            print(f"    Gold: {str(r['gold'])[:120]}")
            print(f"    Score: {r['score']}  |  Latency: {r['latency_s']:.2f}s")

        print(f"\n  ❌ FAILURES (score = 0.0 or lowest)")
        for r in failures:
            print(f"\n    Q:    {r['question']}")
            print(f"    Pred: {r['prediction'][:120]}")
            print(f"    Gold: {str(r['gold'])[:120]}")
            print(f"    Score: {r['score']}  |  Latency: {r['latency_s']:.2f}s")

    # --- Slowest questions ---
    if args.show_slowest:
        slowest = sorted(records, key=lambda x: x["latency_s"], reverse=True)[:args.n_examples]
        print(f"\n{'─'*70}")
        print(f"  SLOWEST QUESTIONS (top {args.n_examples})")
        print(f"{'─'*70}")
        for r in slowest:
            print(f"\n    Type:     {r['question_type']}")
            print(f"    Q:        {r['question']}")
            print(f"    Pred:     {r['prediction'][:150]}")
            print(f"    Gold:     {str(r['gold'])[:150]}")
            print(f"    Score:    {round(r['score'], 4)}")
            print(f"    Latency:  {r['latency_s']:.2f}s")

    # Save to JSON for further use
    out_path = Path(args.output_dir) / "evaluation" / f"{args.retriever}_{args.generator}_examples.json"
    with open(out_path, "w") as f:
        json.dump(all_examples, f, indent=2, default=str)
    print(f"\n[INFO] Examples saved to {out_path}")


if __name__ == "__main__":
    main()