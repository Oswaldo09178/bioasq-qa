# Evaluation — Lowami
#
# Orchestrates the full BioASQ evaluation pipeline across both phases:
#
#   run_phase_a_evaluation() — Retrieval: MAP + Snippet F-measure
#   run_phase_b_evaluation() — Generation: routed by question type
#   run_full_evaluation()    — Combined Phase A + B + conversational report
#
# All results are saved to output/evaluation/.
#
# Expected prediction format (produced by rag_system.run_batch()):
# [
#   {
#     "question_id":       str,
#     "question_type":     str,       # yesno | factoid | list | summary
#     "answer":            str|list,  # parsed answer from generation_utils
#     "retrieved_doc_ids": list[str], # ordered doc IDs from retrieval
#     "retrieved_snippets":list[dict],# retrieved snippet dicts
#     "requires_context":  bool,      # from ConversationManager turn flag
#     "latency_s":         float,     # response time in seconds
#   }
# ]
#
# Expected ground truth format (produced by data_utils.parse_question()):
# [
#   {
#     "id":           str,
#     "type":         str,
#     "ideal_answer": str,
#     "documents":    list[str],  # gold document URLs
#     "snippets":     list[dict], # gold snippet dicts
#   }
# ]

import json
import os
from datetime import datetime
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from evaluation_utils import (
    # Phase A
    compute_map,
    compute_snippet_fmeasure,
    # Yes/No
    compute_yesno_accuracy,
    compute_yesno_macro_f1,
    # Factoid
    compute_mrr,
    # List
    compute_list_f1,
    # Summary
    compute_rouge_l,
    llm_as_judge,
    # Conversational
    compute_context_retention_accuracy,
    LatencyTracker,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _build_lookup(ground_truth: list[dict]) -> dict:
    """Index ground truth by question id for O(1) lookup."""
    return {gt["id"]: gt for gt in ground_truth}


def _pmid_from_url(url: str) -> str:
    """Extract PMID string from a BioASQ document URL."""
    return url.rstrip("/").split("/")[-1]


def _save_results(results: dict, output_dir: str, filename: str) -> str:
    """Serialize results dict to JSON and return the saved filepath."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to {filepath}")
    return filepath


def save_per_question_tsv(per_question_rows: list[dict],
                           output_dir: str,
                           retriever: str,
                           generator: str) -> str:
    """
    Save per-question evaluation scores as a TSV file.
    Filename: <retriever>_<generator>_eval.tsv

    Format (one row per question):
        question_id  question_type  map  snippet_fmeasure  macro_f1  accuracy  mrr  list_f1  rouge_l  latency_s
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, f"{retriever}_{generator}_eval.tsv")

    headers = [
        "question_id", "question_type",
        "map", "snippet_fmeasure",
        "macro_f1", "accuracy",
        "mrr", "list_f1", "rouge_l",
        "latency_s",
    ]

    with open(filepath, "w") as f:
        f.write("\t".join(headers) + "\n")
        for row in per_question_rows:
            values = [
                str(row.get("question_id",      "")),
                str(row.get("question_type",     "")),
                str(row.get("map",               "")),
                str(row.get("snippet_fmeasure",  "")),
                str(row.get("macro_f1",          "")),
                str(row.get("accuracy",          "")),
                str(row.get("mrr",               "")),
                str(row.get("list_f1",           "")),
                str(row.get("rouge_l",           "")),
                str(row.get("latency_s",         "")),
            ]
            f.write("\t".join(values) + "\n")

    print(f"[INFO] Per-question eval TSV saved to {filepath}")
    return filepath


def save_aggregated_report(full_results: dict,
                            output_dir: str,
                            retriever: str,
                            generator: str) -> str:
    """
    Save the aggregated evaluation as a human-readable text report.
    Filename: <retriever>_<generator>_report.txt
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, f"{retriever}_{generator}_report.txt")

    pb = full_results.get("phase_b", {})
    pa = full_results.get("phase_a", {})
    cv = full_results.get("conversational", {})
    lt = full_results.get("latency", {})

    lines = []
    lines.append(f"EVALUATION REPORT — retriever={retriever}  generator={generator}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("PHASE A — RETRIEVAL")
    lines.append("=" * 70)
    lines.append(f"  MAP:                   {pa.get('map', 'N/A')}")
    lines.append(f"  Snippet F-measure:     {pa.get('snippet_fmeasure', 'N/A')}")
    lines.append(f"  N questions:           {pa.get('n_questions', 'N/A')}")
    lines.append("")

    lines.append("PHASE B — GENERATION (by question type)")
    lines.append("=" * 70)

    yesno = pb.get("yesno", {})
    if yesno.get("n", 0) > 0:
        lines.append("  YES/NO:")
        lines.append(f"    Macro F1:            {yesno.get('macro_f1', 'N/A')}")
        lines.append(f"    Accuracy:            {yesno.get('accuracy', 'N/A')}")
        lines.append(f"    N:                   {yesno.get('n', 0)}")

    factoid = pb.get("factoid", {})
    if factoid.get("n", 0) > 0:
        lines.append("  FACTOID:")
        lines.append(f"    MRR:                 {factoid.get('mrr', 'N/A')}")
        lines.append(f"    N:                   {factoid.get('n', 0)}")

    lst = pb.get("list", {})
    if lst.get("n", 0) > 0:
        lines.append("  LIST:")
        lines.append(f"    Mean F1:             {lst.get('mean_f1', 'N/A')}")
        lines.append(f"    N:                   {lst.get('n', 0)}")

    summary = pb.get("summary", {})
    if summary.get("n", 0) > 0:
        lines.append("  SUMMARY:")
        lines.append(f"    ROUGE-L:             {summary.get('rouge_l', 'N/A')}")
        lines.append(f"    LLM Judge Score:     {summary.get('mean_judge_score', 'N/A')}")
        lines.append(f"    N:                   {summary.get('n', 0)}")

    lines.append("")
    lines.append("CONVERSATIONAL")
    lines.append("=" * 70)
    lines.append(f"  Context Retention Acc: {cv.get('context_retention_accuracy', 'N/A')}")
    lines.append(f"  N context turns:       {cv.get('n_context_turns', 'N/A')}")
    lines.append("")

    lines.append("LATENCY")
    lines.append("=" * 70)
    lines.append(f"  Mean latency (s):      {lt.get('mean_latency_s', 'N/A')}")
    lines.append(f"  Min latency (s):       {lt.get('min_latency_s', 'N/A')}")
    lines.append(f"  Max latency (s):       {lt.get('max_latency_s', 'N/A')}")
    lines.append(f"  Throughput (q/s):      {lt.get('throughput_qps', 'N/A')}")
    lines.append(f"  N timed:               {lt.get('n_timed', 'N/A')}")
    lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Aggregated report saved to {filepath}")
    return filepath


def _print_report(title: str, metrics: dict) -> None:
    """Pretty-print a metrics dict to stdout."""
    width = 42
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<30} {val:.4f}")
        elif isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    {k:<28} {v}")
        else:
            print(f"  {key:<30} {val}")
    print(f"{'─' * width}\n")


# ===========================================================================
# Phase A — Retrieval Evaluation
# ===========================================================================

def run_phase_a_evaluation(predictions: list[dict],
                            ground_truth: list[dict],
                            output_dir: str = "output/evaluation/") -> dict:
    """
    Evaluate retrieval performance across all predictions.

    Metrics (proposal Phase A):
      - MAP:              Mean Average Precision over retrieved document IDs.
      - Snippet F-measure: Character-level F-measure over retrieved snippets.

    Args:
        predictions:  List of prediction dicts (see module docstring).
        ground_truth: List of ground truth dicts (see module docstring).
        output_dir:   Directory to save results JSON.

    Returns:
        {
            "map":                float,
            "snippet_fmeasure":   float,
            "n_questions":        int,
        }
    """
    gt_lookup = _build_lookup(ground_truth)

    retrieved_doc_ids_all  = []
    gold_doc_ids_all       = []
    snippet_fmeasures      = []

    for pred in predictions:
        qid = pred.get("question_id", "")
        gt  = gt_lookup.get(qid)
        if not gt:
            print(f"[WARNING] No ground truth found for id '{qid}' — skipping.")
            continue

        # --- MAP: compare retrieved doc IDs vs gold document URLs (as PMIDs) ---
        # Chunk IDs are stored as "pubmed_{pmid}_{begin}_{end}" — strip to bare PMID
        raw_ids       = pred.get("retrieved_doc_ids", [])
        retrieved_ids = [d.split("_")[1] if d.startswith("pubmed_") else d for d in raw_ids]
        gold_ids      = [_pmid_from_url(url) for url in gt.get("documents", [])]

        retrieved_doc_ids_all.append(retrieved_ids)
        gold_doc_ids_all.append(gold_ids)

        # --- Snippet F-measure ---
        retrieved_snippets = pred.get("retrieved_snippets", [])
        gold_snippets      = gt.get("snippets", [])
        snippet_fmeasures.append(
            compute_snippet_fmeasure(retrieved_snippets, gold_snippets)
        )

    map_score      = compute_map(retrieved_doc_ids_all, gold_doc_ids_all)
    mean_snippet_f = round(
        sum(snippet_fmeasures) / len(snippet_fmeasures), 4
    ) if snippet_fmeasures else 0.0

    results = {
        "phase":            "A",
        "map":              map_score,
        "snippet_fmeasure": mean_snippet_f,
        "n_questions":      len(snippet_fmeasures),
    }

    _print_report("Phase A — Retrieval Results", results)
    _save_results(results, output_dir, "phase_a_results.json")
    return results


# ===========================================================================
# Phase B — Generation Evaluation
# ===========================================================================

def run_phase_b_evaluation(predictions: list[dict],
                            ground_truth: list[dict],
                            judge_model=None,
                            output_dir: str = "output/evaluation/") -> dict:
    """
    Evaluate generation performance, routing metrics by question type.

    Metrics (proposal Phase B):
      - yesno:   Macro F1 + Accuracy
      - factoid: MRR
      - list:    Mean F1
      - summary: ROUGE-L + LLM-as-judge (if judge_model provided)

    Args:
        predictions:  List of prediction dicts (see module docstring).
        ground_truth: List of ground truth dicts (see module docstring).
        judge_model:  Optional loaded Gemini model for LLM-as-judge on summaries.
                      If None, judge scores are skipped.
        output_dir:   Directory to save results JSON.

    Returns:
        {
            "phase": "B",
            "yesno":   {"macro_f1", "accuracy", "n"},
            "factoid": {"mrr", "n"},
            "list":    {"mean_f1", "n"},
            "summary": {"rouge_l", "mean_judge_score", "n"},
        }
    """
    gt_lookup = _build_lookup(ground_truth)

    # Buckets per question type
    yesno_preds,   yesno_refs            = [], []
    factoid_preds, factoid_refs          = [], []
    list_preds,    list_refs             = [], []
    summary_preds, summary_refs          = [], []
    summary_judge_scores                 = []
    summary_questions, summary_snippets  = [], []

    per_question_rows = []  # one row per question for TSV output

    for pred in predictions:
        qid   = pred.get("question_id", "")
        qtype = pred.get("question_type", "").lower()
        gt    = gt_lookup.get(qid)
        if not gt:
            continue

        answer = pred.get("answer", "")

        # Use exact_answer (structured) for factoid/list/yesno — it is a list of strings
        # or nested list of strings e.g. [["RET"], ["GDNF"]] for factoid.
        # Use ideal_answer (free-text) for summary ROUGE-L.
        if qtype in ("factoid", "list", "yesno"):
            raw_exact = gt.get("exact_answer")
            if raw_exact is None:
                # Fall back to ideal_answer if exact_answer is missing
                gold = gt.get("ideal_answer", "")
            else:
                gold = raw_exact
        else:
            gold = gt.get("ideal_answer", "")

        # Flatten nested lists: BioASQ factoid exact_answer is [["answer1"], ["answer2"]]
        if isinstance(gold, list) and gold and isinstance(gold[0], list):
            gold = [item for sublist in gold for item in sublist]

        if qtype == "yesno":
            # yesno exact_answer is ["yes"] or ["no"]
            gold_str = gold[0].strip().lower() if isinstance(gold, list) and gold else str(gold).lower().strip()
            yesno_preds.append(str(answer).lower().strip())
            yesno_refs.append(gold_str)
            per_question_rows.append({
                "question_id": qid, "question_type": qtype,
                "macro_f1": int(str(answer).lower().strip() == gold_str),
                "accuracy": int(str(answer).lower().strip() == gold_str),
                "map": "", "snippet_fmeasure": "", "mrr": "", "list_f1": "", "rouge_l": "",
                "latency_s": pred.get("latency_s", ""),
            })

        elif qtype == "factoid":
            candidates = answer if isinstance(answer, list) else [str(answer)]
            gold_list  = gold   if isinstance(gold,   list) else [str(gold)]
            factoid_preds.append(candidates)
            factoid_refs.append(gold_list)
            rr = 0.0
            gold_set = {g.strip().lower() for g in gold_list}
            for rank, c in enumerate(candidates, 1):
                if c.strip().lower() in gold_set:
                    rr = 1.0 / rank
                    break
            per_question_rows.append({
                "question_id": qid, "question_type": qtype,
                "mrr": round(rr, 4),
                "map": "", "snippet_fmeasure": "", "macro_f1": "", "accuracy": "", "list_f1": "", "rouge_l": "",
                "latency_s": pred.get("latency_s", ""),
            })

        elif qtype == "list":
            pred_list = answer if isinstance(answer, list) else [str(answer)]
            gold_list = gold   if isinstance(gold,   list) else [str(gold)]
            list_preds.append(pred_list)
            list_refs.append(gold_list)
            ps = {p.strip().lower() for p in pred_list}
            gs = {g.strip().lower() for g in gold_list}
            tp = len(ps & gs)
            prec = tp / len(ps) if ps else 0.0
            rec  = tp / len(gs) if gs else 0.0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
            per_question_rows.append({
                "question_id": qid, "question_type": qtype,
                "list_f1": round(f1, 4),
                "map": "", "snippet_fmeasure": "", "macro_f1": "", "accuracy": "", "mrr": "", "rouge_l": "",
                "latency_s": pred.get("latency_s", ""),
            })

        elif qtype == "summary":
            summary_preds.append(str(answer))
            summary_refs.append(str(gold))
            summary_questions.append(gt.get("body", ""))
            summary_snippets.append(pred.get("retrieved_snippets", []))
            from rouge_score import rouge_scorer as _rs
            _scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
            _rl = round(_scorer.score(str(gold), str(answer))["rougeL"].fmeasure, 4)
            per_question_rows.append({
                "question_id": qid, "question_type": qtype,
                "rouge_l": _rl,
                "map": "", "snippet_fmeasure": "", "macro_f1": "", "accuracy": "", "mrr": "", "list_f1": "",
                "latency_s": pred.get("latency_s", ""),
            })

    # --- Yes/No metrics ---
    yesno_results = {
        "macro_f1": compute_yesno_macro_f1(yesno_preds, yesno_refs),
        "accuracy": compute_yesno_accuracy(yesno_preds, yesno_refs),
        "n":        len(yesno_preds),
    }

    # --- Factoid metrics ---
    factoid_results = {
        "mrr": compute_mrr(factoid_preds, factoid_refs),
        "n":   len(factoid_preds),
    }

    # --- List metrics ---
    list_results = {
        "mean_f1": compute_list_f1(list_preds, list_refs),
        "n":       len(list_preds),
    }

    # --- Summary metrics ---
    rouge_l = compute_rouge_l(summary_preds, summary_refs)

    mean_judge_score = None
    if judge_model and summary_preds:
        for question, answer, snippets in zip(
            summary_questions, summary_preds, summary_snippets
        ):
            result = llm_as_judge(question, answer, snippets, judge_model)
            summary_judge_scores.append(result.get("factuality_score", 0))

        mean_judge_score = round(
            sum(summary_judge_scores) / len(summary_judge_scores), 4
        ) if summary_judge_scores else None

    summary_results = {
        "rouge_l":          rouge_l,
        "mean_judge_score": mean_judge_score,
        "n":                len(summary_preds),
    }

    results = {
        "phase":   "B",
        "yesno":   yesno_results,
        "factoid": factoid_results,
        "list":    list_results,
        "summary": summary_results,
    }

    _print_report("Phase B — Generation Results", results)
    _save_results(results, output_dir, "phase_b_results.json")
    results["_per_question_rows"] = per_question_rows
    return results


# ===========================================================================
# Full Evaluation
# ===========================================================================

def run_full_evaluation(predictions: list[dict],
                         ground_truth: list[dict],
                         judge_model=None,
                         output_dir: str = "output/evaluation/",
                         retriever: str = "retriever",
                         generator: str = "generator") -> dict:
    """
    Run Phase A + Phase B + conversational metrics in one call.
    Saves a combined report to output/evaluation/full_results.json.

    Args:
        predictions:  List of prediction dicts (see module docstring).
        ground_truth: List of ground truth dicts (see module docstring).
        judge_model:  Optional Gemini model for LLM-as-judge on summaries.
        output_dir:   Directory to save all results.

    Returns:
        Combined results dict containing phase_a, phase_b,
        conversational, and latency sections.
    """
    print(f"\n[INFO] Starting full evaluation — {len(predictions)} predictions")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Phase A
    phase_a = run_phase_a_evaluation(predictions, ground_truth, output_dir)

    # Phase B
    phase_b = run_phase_b_evaluation(
        predictions, ground_truth, judge_model, output_dir
    )

    # Conversational
    context_acc = compute_context_retention_accuracy(predictions)
    conversational_results = {
        "context_retention_accuracy": context_acc,
        "n_context_turns": sum(
            1 for p in predictions if p.get("requires_context", False)
        ),
    }
    _print_report("Conversational Results", conversational_results)

    # Latency — aggregate from per-prediction latency_s field
    latencies = [p["latency_s"] for p in predictions if "latency_s" in p]
    if latencies:
        latency_results = {
            "mean_latency_s":  round(sum(latencies) / len(latencies), 4),
            "min_latency_s":   round(min(latencies), 4),
            "max_latency_s":   round(max(latencies), 4),
            "throughput_qps":  round(len(latencies) / sum(latencies), 4),
            "n_timed":         len(latencies),
        }
    else:
        latency_results = {"note": "No latency data in predictions."}
    _print_report("Latency Results", latency_results)

    # Combined report
    full_results = {
        "timestamp":       timestamp,
        "n_predictions":   len(predictions),
        "phase_a":         phase_a,
        "phase_b":         phase_b,
        "conversational":  conversational_results,
        "latency":         latency_results,
    }

    _save_results(
        full_results, output_dir,
        f"full_results_{timestamp}.json"
    )

    # Per-question TSV
    per_question_rows = phase_b.pop("_per_question_rows", [])
    if per_question_rows:
        save_per_question_tsv(per_question_rows, output_dir, retriever, generator)

    # Aggregated text report
    save_aggregated_report(full_results, output_dir, retriever, generator)

    return full_results