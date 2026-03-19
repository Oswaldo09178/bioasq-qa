# Evaluation Utilities — Lowami
#
# Strictly implements the metrics defined in the proposal:
#
# Phase A — Retrieval:
#   - Mean Average Precision (MAP)
#   - Snippet F-measure (character-level)
#
# Phase B — Generation:
#   - Yes/No  → Macro F1 + Accuracy
#   - Factoid → Mean Reciprocal Rank (MRR)
#   - List    → Mean F1
#   - Summary → ROUGE-L + LLM-as-judge
#
# Conversational:
#   - Context Retention Accuracy
#   - Throughput / Latency

import time
from rouge_score import rouge_scorer


# ===========================================================================
# Phase A — Retrieval
# ===========================================================================

def compute_map(retrieved_docs: list[list[str]],
                gold_docs: list[list[str]]) -> float:
    """
    Mean Average Precision (MAP) for document retrieval — Phase A primary metric.

    For each query, computes Average Precision (AP) over the ranked list
    of retrieved document IDs, then averages across all queries.

    Args:
        retrieved_docs: List of retrieved doc ID lists, one per query.
                        Order matters — earlier = higher ranked.
                        e.g. [["pubmed_123", "pubmed_456"], ["pubmed_789"]]
        gold_docs:      List of ground truth doc ID lists, one per query.
                        e.g. [["pubmed_123"], ["pubmed_789", "pubmed_111"]]

    Returns:
        MAP score (float, 0–1).
    """
    if not retrieved_docs or not gold_docs:
        return 0.0

    average_precisions = []

    for retrieved, gold in zip(retrieved_docs, gold_docs):
        gold_set = set(gold)
        if not gold_set:
            continue

        hits        = 0
        sum_prec    = 0.0

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in gold_set:
                hits      += 1
                sum_prec  += hits / rank

        ap = sum_prec / len(gold_set)
        average_precisions.append(ap)

    return round(sum(average_precisions) / len(average_precisions), 4) if average_precisions else 0.0


def compute_snippet_fmeasure(retrieved_snippets: list[dict],
                              gold_snippets: list[dict]) -> float:
    """
    Character-level F-measure for snippet extraction — Phase A secondary metric.
    Standard BioASQ evaluation: measures character overlap between
    retrieved snippet texts and gold snippet texts.

    Args:
        retrieved_snippets: List of retrieved snippet dicts with "text" field.
        gold_snippets:      List of gold snippet dicts with "text" field.

    Returns:
        F-measure score (float, 0–1).
    """
    retrieved_chars = set(
        char
        for s in retrieved_snippets
        for char in s.get("text", "")
    )
    gold_chars = set(
        char
        for s in gold_snippets
        for char in s.get("text", "")
    )

    if not retrieved_chars or not gold_chars:
        return 0.0

    precision = len(retrieved_chars & gold_chars) / len(retrieved_chars)
    recall    = len(retrieved_chars & gold_chars) / len(gold_chars)

    if precision + recall == 0:
        return 0.0

    f_measure = 2 * precision * recall / (precision + recall)
    return round(f_measure, 4)


# ===========================================================================
# Phase B — Yes/No
# ===========================================================================

def compute_yesno_accuracy(predictions: list[str],
                            references: list[str]) -> float:
    """
    Accuracy for yes/no questions.

    Args:
        predictions: List of predicted strings, each "yes" or "no".
        references:  List of gold strings, each "yes" or "no".

    Returns:
        Accuracy (float, 0–1).
    """
    if not predictions or not references:
        return 0.0

    correct = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return round(correct / len(predictions), 4)


def compute_yesno_macro_f1(predictions: list[str],
                            references: list[str]) -> float:
    """
    Macro F1 for yes/no questions — primary Phase B metric for this type.
    Computes F1 separately for "yes" and "no" classes, then averages.

    Args:
        predictions: List of predicted strings, each "yes" or "no".
        references:  List of gold strings, each "yes" or "no".

    Returns:
        Macro F1 score (float, 0–1).
    """
    if not predictions or not references:
        return 0.0

    f1_scores = []

    for label in ("yes", "no"):
        tp = sum(p.lower() == label and r.lower() == label for p, r in zip(predictions, references))
        fp = sum(p.lower() == label and r.lower() != label for p, r in zip(predictions, references))
        fn = sum(p.lower() != label and r.lower() == label for p, r in zip(predictions, references))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return round(sum(f1_scores) / len(f1_scores), 4)


# ===========================================================================
# Phase B — Factoid
# ===========================================================================

def compute_mrr(ranked_predictions: list[list[str]],
                gold_answers: list[list[str]]) -> float:
    """
    Mean Reciprocal Rank (MRR) for factoid questions — Phase B primary metric.

    For each question, finds the rank of the first correct candidate
    in the model's ranked answer list. MRR = mean of 1/rank across questions.

    Args:
        ranked_predictions: List of candidate lists per question, ranked by
                            confidence (most likely first).
                            e.g. [["RET", "GDNF", "SOX10"], ["p53"]]
        gold_answers:       List of acceptable answer lists per question.
                            e.g. [["RET"], ["TP53", "p53"]]

    Returns:
        MRR score (float, 0–1).
    """
    if not ranked_predictions or not gold_answers:
        return 0.0

    reciprocal_ranks = []

    for candidates, gold in zip(ranked_predictions, gold_answers):
        gold_set     = {g.strip().lower() for g in gold}
        rr           = 0.0

        for rank, candidate in enumerate(candidates, start=1):
            if candidate.strip().lower() in gold_set:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


# ===========================================================================
# Phase B — List
# ===========================================================================

def compute_list_f1(predictions: list[list[str]],
                    references: list[list[str]]) -> float:
    """
    Mean F1 for list questions — Phase B primary metric for this type.
    For each question, computes token-level F1 between the predicted
    item list and the gold item list, then averages across all questions.

    Args:
        predictions: List of predicted item lists per question.
                     e.g. [["RET", "GDNF", "SOX10"], ["BRCA1"]]
        references:  List of gold item lists per question.
                     e.g. [["RET", "GDNF", "EDNRB", "SOX10"], ["BRCA1", "BRCA2"]]

    Returns:
        Mean F1 score (float, 0–1).
    """
    if not predictions or not references:
        return 0.0

    f1_scores = []

    for pred_list, gold_list in zip(predictions, references):
        pred_set = {p.strip().lower() for p in pred_list}
        gold_set = {g.strip().lower() for g in gold_list}

        if not pred_set or not gold_set:
            f1_scores.append(0.0)
            continue

        tp        = len(pred_set & gold_set)
        precision = tp / len(pred_set)
        recall    = tp / len(gold_set)
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return round(sum(f1_scores) / len(f1_scores), 4)


# ===========================================================================
# Phase B — Summary
# ===========================================================================

def compute_rouge_l(predictions: list[str],
                    references: list[str]) -> float:
    """
    Mean ROUGE-L F1 for summary questions — Phase B primary metric for this type.
    ROUGE-L measures longest common subsequence overlap between
    predicted and reference summaries.

    Args:
        predictions: List of predicted summary strings.
        references:  List of gold summary strings.

    Returns:
        Mean ROUGE-L F1 score (float, 0–1).
    """
    if not predictions or not references:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)

    return round(sum(scores) / len(scores), 4)


def llm_as_judge(question: str,
                  answer: str,
                  snippets: list[dict],
                  judge_model) -> dict:
    """
    LLM-as-judge for summary questions — evaluates medical factuality
    and grounding against retrieved snippets.

    Uses build_judge_prompt() from prompt_templates.py and the Google
    GenerativeModel interface (same as Lowami's Gemini setup).

    Args:
        question:    The original question body.
        answer:      The generated summary answer to evaluate.
        snippets:    Retrieved snippets used to generate the answer.
        judge_model: Loaded google.generativeai.GenerativeModel instance.

    Returns:
        {
            "factuality_score": int (1–5),
            "grounded":         bool,
            "flagged":          bool,
            "reasoning":        str,
        }
    """
    import json
    from prompt_templates import build_judge_prompt

    prompt   = build_judge_prompt(question, answer, snippets)
    response = judge_model.generate_content(
        prompt,
        generation_config={"temperature": 0.0},
    )

    raw = response.text.strip() \
                       .removeprefix("```json") \
                       .removeprefix("```") \
                       .removesuffix("```") \
                       .strip()

    try:
        result = json.loads(raw)
        return {
            "factuality_score": int(result.get("factuality_score", 0)),
            "grounded":         bool(result.get("grounded", False)),
            "flagged":          bool(result.get("flagged", True)),
            "reasoning":        result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[WARNING] Judge parse error: {e}\nRaw: {raw}")
        return {
            "factuality_score": 0,
            "grounded":         False,
            "flagged":          True,
            "reasoning":        f"Parse error: {e}",
        }


# ===========================================================================
# Conversational Metrics
# ===========================================================================

def compute_context_retention_accuracy(predictions: list[dict]) -> float:
    """
    Context Retention Accuracy — primary conversational metric.

    Measures the percentage of answers that are correct specifically
    on turns that require information from prior turns
    (i.e. turns where requires_context=True).

    A prediction is considered correct if its answer string has
    non-zero token overlap with the gold answer.

    Args:
        predictions: List of prediction dicts, each containing:
                     {
                       "answer":           str,   ← model's answer
                       "gold_answer":      str,   ← ground truth
                       "requires_context": bool,  ← set by ConversationManager
                     }

    Returns:
        Accuracy over context-dependent turns (float, 0–1).
        Returns 0.0 if no context-dependent turns exist.
    """
    context_turns = [p for p in predictions if p.get("requires_context", False)]

    if not context_turns:
        print("[WARNING] No context-dependent turns found in predictions.")
        return 0.0

    correct = 0
    for p in context_turns:
        answer      = p.get("answer", "")
        gold_answer = p.get("gold_answer", "")
        # Normalize list answers (factoid/list types) to a single string
        if isinstance(answer, list):
            answer = " ".join(answer)
        if isinstance(gold_answer, list):
            gold_answer = " ".join(gold_answer)
        pred_tokens = set(answer.lower().split())
        gold_tokens = set(gold_answer.lower().split())
        if pred_tokens & gold_tokens:
            correct += 1

    return round(correct / len(context_turns), 4)


class LatencyTracker:
    """
    Throughput / Latency tracker — proposal metric for clinical efficiency.

    Wraps individual inference calls to measure:
      - Per-query response time (seconds)
      - Mean latency across all queries
      - Throughput (queries per second)

    Usage:
        tracker = LatencyTracker()
        with tracker.measure():
            answer = rag_system.answer(question)
        print(tracker.summary())
    """

    def __init__(self):
        self._latencies: list[float] = []
        self._start: float | None    = None

    class _Timer:
        def __init__(self, tracker):
            self._tracker = tracker

        def __enter__(self):
            self._tracker._start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self._tracker._start
            self._tracker._latencies.append(elapsed)

    def measure(self) -> "_Timer":
        """Context manager — wraps a single inference call."""
        return self._Timer(self)

    def mean_latency(self) -> float:
        """Mean response time in seconds."""
        if not self._latencies:
            return 0.0
        return round(sum(self._latencies) / len(self._latencies), 4)

    def throughput(self) -> float:
        """Queries per second."""
        if not self._latencies:
            return 0.0
        return round(len(self._latencies) / sum(self._latencies), 4)

    def summary(self) -> dict:
        """Return full latency summary dict."""
        return {
            "total_queries":      len(self._latencies),
            "mean_latency_s":     self.mean_latency(),
            "throughput_qps":     self.throughput(),
            "min_latency_s":      round(min(self._latencies), 4) if self._latencies else 0.0,
            "max_latency_s":      round(max(self._latencies), 4) if self._latencies else 0.0,
        }

    def reset(self):
        """Clear all recorded latencies."""
        self._latencies = []
        self._start     = None