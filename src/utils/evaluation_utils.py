# --- evaluation functions (Lowami)---

def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L score between prediction and ideal answer."""

def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 for list and factoid answers."""

def compute_mrr(ranked_answers: list[str], 
                gold_answer: str) -> float:
    """Mean Reciprocal Rank for factoid questions."""

def compute_map(retrieved_docs: list[str], 
                gold_docs: list[str]) -> float:
    """Mean Average Precision for retrieval Phase A evaluation."""

def compute_snippet_fmeasure(retrieved_snippets: list[dict], 
                              gold_snippets: list[dict]) -> float:
    """Character-level F-measure on extracted snippets."""

def llm_as_judge(question: str, answer: str, 
                 snippets: list[dict], judge_model) -> dict:
    """
    Use an LLM to score medical factuality and grounding.
    Returns: {"factuality_score": float, "reasoning": str, "flagged": bool}
    """

def compute_context_retention_accuracy(predictions: list[dict]) -> float:
    """
    For multi-turn eval: % of answers correct where the query 
    required info from a previous turn.
    """
