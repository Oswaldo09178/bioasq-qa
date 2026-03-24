# Data Utilities — Lowami

import difflib
import json
import random
from pathlib import Path


def load_bioasq_dataset(filepath: str) -> list[dict]:
    """
    Load the raw BioASQ JSON file and return list of questions.
    If the exact filename is not found, attempts fuzzy matching
    against other JSON files in the same directory.
    """
    path = Path(filepath)

    if not path.exists():
        dataset_dir  = path.parent
        resolved_path = None

        if dataset_dir.exists():
            json_candidates = sorted(dataset_dir.glob("*.json"))
            if json_candidates:
                candidate_names = [p.name for p in json_candidates]
                close_match = difflib.get_close_matches(
                    path.name, candidate_names, n=1, cutoff=0.7
                )
                if close_match:
                    resolved_path = dataset_dir / close_match[0]
                    print(f"[INFO] Fuzzy match: '{path.name}' → '{close_match[0]}'")
                elif len(json_candidates) == 1:
                    resolved_path = json_candidates[0]
                    print(f"[INFO] Single JSON found, using: '{resolved_path.name}'")

        if resolved_path is None:
            raise FileNotFoundError(
                f"Dataset file not found: {filepath}\n"
                f"Searched in: {path.parent.resolve()}"
            )

        path = resolved_path

    with open(path, "r") as f:
        if filepath.endswith(".jsonl"):
            questions = [json.loads(line)["request"]["contents"][0]["parts"][0]["text"] for line in f.readlines()]
        else:
            data = json.load(f)
            questions = data["questions"]
    print(f"[INFO] Loaded {len(questions)} questions from {path.name}")
    return questions


def parse_question(question: dict, minimized: bool = False) -> dict:
    """
    Extract and normalize fields from a raw question dict.

    Args:
        question:  Raw BioASQ question dict.
        minimized: If True, returns a compact format used by
                   synthetic_data_utils for batch prompt building:
                   {question, answer, context}
                   If False (default), returns the full normalized dict:
                   {id, body, type, documents, snippets, ideal_answer}
    """
    # ideal_answer: free-text narrative (list → take first element) — used for summary ROUGE-L
    # exact_answer: structured gold (list of strings or list of lists) — used for factoid/list/yesno
    raw_ideal = question.get("ideal_answer", "")
    if isinstance(raw_ideal, list):
        ideal_answer = raw_ideal[0] if raw_ideal else ""
    else:
        ideal_answer = raw_ideal or ""

    exact_answer = question.get("exact_answer")  # kept raw — may be list/nested list

    if minimized:
        return {
            "question": question.get("body"),
            "answer":   ideal_answer,
            "context":  [s["text"] for s in question.get("snippets", [])],
        }

    return {
        "id":           question.get("id"),
        "body":         question.get("body"),
        "type":         question.get("type"),
        "documents":    question.get("documents"),
        "snippets":     question.get("snippets"),
        "ideal_answer": ideal_answer,    # str — for summary generation + ROUGE-L
        "exact_answer": exact_answer,    # raw list — for factoid/list/yesno evaluation
    }


def get_snippets(question: dict) -> list[dict]:
    """
    Extract snippets from a question dict, normalizing field names
    for downstream use by retrieval_utils.build_corpus_from_bioasq().

    BioASQ raw fields:
        offsetInBeginSection / offsetInEndSection → character offsets (int)
        beginSection / endSection                 → section names (e.g. "abstract")

    Returns:
        [{"text", "document", "begin", "end", "section"}]
        where begin/end are integer character offsets used to build
        unique chunk IDs in retrieval_utils.
    """
    snippets = []
    for snippet in question.get("snippets", []):
        snippets.append({
            "text":     snippet.get("text"),
            "document": snippet.get("document"),
            "begin":    snippet.get("offsetInBeginSection"),  # int offset
            "end":      snippet.get("offsetInEndSection"),    # int offset
            "section":  snippet.get("beginSection"),          # section name
        })
    return snippets


def filter_by_type(questions: list[dict], qtype: str) -> list[dict]:
    """
    Filter questions by type.
    Valid types: 'yesno', 'factoid', 'list', 'summary'
    """
    valid_types = {"yesno", "factoid", "list", "summary"}
    if qtype not in valid_types:
        raise ValueError(f"Unknown question type '{qtype}'. Must be one of {valid_types}")
    return [q for q in questions if q.get("type") == qtype]


def split_dataset(questions: list[dict],
                  val_ratio: float = 0.1,
                  shuffle: bool = True,
                  seed: int = 42) -> tuple[list, list]:
    """
    Split questions into train and validation sets.

    Args:
        questions: List of parsed or raw BioASQ question dicts.
        val_ratio: Fraction of data to use for validation (default 0.1).
        shuffle:   Whether to shuffle before splitting (default True).
                   Important: BioASQ questions are ordered by type in the
                   source file, so without shuffling the val set will be
                   dominated by whichever type appears last.
        seed:      Random seed for reproducibility.

    Returns:
        (train_questions, val_questions)
    """
    if shuffle:
        questions = questions.copy()
        random.seed(seed)
        random.shuffle(questions)

    split_idx = int(len(questions) * (1 - val_ratio))
    train, val = questions[:split_idx], questions[split_idx:]

    print(f"[INFO] Dataset split — train: {len(train)}, val: {len(val)}")
    return train, val

def get_questions_by_type(questions, q_type, n):
    """
    Helper function to get the first n questions of a specific type.
    This function can be used in analysis of generation strength by question type.
    """
    results = []
    l = 0
    for q in questions:
        if q['type'] == q_type:
            results.append(q)
            l += 1
        if l == n:
            break
    return results