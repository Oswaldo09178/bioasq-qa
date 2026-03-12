
# Data Utilities -- Lowami
import os
import json

def load_bioasq_dataset(filepath: str) -> list[dict]:
    """Load the raw BioASQ JSON file and return list of questions."""
    if not os.path.exists(filepath):
        print(os.path.abspath(filepath))
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['questions']

def parse_question(question: dict, minimized: bool = False) -> dict:
    """
    Extract and normalize fields from a raw question dict.
    Returns: {id, body, type, documents, snippets, ideal_answer}
    """
    if minimized:
        return {
            "question": question.get("body"),
            "answer": question.get("ideal_answer") or question.get("exact_answer"),
            "context": [s['text'] for s in question['snippets']]
        }
    return {
        "id": question.get("id"),
        "body": question.get("body"),
        "type": question.get("type"),
        "documents": question.get("documents"),
        "snippets": question.get("snippets"),
        "ideal_answer": question.get("ideal_answer") or question.get("exact_answer")
    }

def get_snippets(question: dict) -> list[dict]:
    """
    Extract snippet texts and their source document URLs.
    Returns: [{"text": ..., "document": ..., "begin": ..., "end": ...}]
    """
    snippets = []
    for snippet in question.get("snippets", []):
        snippets.append({
            "text": snippet.get("text"),
            "document": snippet.get("document"),
            "begin": snippet.get("beginSection"),
            "end": snippet.get("endSection")
        })
    return snippets


def filter_by_type(questions: list[dict], qtype: str) -> list[dict]:
    """Filter questions by type: 'yesno', 'factoid', 'list', 'summary'."""
    return [q for q in questions if q.get("type") == qtype]

def split_dataset(questions: list[dict], 
                  val_ratio: float = 0.1) -> tuple[list, list]:
    """Split into train/validation sets."""
    split_idx = int(len(questions) * (1 - val_ratio))
    return questions[:split_idx], questions[split_idx:]