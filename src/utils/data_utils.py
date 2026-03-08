
# Data Utilities -- Lowami

def load_bioasq_dataset(filepath: str) -> list[dict]:
    """Load the raw BioASQ JSON file and return list of questions."""

def parse_question(question: dict) -> dict:
    """
    Extract and normalize fields from a raw question dict.
    Returns: {id, body, type, documents, snippets, ideal_answer}
    """

def get_snippets(question: dict) -> list[dict]:
    """
    Extract snippet texts and their source document URLs.
    Returns: [{"text": ..., "document": ..., "begin": ..., "end": ...}]
    """

def filter_by_type(questions: list[dict], qtype: str) -> list[dict]:
    """Filter questions by type: 'yesno', 'factoid', 'list', 'summary'."""

def split_dataset(questions: list[dict], 
                  val_ratio: float = 0.1) -> tuple[list, list]:
    """Split into train/validation sets."""