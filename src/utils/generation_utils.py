# --- generation functions (Oswaldo)---

def load_llm(model_name: str, backend: str = "huggingface"):
    """
    Load an LLM backend.
    backend: 'huggingface' | 'openai' | 'google'
    Supports: PubMedBERT, MedGemma, GPT-4
    """

def generate_answer(prompt: str, 
                    model, 
                    max_tokens: int = 512) -> str:
    """Run inference and return the generated answer string."""

def route_by_question_type(question: dict, 
                            snippets: list[dict], 
                            history: list[dict],
                            model) -> str:
    """
    Select the right prompt template based on question type
    ('summary', 'yesno', 'factoid', 'list') and generate answer.
    """

def check_answer_grounded(answer: str, 
                           snippets: list[dict]) -> bool:
    """
    'Strict Grounding' check — verify the answer is supported
    by at least one retrieved snippet. Can use LLM-as-judge internally.
    """