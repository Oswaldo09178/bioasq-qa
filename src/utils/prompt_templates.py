# --- prompt functions  ---

def build_summary_prompt(question: str, 
                         snippets: list[dict], 
                         history: list[dict]) -> str:
    """CoT prompt for summary-type questions with strict grounding instruction."""

def build_yesno_prompt(question: str, 
                       snippets: list[dict], 
                       history: list[dict]) -> str:
    """Prompt for yes/no questions, asks model to justify with evidence."""

def build_factoid_prompt(question: str, 
                         snippets: list[dict], 
                         history: list[dict]) -> str:
    """Prompt for factoid extraction."""

def build_list_prompt(question: str, 
                      snippets: list[dict], 
                      history: list[dict]) -> str:
    """Prompt for list-type questions."""

def build_clarification_prompt(question: str, 
                               history: list[dict]) -> str:
    """Prompt to generate a clarifying question when query is underspecified."""


