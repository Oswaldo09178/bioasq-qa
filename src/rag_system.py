# Orchestrator -- Oswaldo

class BioASQRAGSystem:
    def __init__(self, config: dict):
        """Initialize all modules from config.yaml."""

    def index_corpus(self, corpus: list[dict]) -> None:
        """Build BM25 + dense indices over the document corpus."""

    def retrieve(self, query: str, 
                 top_k: int = 5) -> list[dict]:
        """Full hybrid retrieval pipeline for a given query."""

    def answer(self, question: dict, 
               session_id: str = None) -> dict:
        """
        Full single-question pipeline:
        1. Resolve anaphora & build contextualized query
        2. Retrieve documents
        3. Route to correct prompt template
        4. Generate & ground-check answer
        Returns: {"answer", "retrieved_docs", "grounded": bool}
        """

    def chat(self, user_message: str, 
             session_id: str) -> str:
        """
        Multi-turn interface: maintains ConversationManager state
        per session_id. Entry point for the Streamlit/Gradio UI.
        """

    def run_batch(self, questions: list[dict]) -> list[dict]:
        """Run the full pipeline over a list of BioASQ questions for eval."""