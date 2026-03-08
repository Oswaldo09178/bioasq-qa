# Conversation Manager -- Oswaldo


class ConversationManager:
    def __init__(self, window_size: int = 5, 
                 memory_strategy: str = "sliding_window"):
        """
        window_size: number of past turns to retain
        memory_strategy: 'sliding_window' | 'summary'
        """

    def add_turn(self, role: str, content: str, 
                 retrieved_docs: list[dict] = None) -> None:
        """
        Append a turn to conversation history.
        role: 'user' | 'assistant'
        Also stores which documents were cited in that turn.
        """

    def get_context_window(self) -> list[dict]:
        """Return the last N turns as a list of {role, content} dicts."""

    def resolve_anaphora(self, query: str) -> str:
        """
        Detect and resolve references like 'that drug', 'this condition'
        by substituting with entities from recent turns.
        """

    def build_contextualized_query(self, raw_query: str) -> str:
        """
        Combine raw query with conversation context for retrieval.
        E.g. append relevant entities from prior turns.
        """

    def is_query_underspecified(self, query: str) -> bool:
        """
        Detect if a query lacks sufficient context to be answered.
        Returns True if a clarifying question should be generated.
        """

    def summarize_history(self) -> str:
        """Compress conversation history into a short summary (for summary strategy)."""

    def reset(self) -> None:
        """Clear conversation state for a new session."""