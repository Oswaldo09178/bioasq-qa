# Conversation Manager (Oswaldo)
# Handles multi-turn dialogue state, anaphora resolution,
# and contextualized query building for the BioASQ RAG system.

import re
from datetime import datetime


# ===========================================================================
# Turn Schema
# ===========================================================================
# Each turn stored internally looks like:
# {
#   "turn_id":        int,
#   "role":           "user" | "assistant",
#   "content":        str,
#   "timestamp":      str (ISO),
#   "retrieved_docs": list[dict] | None,   # docs cited in this turn
#   "entities":       list[str],           # extracted medical entities
#   "requires_context": bool               # True if anaphoric/contextual
# }


# ===========================================================================
# Medical Entity & Anaphora Patterns
# ===========================================================================

# Anaphoric expressions common in clinical dialogue
ANAPHORIC_PATTERNS = [
    r"\bthis (drug|medication|treatment|gene|condition|disease|mutation|therapy|compound|protein|receptor)\b",
    r"\bthat (drug|medication|treatment|gene|condition|disease|mutation|therapy|compound|protein|receptor)\b",
    r"\bthese (drugs|medications|treatments|genes|conditions|mutations|therapies|compounds|proteins|receptors)\b",
    r"\bthose (drugs|medications|treatments|genes|conditions|mutations|therapies|compounds|proteins|receptors)\b",
    r"\bit\b",
    r"\bthey\b",
    r"\bthe (same|aforementioned|above)\b",
    r"\bthe latter\b",
    r"\bthe former\b",
]

# Simple medical entity extraction patterns
# In production, replace with a NER model (e.g., scispaCy en_ner_bc5cdr_md)
ENTITY_PATTERNS = [
    r"\b[A-Z]{2,}(?:\d+)?\b",                    # gene symbols: RET, GDNF, SOX10
    r"\b\w+(?:mab|nib|vir|mycin|cillin|pril)\b",  # drug suffixes
    r"\b(?:syndrome|disease|disorder|carcinoma|tumor|cancer|mutation|receptor|pathway)\b",
]


# ===========================================================================
# ConversationManager
# ===========================================================================

class ConversationManager:

    def __init__(self,
                 window_size: int = 5,
                 memory_strategy: str = "sliding_window",
                 session_id: str = None):
        """
        Args:
            window_size:       Number of past turns to retain in context window.
            memory_strategy:   'sliding_window' | 'summary'
                               - sliding_window: keep last N turns verbatim
                               - summary: compress history beyond window into
                                 a single summary turn
            session_id:        Optional identifier for this conversation session.
                               Used by rag_system.py to route multi-user sessions.
        """
        self.window_size = window_size
        self.memory_strategy = memory_strategy
        self.session_id = session_id or datetime.utcnow().isoformat()

        self._history: list[dict] = []      # full turn history
        self._summary: str = ""             # compressed history (summary strategy)
        self._turn_counter: int = 0
        self._entity_registry: dict = {}    # maps anaphoric targets to resolved entities

    # -----------------------------------------------------------------------
    # Core History Management
    # -----------------------------------------------------------------------

    def add_turn(self,
                 role: str,
                 content: str,
                 retrieved_docs: list[dict] = None) -> None:
        """
        Append a turn to the conversation history.

        Args:
            role:           'user' | 'assistant'
            content:        The raw text of the turn.
            retrieved_docs: Documents cited/retrieved in this turn.
                            Format: [{"doc_id", "text", "pmid", ...}]
                            Stored so future turns can reference them.

        Side effects:
            - Extracts medical entities from the turn and updates _entity_registry.
            - If memory_strategy='summary' and history exceeds window_size,
              compresses the oldest turns into _summary.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{role}'")

        self._turn_counter += 1

        entities = self._extract_entities(content)
        self._update_entity_registry(entities)

        turn = {
            "turn_id": self._turn_counter,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "retrieved_docs": retrieved_docs or [],
            "entities": entities,
            "requires_context": self._check_requires_context(content),
        }

        self._history.append(turn)

        # If using summary strategy, compress when history grows beyond window
        if self.memory_strategy == "summary":
            self._maybe_compress_history()

    def get_context_window(self) -> list[dict]:
        """
        Return the turns visible to the LLM as {role, content} dicts.

        For 'sliding_window': returns the last window_size turns verbatim.
        For 'summary': prepends a synthetic summary turn to the window turns,
                       giving the LLM awareness of earlier context.
        """
        recent_turns = self._history[-self.window_size:]

        if self.memory_strategy == "summary" and self._summary:
            summary_turn = {
                "role": "assistant",
                "content": f"[Earlier conversation summary]: {self._summary}"
            }
            return [summary_turn] + [
                {"role": t["role"], "content": t["content"]}
                for t in recent_turns
            ]

        return [
            {"role": t["role"], "content": t["content"]}
            for t in recent_turns
        ]

    def get_full_history(self) -> list[dict]:
        """Return the complete unfiltered history (for evaluation and logging)."""
        return self._history

    def get_cited_documents(self, last_n_turns: int = None) -> list[dict]:
        """
        Retrieve all documents cited across the last N turns.
        Useful for building the evidence pool for the next answer.

        Args:
            last_n_turns: If None, returns docs from the entire history.
        """
        turns = self._history[-last_n_turns:] if last_n_turns else self._history
        seen_doc_ids = set()
        docs = []
        for turn in turns:
            for doc in turn.get("retrieved_docs", []):
                if doc["doc_id"] not in seen_doc_ids:
                    seen_doc_ids.add(doc["doc_id"])
                    docs.append(doc)
        return docs

    # -----------------------------------------------------------------------
    # Query Enhancement
    # -----------------------------------------------------------------------

    def resolve_anaphora(self, query: str) -> str:
        """
        Detect and resolve anaphoric references in the user's query.
        Substitutes expressions like 'this drug', 'that gene', 'they'
        with the most recently mentioned entity of the matching type
        from _entity_registry.

        Args:
            query: Raw user query string.

        Returns:
            Query with anaphoric references substituted where possible.
            If no resolution is found, the original expression is kept.

        Example:
            History mentions "RET gene".
            Query: "What are the contraindications for that gene?"
            Returns: "What are the contraindications for RET?"
        """
        resolved = query

        for pattern in ANAPHORIC_PATTERNS:
            match = re.search(pattern, resolved, flags=re.IGNORECASE)
            if not match:
                continue

            matched_text = match.group(0)

            # Extract the noun category if present (e.g. "drug", "gene")
            words = matched_text.lower().split()
            category = words[-1] if len(words) > 1 else None

            # Find the most recent entity matching the category
            candidate = self._resolve_from_registry(category)

            if candidate:
                resolved = resolved[:match.start()] + candidate + resolved[match.end():]

        if resolved != query:
            print(f"[ConversationManager] Anaphora resolved: '{query}' → '{resolved}'")

        return resolved

    def build_contextualized_query(self, raw_query: str) -> str:
        """
        Enrich the raw user query with entities and context from prior turns
        to improve retrieval recall.

        Strategy:
            1. Resolve any anaphoric references.
            2. Append the most recently mentioned entities not already
               present in the query.

        Args:
            raw_query: The user's latest message.

        Returns:
            An enriched query string for the retrieval module.

        Example:
            raw_query = "What are the contraindications?"
            recent entities = ["RET", "Hirschsprung disease"]
            → "What are the contraindications? RET Hirschsprung disease"
        """
        # Step 1: resolve anaphora
        resolved_query = self.resolve_anaphora(raw_query)

        # Step 2: gather recent entities not already in the query
        recent_entities = self._get_recent_entities(last_n_turns=3)
        query_lower = resolved_query.lower()
        appendix = [
            e for e in recent_entities
            if e.lower() not in query_lower
        ]

        if appendix:
            contextualized = resolved_query + " " + " ".join(appendix)
            print(f"[ConversationManager] Context appended: {appendix}")
            return contextualized

        return resolved_query

    def is_query_underspecified(self, query: str) -> bool:
        """
        Detect if a query is too vague to answer without clarification.

        A query is considered underspecified if:
            - It contains an unresolvable anaphoric reference (e.g., 'it',
              'they', 'that drug') with no matching entity in the registry.
            - It is shorter than 4 tokens (extremely short queries).
            - It contains a question word but no medical noun or entity.

        Args:
            query: Raw user query.

        Returns:
            True if the system should generate a clarifying question
            instead of attempting retrieval.
        """
        tokens = query.strip().split()

        # Too short
        if len(tokens) < 4:
            return True

        # Contains anaphoric reference that cannot be resolved
        for pattern in ANAPHORIC_PATTERNS:
            if re.search(pattern, query, flags=re.IGNORECASE):
                # Check if we can resolve it
                words = query.lower().split()
                category = words[-1] if len(words) > 1 else None
                if not self._resolve_from_registry(category):
                    return True

        return False

    # -----------------------------------------------------------------------
    # History Compression (Summary Strategy)
    # -----------------------------------------------------------------------

    def summarize_history(self, llm=None) -> str:
        """
        Compress conversation history older than the current window into
        a plain-text summary.

        If an LLM is provided, uses it to generate the summary.
        Otherwise, falls back to a simple template-based extractive summary
        listing the questions asked and key entities mentioned.

        Args:
            llm: Optional loaded LLM (from generation_utils.load_llm).
                 If None, uses extractive fallback.

        Returns:
            A summary string stored in self._summary.
        """
        old_turns = self._history[:-self.window_size]
        if not old_turns:
            return self._summary

        if llm is not None:
            # LLM-based compression — prompt built inline here
            # (avoids circular import with generation_utils)
            history_text = "\n".join(
                f"{t['role'].upper()}: {t['content']}" for t in old_turns
            )
            prompt = (
                "Summarize the following clinical conversation history in 2-3 sentences. "
                "Focus on the medical conditions, drugs, genes, and questions discussed. "
                "Be concise.\n\n"
                f"{history_text}\n\nSummary:"
            )
            from generation_utils import generate_answer
            self._summary = generate_answer(prompt, llm, max_tokens=150)
        else:
            # Extractive fallback
            questions = [
                t["content"] for t in old_turns if t["role"] == "user"
            ]
            all_entities = []
            for t in old_turns:
                all_entities.extend(t.get("entities", []))
            unique_entities = list(dict.fromkeys(all_entities))

            self._summary = (
                f"Previously discussed questions: {'; '.join(questions[:3])}. "
                f"Key entities mentioned: {', '.join(unique_entities[:10])}."
            )

        return self._summary

    # -----------------------------------------------------------------------
    # Session Management
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all conversation state for a new session.
        Resets history, summary, entity registry, and turn counter.
        """
        self._history = []
        self._summary = ""
        self._turn_counter = 0
        self._entity_registry = {}
        print(f"[ConversationManager] Session {self.session_id} reset.")

    def to_dict(self) -> dict:
        """
        Serialize the full session state to a dict.
        Used by rag_system.py to persist sessions and by Lowami's
        evaluation pipeline to log multi-turn dialogues.

        Returns: {session_id, history, summary, entity_registry}
        """
        return {
            "session_id": self.session_id,
            "memory_strategy": self.memory_strategy,
            "window_size": self.window_size,
            "turn_count": self._turn_counter,
            "summary": self._summary,
            "entity_registry": self._entity_registry,
            "history": self._history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationManager":
        """
        Restore a ConversationManager from a serialized dict.
        Useful for resuming sessions across API calls.
        """
        manager = cls(
            window_size=data.get("window_size", 5),
            memory_strategy=data.get("memory_strategy", "sliding_window"),
            session_id=data.get("session_id"),
        )
        manager._history = data.get("history", [])
        manager._summary = data.get("summary", "")
        manager._turn_counter = data.get("turn_count", 0)
        manager._entity_registry = data.get("entity_registry", {})
        return manager

    # -----------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract medical entities from text using regex patterns.
        In production, swap this for a scispaCy NER pipeline.
        Returns a deduplicated list of entity strings.
        """
        entities = []
        for pattern in ENTITY_PATTERNS:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            entities.extend(matches)
        # Deduplicate while preserving order
        return list(dict.fromkeys(entities))

    def _update_entity_registry(self, entities: list[str]) -> None:
        """
        Update the entity registry with newly extracted entities.
        Maps each entity to its last seen turn_id for recency-based resolution.
        """
        for entity in entities:
            self._entity_registry[entity.lower()] = {
                "surface": entity,
                "last_turn": self._turn_counter
            }

    def _resolve_from_registry(self, category: str = None) -> str | None:
        """
        Find the most recently mentioned entity in the registry.
        If category is provided (e.g. 'gene', 'drug'), filters loosely
        by checking if the entity is associated with that category in history.
        Returns the surface form of the best candidate, or None.
        """
        if not self._entity_registry:
            return None

        # Sort by most recently seen turn
        sorted_entities = sorted(
            self._entity_registry.values(),
            key=lambda x: x["last_turn"],
            reverse=True
        )
        # Return the most recent entity
        return sorted_entities[0]["surface"] if sorted_entities else None

    def _get_recent_entities(self, last_n_turns: int = 3) -> list[str]:
        """
        Return entities mentioned in the last N turns, most recent first.
        Deduplicates across turns.
        """
        recent = self._history[-last_n_turns:]
        seen = set()
        entities = []
        for turn in reversed(recent):
            for e in turn.get("entities", []):
                if e.lower() not in seen:
                    seen.add(e.lower())
                    entities.append(e)
        return entities

    def _check_requires_context(self, query: str) -> bool:
        """
        Determine whether this turn requires information from prior turns.
        Used to set the requires_context flag — critical for Lowami's
        compute_context_retention_accuracy() metric.
        Returns True if any anaphoric pattern is detected.
        """
        for pattern in ANAPHORIC_PATTERNS:
            if re.search(pattern, query, flags=re.IGNORECASE):
                return True
        return False

    def _maybe_compress_history(self) -> None:
        """
        Trigger history compression when history exceeds the window size.
        Called automatically by add_turn() under the summary strategy.
        """
        if len(self._history) > self.window_size:
            self.summarize_history(llm=None)  # extractive fallback by default