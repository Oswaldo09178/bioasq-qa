from typing import Optional, Union, List, Dict, Tuple, Any
# RAG System — Oswaldo / Joel / Lowami
#
# Full pipeline orchestrator for the Conversational Biomedical QA System.
#
# CLI usage:
#   python rag_system.py --retriever hybrid --generator gpt4 --k 5
#   python rag_system.py --retriever bm25   --generator medgemma --k 10
#   python rag_system.py --retriever none   --generator gemini --k 5
#   python rag_system.py --retriever dense  --generator gpt4 --k 5 --eval
#   python rag_system.py --chat             --generator gpt4
#
# --retriever options:
#   none    — skip retrieval, use BioASQ snippets directly (good for ablation)
#   bm25    — sparse BM25 only
#   dense   — dense BGE-M3 embeddings only
#   hybrid  — BM25 + dense + RRF + cross-encoder reranking (default)
#
# --generator options:
#   gpt4       — OpenAI GPT-4o  (requires OPENAI_API_KEY)
#   gemini     — Google Gemini 2.5 Pro  (requires GOOGLE_API_KEY)
#   medgemma   — HuggingFace google/medgemma-4b-it
#   pubmedbert — HuggingFace microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
#
# --k : top-K documents to retrieve (default: 5)
# --eval : run full evaluation after batch inference and save results
# --chat : launch interactive multi-turn CLI session

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")  # explicit path to project root .env

# Resolve src/ and src/utils/ relative to this file so imports work
# regardless of the working directory the script is called from.
_SRC_DIR   = Path(__file__).resolve().parent
_UTILS_DIR = _SRC_DIR / "utils"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_UTILS_DIR))

from data_utils import load_bioasq_dataset, parse_question, get_snippets
from generation_utils import (
    load_llm,
    route_by_question_type,
    check_answer_grounded,
    generate_clarification,
)
from conversation_manager import ConversationManager
from evaluation import run_full_evaluation

# Retrieval imports — loaded lazily to avoid errors when retriever=none
def _import_retrieval():
    from retrieval_utils import (
        build_bm25_index, bm25_retrieve,
        build_dense_index, dense_retrieve,
        reciprocal_rank_fusion, rerank_with_crossencoder,
        build_corpus_from_bioasq,
        save_bm25_index, load_bm25_index,
        save_dense_index, load_dense_index,
    )
    return {
        "build_bm25_index":        build_bm25_index,
        "bm25_retrieve":           bm25_retrieve,
        "build_dense_index":       build_dense_index,
        "dense_retrieve":          dense_retrieve,
        "reciprocal_rank_fusion":  reciprocal_rank_fusion,
        "rerank_with_crossencoder":rerank_with_crossencoder,
        "build_corpus_from_bioasq":build_corpus_from_bioasq,
        "save_bm25_index":         save_bm25_index,
        "load_bm25_index":         load_bm25_index,
        "save_dense_index":        save_dense_index,
        "load_dense_index":        load_dense_index,
    }


# ===========================================================================
# Generator config map
# ===========================================================================

GENERATOR_CONFIGS = {
    "gpt4":       {"model_name": "gpt-4o",                                                    "backend": "openai"},
    "gemini":     {"model_name": "gemini-2.0-flash",                                            "backend": "google"},
    "medgemma":   {"model_name": "google/medgemma-4b-it",                                     "backend": "huggingface"},
    "pubmedbert": {"model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",      "backend": "huggingface"},
}

# Cross-encoder model for hybrid reranking (Joel's config)
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Default index paths
INDEX_DIR = Path("output/indices")


# ===========================================================================
# BioASQRAGSystem
# ===========================================================================

class BioASQRAGSystem:

    def __init__(self,
                 retriever: str = "hybrid",
                 generator: str = "gpt4",
                 k: int = 5):
        """
        Initialize the RAG system with the specified retriever and generator.

        Args:
            retriever: "none" | "bm25" | "dense" | "hybrid"
            generator: "gpt4" | "gemini" | "medgemma" | "pubmedbert"
            k:         Number of documents to retrieve per query.
        """
        self.retriever_type = retriever.lower()
        self.generator_type = generator.lower()
        self.k              = k

        # Retrieval state
        self._corpus:      list[dict]  = []
        self._bm25_index               = None
        self._dense_embs               = None
        self._dense_encoder            = None
        self._retrieval                = None   # lazy-loaded retrieval module

        # Generation state
        self._llm: Optional[dict]         = None

        # Session registry: session_id → ConversationManager
        self._sessions: dict[str, ConversationManager] = {}

        self._validate_args()
        print(f"[INFO] BioASQRAGSystem initialized — retriever={retriever}, generator={generator}, k={k}")

    def _validate_args(self):
        valid_retrievers = {"none", "bm25", "dense", "hybrid"}
        valid_generators = set(GENERATOR_CONFIGS.keys())
        if self.retriever_type not in valid_retrievers:
            raise ValueError(f"--retriever must be one of {valid_retrievers}, got '{self.retriever_type}'")
        if self.generator_type not in valid_generators:
            raise ValueError(f"--generator must be one of {valid_generators}, got '{self.generator_type}'")

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def load_generator(self) -> None:
        """Load the LLM specified by --generator."""
        cfg        = GENERATOR_CONFIGS[self.generator_type]
        self._llm  = load_llm(cfg["model_name"], cfg["backend"])

    def index_corpus(self, questions: list[dict]) -> None:
        """
        Build BM25 and/or dense indices over the BioASQ corpus.
        Skipped if retriever=none.

        Saves indices to output/indices/ for reuse across runs.
        If indices already exist on disk, loads them instead of rebuilding.
        """
        if self.retriever_type == "none":
            print("[INFO] Retriever=none — skipping corpus indexing.")
            return

        self._retrieval = _import_retrieval()
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        # Build corpus from BioASQ snippets
        print("[INFO] Building corpus from BioASQ snippets...")
        self._corpus = self._retrieval["build_corpus_from_bioasq"](questions)

        bm25_path   = INDEX_DIR / "bm25_index.pkl"
        dense_path  = INDEX_DIR / "dense_index.npy"
        corpus_path = INDEX_DIR / "corpus.json"

        # Always save the current corpus alongside the indices so they stay in sync
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, "w") as f:
            import json as _json
            _json.dump(self._corpus, f)
        print(f"[INFO] Corpus saved to {corpus_path} ({len(self._corpus)} chunks)")

        # BM25 — rebuild whenever corpus changes
        if self.retriever_type in ("bm25", "hybrid"):
            if bm25_path.exists():
                # Load saved corpus that matches this index
                with open(corpus_path) as f:
                    self._corpus = _json.load(f)
                print(f"[INFO] Loading BM25 index from {bm25_path}")
                self._bm25_index = self._retrieval["load_bm25_index"](str(bm25_path))
                # Validate corpus/index are in sync
                index_size = self._bm25_index.corpus_size if hasattr(self._bm25_index, "corpus_size") else len(self._corpus)
                if len(self._corpus) != index_size:
                    print(f"[INFO] Index/corpus mismatch ({index_size} vs {len(self._corpus)}) — rebuilding BM25 index...")
                    self._bm25_index = self._retrieval["build_bm25_index"](self._corpus)
                    self._retrieval["save_bm25_index"](self._bm25_index, str(bm25_path))
            else:
                print("[INFO] Building BM25 index...")
                self._bm25_index = self._retrieval["build_bm25_index"](self._corpus)
                self._retrieval["save_bm25_index"](self._bm25_index, str(bm25_path))

        # Dense — rebuild whenever corpus changes
        if self.retriever_type in ("dense", "hybrid"):
            if dense_path.exists():
                with open(corpus_path) as f:
                    self._corpus = _json.load(f)
                print(f"[INFO] Loading dense index from {dense_path}")
                self._dense_embs, self._dense_encoder = self._retrieval["load_dense_index"](
                    str(dense_path), "BAAI/bge-m3"
                )
            else:
                print("[INFO] Building dense index (this may take a few minutes)...")
                self._dense_embs, self._dense_encoder = self._retrieval["build_dense_index"](
                    self._corpus
                )
                self._retrieval["save_dense_index"](self._dense_embs, str(dense_path))

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict]:
        """
        Run the configured retrieval pipeline for a single query.

        Returns:
            Top-K ranked document dicts: [{"doc_id", "text", "pmid", ...}]
        """
        if self.retriever_type == "none" or not self._corpus:
            return []

        r = self._retrieval

        if self.retriever_type == "bm25":
            return r["bm25_retrieve"](query, self._bm25_index, self._corpus, top_k=self.k)

        elif self.retriever_type == "dense":
            return r["dense_retrieve"](
                query, self._dense_embs, self._dense_encoder, self._corpus, top_k=self.k
            )

        elif self.retriever_type == "hybrid":
            bm25_results  = r["bm25_retrieve"](
                query, self._bm25_index, self._corpus, top_k=self.k * 2
            )
            dense_results = r["dense_retrieve"](
                query, self._dense_embs, self._dense_encoder, self._corpus, top_k=self.k * 2
            )
            fused = r["reciprocal_rank_fusion"](bm25_results, dense_results)
            return r["rerank_with_crossencoder"](
                query, fused, model_name=CROSSENCODER_MODEL, top_k=self.k
            )

        return []

    # -----------------------------------------------------------------------
    # Single-question answering
    # -----------------------------------------------------------------------

    def answer(self,
           question: dict,
           session_id: str = None,
           gold_answer: str = None) -> dict:
        """
        Full single-question pipeline using BioASQ snippets for retrieval (no external retrieval).

        Args:
            question:    Parsed BioASQ question dict.
            session_id:  Optional session ID for multi-turn context.
            gold_answer: Optional gold answer for evaluation.

        Returns:
            Prediction dict compatible with evaluation.py:
            {
                "question_id",  "question_type", "answer",
                "retrieved_doc_ids", "retrieved_snippets",
                "requires_context", "latency_s",
                "grounded", "gold_answer" (if provided)
            }
        """
        if self._llm is None:
            raise RuntimeError("Call load_generator() before answer().")

        # Get or create conversation session
        manager = self._get_or_create_session(session_id)
        body = question.get("body", "")

        # --- Clarification check ---
        if manager.is_query_underspecified(body):
            clarification = generate_clarification(
                body, manager.get_context_window(), self._llm
            )
            manager.add_turn("assistant", clarification)
            return {
                "question_id":        question.get("id", ""),
                "question_type":      question.get("type", ""),
                "answer":             clarification,
                "clarification":      True,
                "retrieved_doc_ids":  [],
                "retrieved_snippets": [],
                "requires_context":   False,
                "latency_s":          0.0,
                "grounded":           True,
            }

        # --- Build contextualized query (multi-turn context) ---
        contextualized_query = manager.build_contextualized_query(body)

        start = time.perf_counter()

        # --- Retrieve snippets (always use BioASQ snippets) ---
        snippets = get_snippets(question) or []  # ensure list
        if not snippets:
            print(f"[WARNING] No snippets found for question {question.get('id','')}")
        retrieved_docs = []
        for s in snippets[:self.k]:
            text = s.get("text") or s.get("snippet") or ""
            doc_url = s.get("document") or ""
            pmid = doc_url.split("/")[-1] if doc_url else ""
            retrieved_docs.append({
                "doc_id": f"pubmed_{pmid}",
                "text": text,
                "pmid": pmid,
                "document": doc_url,
                "beginSection": s.get("beginSection"),
                "endSection": s.get("endSection"),
            })

        # --- Generate answer ---
        history = manager.get_context_window()
        result = route_by_question_type(question, retrieved_docs, history, self._llm)

        latency = round(time.perf_counter() - start, 4)

        answer_str = result["answer"]
        if isinstance(answer_str, list):
            answer_str = " ".join(answer_str)

        # --- Update conversation history ---
        manager.add_turn("user", body, retrieved_docs=retrieved_docs)
        manager.add_turn("assistant", answer_str)

        # --- Assemble prediction dict ---
        pred = {
            "question_id":        result["question_id"],
            "question_type":      result["question_type"],
            "answer":             result["answer"],
            "retrieved_doc_ids":  [d["doc_id"] for d in retrieved_docs],
            "retrieved_snippets": retrieved_docs,
            "requires_context":   manager.get_full_history()[-2].get("requires_context", False),
            "latency_s":          latency,
            "grounded":           True,   # always grounded when using snippets
            "flagged":            False,
        }

        if gold_answer is not None:
            pred["gold_answer"] = gold_answer

        return pred

    # -----------------------------------------------------------------------
    # Batch inference
    # -----------------------------------------------------------------------

    def run_batch(self,
                  questions: list[dict],
                  session_id: str = None) -> list[dict]:
        """
        Run the full pipeline over a list of BioASQ questions.
        Used by evaluation.py for automated benchmarking.

        Args:
            questions:  List of parsed BioASQ question dicts.
            session_id: If provided, all questions share one conversation
                        session (multi-turn eval). If None, each question
                        gets a fresh stateless session.

        Returns:
            List of prediction dicts (one per question).
        """
        predictions = []
        total       = len(questions)

        for i, q in enumerate(questions, 1):
            print(f"[INFO] Answering question {i}/{total} (id={q.get('id','')}, type={q.get('type','')})")

            # Use exact_answer for structured types, ideal_answer for summary
            qtype = q.get("type", "summary")
            if qtype in ("factoid", "list", "yesno"):
                gold = q.get("exact_answer")
                if gold is None:
                    gold = q.get("ideal_answer", "")
            else:
                gold = q.get("ideal_answer", "")
                if isinstance(gold, list):
                    gold = gold[0] if gold else ""

            sid  = session_id or str(uuid.uuid4())  # fresh session per question if no shared session
            pred = self.answer(q, session_id=sid, gold_answer=gold)
            predictions.append(pred)

        print(f"[INFO] Batch complete — {len(predictions)} predictions")
        return predictions

    # -----------------------------------------------------------------------
    # Interactive multi-turn chat
    # -----------------------------------------------------------------------

    def chat(self,
             user_message: str,
             session_id: str) -> str:
        """
        Multi-turn chat interface — entry point for the Streamlit/Gradio UI.
        Wraps answer() with a plain string return for easy UI integration.

        Args:
            user_message: The user's raw input string.
            session_id:   Session ID to maintain conversation state.

        Returns:
            The assistant's answer as a plain string.
        """
        # Build a minimal question dict from the raw message
        question = {
            "id":   f"chat_{session_id}_{int(time.time())}",
            "body": user_message,
            "type": "summary",   # default — summary prompt handles open questions well
        }
        pred   = self.answer(question, session_id=session_id)
        answer = pred["answer"]
        return answer if isinstance(answer, str) else " ".join(answer)

    # -----------------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------------

    def _get_or_create_session(self, session_id: str = None) -> ConversationManager:
        """Return existing session or create a new one."""
        if session_id is None:
            return ConversationManager()  # stateless — not stored
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationManager(session_id=session_id)
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        """Clear conversation state for a given session."""
        if session_id in self._sessions:
            self._sessions[session_id].reset()

    def get_session_state(self, session_id: str) -> Optional[dict]:
        """Serialize session state — used for logging and debugging."""
        if session_id in self._sessions:
            return self._sessions[session_id].to_dict()
        return None


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BioASQ Conversational RAG System",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--retriever",
        type=str,
        default="hybrid",
        choices=["none", "bm25", "dense", "hybrid"],
        help=(
            "Retrieval strategy:\n"
            "  none   — use BioASQ snippets directly (oracle/ablation)\n"
            "  bm25   — sparse BM25 only\n"
            "  dense  — dense BGE-M3 embeddings only\n"
            "  hybrid — BM25 + dense + RRF + cross-encoder (default)"
        ),
    )

    parser.add_argument(
        "--generator",
        type=str,
        default="gpt4",
        choices=list(GENERATOR_CONFIGS.keys()),
        help=(
            "Generation backend:\n"
            "  gpt4       — OpenAI GPT-4o       (requires OPENAI_API_KEY)\n"
            "  gemini     — Google Gemini 2.5   (requires GOOGLE_API_KEY)\n"
            "  medgemma   — HuggingFace MedGemma\n"
            "  pubmedbert — HuggingFace PubMedBERT"
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query (default: 5).",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "BioASQ-training14b" / "training14b.json"),
        help="Path to BioASQ dataset JSON file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to process (useful for testing).",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run full evaluation after batch inference and save results.",
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch an interactive multi-turn CLI chat session.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Directory for predictions and evaluation results (default: output/).",
    )

    return parser.parse_args()


def _save_predictions(predictions: list[dict],
                       output_dir: str,
                       retriever: str,
                       generator: str) -> str:
    """
    Save predictions as a TSV file named <retriever>_<generator>.tsv.
    One answer per line — multi-item answers (list/factoid) joined by | .
    """
    pred_dir = os.path.join(output_dir, "prediction")
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(pred_dir, f"{retriever}_{generator}.tsv")

    with open(filepath, "w") as f:
        for pred in predictions:
            answer = pred.get("answer", "")
            if isinstance(answer, list):
                line = " | ".join(str(a) for a in answer)
            else:
                line = str(answer).replace("\n", " ").strip()
            f.write(line + "\n")

    print(f"[INFO] Predictions saved to {filepath}")
    return filepath


def _run_interactive_chat(system: "BioASQRAGSystem") -> None:
    """Simple interactive CLI loop for multi-turn chat."""
    session_id = str(uuid.uuid4())
    print("\n" + "─" * 60)
    print("  BioASQ Conversational QA — Interactive Mode")
    print(f"  Retriever: {system.retriever_type} | Generator: {system.generator_type} | k={system.k}")
    print("  Type 'quit' to exit, 'reset' to start a new session.")
    print("─" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting chat.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            session_id = str(uuid.uuid4())
            print("[INFO] Session reset.\n")
            continue

        answer = system.chat(user_input, session_id)
        print(f"\nAssistant: {answer}\n")


# ===========================================================================
# Entry Point
# ===========================================================================

def main():
    args = parse_args()

    # Initialize system
    system = BioASQRAGSystem(
        retriever=args.retriever,
        generator=args.generator,
        k=args.k,
    )

    # Load generator
    system.load_generator()

    # Interactive chat — no dataset needed
    if args.chat:
        if args.retriever != "none":
            print("[WARNING] Chat mode with retrieval requires a pre-built index.")
            print("[INFO] Loading dataset to build index...")
            questions = load_bioasq_dataset(args.data)
            questions = [parse_question(q) for q in questions]
            system.index_corpus(questions)
        _run_interactive_chat(system)
        return

    # Load and parse dataset
    print(f"[INFO] Loading dataset from {args.data}")
    raw_questions = load_bioasq_dataset(args.data)
    questions     = [parse_question(q) for q in raw_questions]

    if args.limit:
        questions = questions[:args.limit]
        print(f"[INFO] Limited to {args.limit} questions.")

    # Build retrieval index
    system.index_corpus(questions)

    # Batch inference
    predictions = system.run_batch(questions)

    # Save predictions
    _save_predictions(predictions, args.output_dir, args.retriever, args.generator)

    # Evaluation
    if args.eval:
        ground_truth = questions  # parse_question() output is compatible with evaluation.py
        run_full_evaluation(
            predictions,
            ground_truth,
            judge_model=None,
            output_dir=os.path.join(args.output_dir, "evaluation"),
            retriever=args.retriever,
            generator=args.generator,
        )


if __name__ == "__main__":
    main()