# Retrieval functions (Joel)
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
import numpy as np
import requests
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Allow imports from sibling files (data_utils.py by Lowami)
sys.path.append(os.path.dirname(__file__))
from data_utils import get_snippets


# ===========================================================================
# BM25 (Sparse Retrieval)
# ===========================================================================

def build_bm25_index(corpus: list[dict],
                     config: dict = None) -> BM25Okapi:
    """
    Build a BM25 index over the corpus.
    corpus: [{"doc_id": ..., "text": ..., "pmid": ...}]
    """
    # Tokenize each document by splitting on whitespace (simple but effective)
    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
    index = BM25Okapi(tokenized_corpus)
    return index


def bm25_retrieve(query: str,
                  index: BM25Okapi,
                  corpus: list[dict],
                  top_k: int = 10,
                  config: dict = None) -> list[dict]:
    """
    Retrieve top-K documents using BM25 sparse matching.
    Returns: [{"doc_id", "text", "pmid", "score"}]
    """
    # Tokenize the query the same way as the corpus

    top_k = top_k or (config or {}).get("retrieval", {}).get("top_k", 10)

    tokenized_query = query.lower().split()

    # Get BM25 scores for all documents
    scores = index.get_scores(tokenized_query)

    # Get indices of top_k highest scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        doc = corpus[idx]
        results.append({
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "pmid": doc["pmid"],
            "score": float(scores[idx])
        })

    return results


# ===========================================================================
# PubMed Fetching & Corpus Building
# ===========================================================================

import xml.etree.ElementTree as ET

def fetch_pubmed_abstract(pmid: str) -> dict:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "abstract",
        "retmode": "xml"
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.text)

        # Extract title
        title_el = root.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Concatenate all AbstractText sections (handles structured abstracts
        # with BACKGROUND / METHODS / RESULTS / CONCLUSIONS labels)
        abstract_parts = []
        for el in root.findall(".//AbstractText"):
            label = el.get("Label")
            text = "".join(el.itertext()).strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts)

        time.sleep(0.34)
        return {"pmid": pmid, "title": title, "abstract": abstract}

    except ET.ParseError as e:
        print(f"[WARNING] XML parse error for PMID {pmid}: {e}")
        return {"pmid": pmid, "title": "", "abstract": ""}
    except Exception as e:
        print(f"[WARNING] Could not fetch PMID {pmid}: {e}")
        return {"pmid": pmid, "title": "", "abstract": ""}


def build_corpus_from_bioasq(questions: list[dict]) -> list[dict]:
    corpus = []
    seen_chunk_ids = set()

    for question in questions:
        snippets = get_snippets(question)

        for snippet in snippets:
            doc_url = snippet.get("document", "")
            pmid = doc_url.rstrip("/").split("/")[-1]

            if not pmid:
                continue

            snippet_text = snippet.get("text", "").strip()
            # Use offset to create a unique chunk ID per snippet
            begin = snippet.get("begin", 0)
            end = snippet.get("end", 0)
            chunk_id = f"pubmed_{pmid}_{begin}_{end}"

            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            if snippet_text:
                corpus.append({
                    "doc_id": chunk_id,
                    "text": snippet_text,
                    "pmid": pmid
                })
            else:
                fetched = fetch_pubmed_abstract(pmid)
                full_text = fetched["abstract"] or fetched["title"]
                if full_text:
                    corpus.append({
                        "doc_id": chunk_id,
                        "text": full_text,
                        "pmid": pmid
                    })

    print(f"[INFO] Corpus built: {len(corpus)} chunks from {len(questions)} questions.")
    return corpus


# ===========================================================================
# Dense Retrieval (BGE-M3 Embeddings)
# ===========================================================================

def build_dense_index(corpus: list[dict],
                      model_name: str = None,
                      config: dict = None) -> tuple:
    """
    Encode corpus with a biomedical embedding model.
    Returns: (embeddings_matrix, encoder_model)

    Note: On machines with limited RAM, use "BAAI/bge-small-en-v1.5" instead.
    """
    if not corpus:
        raise ValueError("[ERROR] Cannot build dense index from empty corpus")
    cfg = (config or {}).get("retrieval", {})
    model_name = model_name or cfg.get("dense_model", "BAAI/bge-m3")
    batch_size = cfg.get("batch_size", 32)

    print(f"[INFO] Loading embedding model: {model_name}")
    encoder = SentenceTransformer(model_name)

    texts = [doc["text"] for doc in corpus]

    print(f"[INFO] Encoding {len(texts)} documents...")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # makes cosine similarity = dot product
    )

    print(f"[INFO] Dense index built. Shape: {embeddings.shape}")
    return embeddings, encoder


def dense_retrieve(query: str,
                   embeddings,
                   encoder,
                   corpus: list[dict],
                   top_k: int = 10) -> list[dict]:
    """
    Retrieve top-K documents via cosine similarity over dense embeddings.
    Returns: [{"doc_id", "text", "pmid", "score"}]
    """
    # Encode the query
    query_embedding = encoder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Compute cosine similarity between query and all documents
    # Since embeddings are normalized, dot product = cosine similarity
    scores = cosine_similarity(query_embedding, embeddings)[0]

    # Get top_k indices sorted by score (highest first)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        doc = corpus[idx]
        results.append({
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "pmid": doc["pmid"],
            "score": float(scores[idx])
        })

    return results


def save_dense_index(embeddings: np.ndarray, 
                     filepath: str) -> None:
    """Save embeddings matrix to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, embeddings)
    print(f"[INFO] Dense index saved to {filepath}")


def load_dense_index(filepath: str, 
                     model_name: str) -> tuple:
    """Load embeddings from disk and reinitialize the encoder."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No dense index found at {filepath}")
    embeddings = np.load(filepath)
    encoder = SentenceTransformer(model_name)
    print(f"[INFO] Dense index loaded from {filepath}. Shape: {embeddings.shape}")
    return embeddings, encoder


def save_bm25_index(index: BM25Okapi, filepath: str) -> None:
    """Pickle the BM25 index to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(index, f)
    print(f"[INFO] BM25 index saved to {filepath}")


def load_bm25_index(filepath: str) -> BM25Okapi:
    """Load a pickled BM25 index from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No BM25 index found at {filepath}")
    with open(filepath, "rb") as f:
        index = pickle.load(f)
    print(f"[INFO] BM25 index loaded from {filepath}")
    return index


# ===========================================================================
# Reciprocal Rank Fusion (RRF)
# ===========================================================================

def reciprocal_rank_fusion(bm25_results: list[dict],
                           dense_results: list[dict],
                           k: int = None,
                           config: dict = None) -> list[dict]:
    """
    Merge sparse and dense results using RRF scoring.
    RRF formula: score(d) = sum( 1 / (k + rank(d)) ) across both lists.
    Returns: reranked list of [{"doc_id", "text", "pmid", "rrf_score"}]
    """
    k = k or (config or {}).get("retrieval", {}).get("rrf_k", 60)
    rrf_scores = {}
    doc_store = {}  # keep text/pmid for each doc_id

    # Score BM25 results by rank
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["doc_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_store[doc_id] = {"text": doc["text"], "pmid": doc["pmid"]}

    # Score dense results by rank
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["doc_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_store[doc_id] = {"text": doc["text"], "pmid": doc["pmid"]}

    # Sort all docs by their combined RRF score (highest first)
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, rrf_score in sorted_docs:
        results.append({
            "doc_id": doc_id,
            "text": doc_store[doc_id]["text"],
            "pmid": doc_store[doc_id]["pmid"],
            "rrf_score": rrf_score
        })

    return results


# ===========================================================================
# Cross-Encoder Reranking
# ===========================================================================

def rerank_with_crossencoder(query: str,
                              candidates: list[dict],
                              model_name: str = None,
                              top_k: int = 5,
                              config: dict = None) -> list[dict]:
    """
    Apply cross-encoder reranking on fused candidates.
    The cross-encoder scores each (query, document) pair jointly —
    more accurate than bi-encoder cosine similarity but slower.
    Returns: final top-K [{"doc_id", "text", "pmid", "rerank_score"}]

    Recommended model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    For biomedical: "cross-encoder/nli-MiniLM2-L6-H768"
    """
    cfg = (config or {}).get("retrieval", {})
    model_name = model_name or cfg.get("crossencoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k = top_k or cfg.get("rerank_top_k", 5)

    cross_encoder = CrossEncoder(model_name)

    # Build (query, document_text) pairs for the cross-encoder
    pairs = [(query, doc["text"]) for doc in candidates]

    # Score all pairs
    scores = cross_encoder.predict(pairs)

    # Attach scores to candidates
    scored_candidates = []
    for doc, score in zip(candidates, scores):
        scored_candidates.append({
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "pmid": doc["pmid"],
            "rerank_score": float(score)
        })

    # Sort by rerank score and return top_k
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored_candidates[:top_k]

# ===========================================================================
# Parallel Batching
# ===========================================================================

def process_single_question(args):
    q, bm25_index, embeddings, encoder, cross_encoder, corpus, top_k = args
    query = q.get("body", "")
    q_id = q.get("id", "")

    bm25_res = bm25_retrieve(query, bm25_index, corpus, top_k=top_k)
    dense_res = dense_retrieve(query, embeddings, encoder, corpus, top_k=top_k)
    hybrid_res = reciprocal_rank_fusion(bm25_res, dense_res)

    pairs = [(query, doc["text"]) for doc in hybrid_res[:top_k]]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(
        [{"doc_id": d["doc_id"], "text": d["text"], "pmid": d["pmid"], "rerank_score": float(s)}
         for d, s in zip(hybrid_res[:top_k], scores)],
        key=lambda x: x["rerank_score"], reverse=True
    )

    return {"question_id": q_id, "results": reranked}


def batch_retrieve_parallel(questions, bm25_index, embeddings, encoder, cross_encoder, corpus, top_k=10, max_workers=8):
    args_list = [
        (q, bm25_index, embeddings, encoder, cross_encoder, corpus, top_k)
        for q in questions
    ]
    results = [None] * len(questions)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_question, args): i
            for i, args in enumerate(args_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[ERROR] Question {idx} failed: {e}")
                results[idx] = {"question_id": questions[idx].get("id", ""), "results": []}

    return results

# ===========================================================================
# Main Runner
# ===========================================================================

if __name__ == "__main__":
    import json
    sys.path.append(os.path.dirname(__file__))
    from data_utils import load_bioasq_dataset

    DATA_PATH = "data/BioASQ-training14b/training14b.json"
    OUTPUT_PATH = "output/retrieval_results.json"
    os.makedirs("output", exist_ok=True)

    print("[INFO] Loading BioASQ data...")
    questions = load_bioasq_dataset(DATA_PATH)
    print(f"[INFO] Loaded {len(questions)} questions.")

    corpus = build_corpus_from_bioasq(questions)
    bm25_index = build_bm25_index(corpus)
    embeddings, encoder = build_dense_index(corpus, model_name="BAAI/bge-m3")

    print("[INFO] Running hybrid retrieval...")
    print("[INFO] Loading cross-encoder model...")
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("[INFO] Running parallel hybrid retrieval...")
    results = batch_retrieve_parallel(
        questions, bm25_index, embeddings, encoder, cross_encoder, corpus,
        top_k=10, max_workers=8
    )
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)
    print(f"[INFO] Done! Results saved to {OUTPUT_PATH}")
