# Retrieval functions (Joel)

import sys
import os
import numpy as np
import requests
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Allow imports from sibling files (data_utils.py by Lowami)
sys.path.append(os.path.dirname(__file__))
from data_utils import get_snippets


# ===========================================================================
# BM25 (Sparse Retrieval)
# ===========================================================================

def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
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
                  top_k: int = 10) -> list[dict]:
    """
    Retrieve top-K documents using BM25 sparse matching.
    Returns: [{"doc_id", "text", "pmid", "score"}]
    """
    # Tokenize the query the same way as the corpus
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

def fetch_pubmed_abstract(pmid: str) -> dict:
    """
    Fetch abstract from PubMed via E-utilities API given a PMID.
    Returns: {"pmid", "title", "abstract"}
    """
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
        xml_text = response.text

        # Extract title
        title = ""
        if "<ArticleTitle>" in xml_text and "</ArticleTitle>" in xml_text:
            start = xml_text.index("<ArticleTitle>") + len("<ArticleTitle>")
            end = xml_text.index("</ArticleTitle>")
            title = xml_text[start:end].strip()

        # Extract abstract text
        abstract = ""
        if "<AbstractText>" in xml_text and "</AbstractText>" in xml_text:
            start = xml_text.index("<AbstractText>") + len("<AbstractText>")
            end = xml_text.index("</AbstractText>")
            abstract = xml_text[start:end].strip()

        # Be polite to NCBI — max 3 requests per second without API key
        time.sleep(0.34)

        return {"pmid": pmid, "title": title, "abstract": abstract}

    except Exception as e:
        print(f"[WARNING] Could not fetch PMID {pmid}: {e}")
        return {"pmid": pmid, "title": "", "abstract": ""}


def build_corpus_from_bioasq(questions: list[dict]) -> list[dict]:
    """
    Extract all unique documents from BioASQ questions
    and fetch their abstracts to build the retrieval corpus.
    Uses Lowami's get_snippets() to extract snippet texts per question.
    """
    corpus = []
    seen_pmids = set()

    for question in questions:
        # Use Lowami's get_snippets to extract snippet data
        snippets = get_snippets(question)

        for snippet in snippets:
            # Extract PMID from the document URL
            # BioASQ document URLs look like: http://www.ncbi.nlm.nih.gov/pubmed/12345678
            doc_url = snippet.get("document", "")
            pmid = doc_url.rstrip("/").split("/")[-1]

            if not pmid or pmid in seen_pmids:
                continue

            seen_pmids.add(pmid)

            # Try to use the snippet text directly first (avoids API calls)
            snippet_text = snippet.get("text", "").strip()

            if snippet_text:
                # We already have the text from BioASQ — use it directly
                corpus.append({
                    "doc_id": f"pubmed_{pmid}",
                    "text": snippet_text,
                    "pmid": pmid
                })
            else:
                # Fall back to fetching from PubMed API
                fetched = fetch_pubmed_abstract(pmid)
                full_text = fetched["abstract"] or fetched["title"]

                if full_text:
                    corpus.append({
                        "doc_id": f"pubmed_{pmid}",
                        "text": full_text,
                        "pmid": pmid
                    })

    print(f"[INFO] Corpus built: {len(corpus)} unique documents from {len(questions)} questions.")
    return corpus


# ===========================================================================
# Dense Retrieval (BGE-M3 Embeddings)
# ===========================================================================

def build_dense_index(corpus: list[dict],
                      model_name: str = "BAAI/bge-m3") -> tuple:
    """
    Encode corpus with a biomedical embedding model.
    Returns: (embeddings_matrix, encoder_model)

    Note: On machines with limited RAM, use "BAAI/bge-small-en-v1.5" instead.
    """
    print(f"[INFO] Loading embedding model: {model_name}")
    encoder = SentenceTransformer(model_name)

    texts = [doc["text"] for doc in corpus]

    print(f"[INFO] Encoding {len(texts)} documents...")
    embeddings = encoder.encode(
        texts,
        batch_size=32,
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


# ===========================================================================
# Reciprocal Rank Fusion (RRF)
# ===========================================================================

def reciprocal_rank_fusion(bm25_results: list[dict],
                           dense_results: list[dict],
                           k: int = 60) -> list[dict]:
    """
    Merge sparse and dense results using RRF scoring.
    RRF formula: score(d) = sum( 1 / (k + rank(d)) ) across both lists.
    Returns: reranked list of [{"doc_id", "text", "pmid", "rrf_score"}]
    """
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
                              model_name: str,
                              top_k: int = 5) -> list[dict]:
    """
    Apply cross-encoder reranking on fused candidates.
    The cross-encoder scores each (query, document) pair jointly —
    more accurate than bi-encoder cosine similarity but slower.
    Returns: final top-K [{"doc_id", "text", "pmid", "rerank_score"}]

    Recommended model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    For biomedical: "cross-encoder/nli-MiniLM2-L6-H768"
    """
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