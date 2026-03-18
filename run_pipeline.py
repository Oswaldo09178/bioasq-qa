import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "src/utils"))
from data_utils import load_bioasq_dataset, get_snippets
from retrieval_utils import (
    build_bm25_index, bm25_retrieve,
    build_corpus_from_bioasq, build_dense_index,
    dense_retrieve, reciprocal_rank_fusion,
    rerank_with_crossencoder
)

DATA_PATH = "data/BioASQ-training14b/training14b.json"
OUTPUT_PATH = "output/retrieval_results_reranked.json"
os.makedirs("output", exist_ok=True)

print("[INFO] Loading BioASQ data...")
questions = load_bioasq_dataset(DATA_PATH)
print(f"[INFO] Loaded {len(questions)} questions.")

print("[INFO] Building corpus...")
corpus = build_corpus_from_bioasq(questions)

print("[INFO] Building BM25 index...")
bm25_index = build_bm25_index(corpus)

print("[INFO] Building dense index with BGE-M3...")
embeddings, encoder = build_dense_index(corpus, model_name="BAAI/bge-m3")

print("[INFO] Running hybrid retrieval + cross-encoder reranking...")
results = []
for i, q in enumerate(questions):
    query = q.get("body", "")
    bm25_res = bm25_retrieve(query, bm25_index, corpus, top_k=10)
    dense_res = dense_retrieve(query, embeddings, encoder, corpus, top_k=10)
    hybrid_res = reciprocal_rank_fusion(bm25_res, dense_res)
    reranked = rerank_with_crossencoder(
        query, hybrid_res,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=10
    )
    results.append({"question_id": q.get("id"), "results": reranked})
    if i % 100 == 0:
        print(f"[INFO] Processed {i}/{len(questions)} questions...")

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f)
print(f"[INFO] Done! Results saved to {OUTPUT_PATH}")
