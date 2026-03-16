import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "src/utils"))

from data_utils import load_bioasq_dataset, get_snippets
from retrieval_utils import (
    build_bm25_index, bm25_retrieve,
    build_corpus_from_bioasq, build_dense_index,
    dense_retrieve, reciprocal_rank_fusion
)

DATA_PATH = "data/BioASQ-training14b/training14b.json"
OUTPUT_PATH = "output/retrieval_results.json"
os.makedirs("output", exist_ok=True)

# Load questions
print("[INFO] Loading BioASQ data...")
questions = load_bioasq_dataset(DATA_PATH)
print(f"[INFO] Loaded {len(questions)} questions.")

# Build corpus
print("[INFO] Building corpus...")
corpus = build_corpus_from_bioasq(questions)

# BM25 index
print("[INFO] Building BM25 index...")
bm25_index = build_bm25_index(corpus)

# Dense index with full BGE-M3
print("[INFO] Building dense index with BGE-M3...")
embeddings, encoder = build_dense_index(corpus, model_name="BAAI/bge-m3")

# Run hybrid retrieval on all questions
print("[INFO] Running hybrid retrieval...")
results = []
for i, q in enumerate(questions):
    query = q.get("body", "")
    bm25_res = bm25_retrieve(query, bm25_index, corpus, top_k=10)
    dense_res = dense_retrieve(query, embeddings, encoder, corpus, top_k=10)
    hybrid_res = reciprocal_rank_fusion(bm25_res, dense_res)
    results.append({"question_id": q.get("id"), "results": hybrid_res[:10]})
    if i % 100 == 0:
        print(f"[INFO] Processed {i}/{len(questions)} questions...")

# Save results
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f)
print(f"[INFO] Done! Results saved to {OUTPUT_PATH}")
