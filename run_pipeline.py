import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "src/utils"))
from sentence_transformers import CrossEncoder
from data_utils import load_bioasq_dataset, get_snippets
from retrieval_utils import (
    build_bm25_index, bm25_retrieve,
    build_corpus_from_bioasq, build_dense_index,
    dense_retrieve, reciprocal_rank_fusion,
    rerank_with_crossencoder, batch_retrieve_parallel
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


print("[INFO] Loading cross-encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("[INFO] Running parallel hybrid retrieval + reranking...")
results = batch_retrieve_parallel(
    questions, bm25_index, embeddings, encoder, cross_encoder, corpus,
    top_k=10, max_workers=8
)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f)
print(f"[INFO] Done! Results saved to {OUTPUT_PATH}")
