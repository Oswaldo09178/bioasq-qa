import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src/utils"))
from data_utils import load_bioasq_dataset
from retrieval_utils import build_bm25_index, bm25_retrieve, build_corpus_from_bioasq

def hits(retrieved, gold_docs):
    count = 0
    for doc in retrieved:
        pmid = doc["doc_id"].replace("pubmed_", "")
        if f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" in gold_docs:
            count += 1
    return count

questions = load_bioasq_dataset("data/BioASQ-training14b/training14b.json")
gold_map = {q["id"]: q.get("documents", []) for q in questions}
question_map = {q["id"]: q for q in questions}

with open("output/retrieval_results.json") as f:
    hybrid_results = {item["question_id"]: item["results"] for item in json.load(f)}

with open("output/retrieval_results_reranked.json") as f:
    reranked_results = {item["question_id"]: item["results"] for item in json.load(f)}

# Target question
TARGET = "Which are the inhibitors of histone methyltransferases?"
target_q = next(q for q in questions if q.get("body") == TARGET)
qid = target_q["id"]
gold_docs = gold_map[qid]

# Build BM25
corpus = build_corpus_from_bioasq(questions)
bm25_index = build_bm25_index(corpus)
bm25_res = bm25_retrieve(TARGET, bm25_index, corpus, top_k=5)

def print_results(label, results, gold_docs):
    h = hits(results, gold_docs)
    print(f"\n{label} — {h}/10 hits")
    for i, doc in enumerate(results[:10], 1):
        pmid = doc["doc_id"].replace("pubmed_", "")
        hit = "✅" if f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" in gold_docs else "❌"
        print(f"  {i}. PMID {pmid} {hit} — {doc['text'][:80]}...")

print(f"Question: {TARGET}")
print_results("BM25 Only", bm25_res, gold_docs)
print_results("Hybrid (BM25 + BGE-M3)", hybrid_results[qid], gold_docs)
print_results("Hybrid + Cross-Encoder", reranked_results[qid], gold_docs)
