import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src/utils"))
from data_utils import load_bioasq_dataset
from retrieval_utils import build_bm25_index, bm25_retrieve, build_corpus_from_bioasq

def average_precision(retrieved_doc_ids, gold_doc_ids, k=10):
    gold_set = set(gold_doc_ids)
    hits = 0
    score = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        pmid = doc_id.replace("pubmed_", "")
        if any(pmid in g for g in gold_set):
            hits += 1
            score += hits / i
    return score / min(len(gold_set), k) if gold_set else 0.0

questions = load_bioasq_dataset("data/BioASQ-training14b/training14b.json")
gold_map = {q["id"]: q.get("documents", []) for q in questions}
type_map = {q["id"]: q.get("type", "unknown") for q in questions}

corpus = build_corpus_from_bioasq(questions)
bm25_index = build_bm25_index(corpus)

ap_scores = []
type_scores = {"yesno": [], "factoid": [], "list": [], "summary": []}

for q in questions:
    query = q.get("body", "")
    retrieved = [r["doc_id"] for r in bm25_retrieve(query, bm25_index, corpus, top_k=10)]
    gold_docs = gold_map.get(q["id"], [])
    qtype = type_map.get(q["id"], "unknown")
    ap = average_precision(retrieved, gold_docs, k=10)
    ap_scores.append(ap)
    if qtype in type_scores:
        type_scores[qtype].append(ap)

print(f"Overall MAP@10: {sum(ap_scores)/len(ap_scores):.4f}")
for qtype, scores in type_scores.items():
    if scores:
        print(f"  {qtype:10s}: {sum(scores)/len(scores):.4f}")
