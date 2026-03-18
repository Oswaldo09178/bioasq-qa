import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src/utils"))
from data_utils import load_bioasq_dataset

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

# Load ground truth
print("[INFO] Loading ground truth...")
questions = load_bioasq_dataset("data/BioASQ-training14b/training14b.json")
gold_map = {q["id"]: q.get("documents", []) for q in questions}
type_map = {q["id"]: q.get("type", "unknown") for q in questions}

# Load retrieval results
print("[INFO] Loading retrieval results...")
with open("output/retrieval_results.json") as f:
    results = json.load(f)

# Compute MAP@10 overall and per question type
ap_scores = []
type_scores = {"yesno": [], "factoid": [], "list": [], "summary": []}

for item in results:
    qid = item["question_id"]
    retrieved = [r["doc_id"] for r in item["results"]]
    gold_docs = gold_map.get(qid, [])
    qtype = type_map.get(qid, "unknown")
    ap = average_precision(retrieved, gold_docs, k=10)
    ap_scores.append(ap)
    if qtype in type_scores:
        type_scores[qtype].append(ap)

map10 = sum(ap_scores) / len(ap_scores)
print(f"\n[RESULT] Overall MAP@10 on full 5,729 questions: {map10:.4f}")
print(f"[RESULT] Total questions evaluated: {len(ap_scores)}")
print(f"\n[RESULT] MAP@10 by question type:")
for qtype, scores in type_scores.items():
    if scores:
        print(f"  {qtype:10s}: {sum(scores)/len(scores):.4f}  ({len(scores)} questions)")
