import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src/utils"))
from data_utils import load_bioasq_dataset

# Load ground truth
questions = load_bioasq_dataset("data/BioASQ-training14b/training14b.json")
gold_map = {q["id"]: q.get("documents", []) for q in questions}
question_map = {q["id"]: q for q in questions}

# Load reranked results
with open("output/retrieval_results_reranked.json") as f:
    results = json.load(f)

# Find a good list question example
for item in results:
    qid = item["question_id"]
    q = question_map.get(qid)
    if q and q.get("type") == "list":
        query = q.get("body")
        gold_docs = gold_map.get(qid, [])
        retrieved = item["results"][:5]

        print(f"Question: {query}")
        print(f"Type: {q.get('type')}")
        print(f"\nTop 5 Retrieved Documents:")
        for i, doc in enumerate(retrieved, 1):
            pmid = doc["doc_id"].replace("pubmed_", "")
            gold_url = f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
            hit = "✅ HIT" if gold_url in gold_docs else "❌ MISS"
            print(f"  {i}. PMID {pmid} — {hit}")
            print(f"     {doc['text'][:100]}...")

        print(f"\nGold Documents: {len(gold_docs)} total")
        break
