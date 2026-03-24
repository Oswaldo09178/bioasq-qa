# scripts/test_multiturn.py
import sys, json
sys.path.append("src/utils")
sys.path.append("src")

from rag_system import BioASQRAGSystem
from evaluation import run_full_evaluation
from synthetic_data_utils import load_synthetic_dataset

system = BioASQRAGSystem(retriever="hybrid", generator="gemini", k=5)
system.load_generator()
system.index_corpus([])  # indices already built

dialogues    = load_synthetic_dataset("data/synthetic/dialogues.json")
predictions  = []

for dialogue in dialogues:
    session_id = dialogue["source_id"]   # one session per dialogue
    for turn in dialogue["turns"]:
        if turn["role"] != "user":       # skip assistant turns
            continue
        question = {
            "id":           f"{session_id}_turn{turn['turn_id']}",
            "body":         turn["query"],
            "type":         dialogue["question_type"],
            "snippets":     dialogue["snippets"],
        }
        pred = system.answer(
            question,
            session_id=session_id,
            gold_answer=turn["answer"],
        )
        pred["requires_context"] = turn["requires_context"]
        predictions.append(pred)

run_full_evaluation(predictions, [], output_dir="output/evaluation/multiturn/")