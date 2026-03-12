
# Synthetic Data Utilities -- Lowami
from email.mime import text

from google.cloud import storage
from google.oauth2 import service_account
import json
import google.generativeai as genai
from dotenv import load_dotenv
from data_utils import load_bioasq_dataset, parse_question
import os
load_dotenv()

# def convert_to_multiturn(question: dict, 
#                          llm, 
#                          num_turns: int = 3) -> dict:
#     """
#     Takes a single BioASQ question and generates a multi-turn 
#     dialogue around it using self-instruct prompting.
    
#     Input:  single BioASQ question dict (body, snippets, ideal_answer)
#     Output: {
#         "source_id": str,         # original BioASQ question id
#         "question_type": str,
#         "turns": [
#             {"turn_id": 1, "query": str, "answer": str, 
#              "requires_context": False},
#             {"turn_id": 2, "query": str, "answer": str,   # anaphoric
#              "requires_context": True},
#             ...
#         ],
#         "snippets": list[dict]    # shared snippet pool from original
#     }
#     """

# def generate_followup_question(original_question: str,
#                                 previous_answer: str,
#                                 snippets: list[dict],
#                                 llm,
#                                 anaphoric: bool = True) -> str:
#     """
#     Generate a follow-up question based on the previous answer.
#     If anaphoric=True, inject references like 'that gene', 'this drug',
#     'those mutations' to simulate natural clinical dialogue.
#     """

# def generate_followup_answer(followup_question: str,
#                               conversation_history: list[dict],
#                               snippets: list[dict],
#                               llm) -> str:
#     """
#     Generate a grounded answer to a follow-up question,
#     using the shared snippet pool as the only evidence source.
#     """

# def build_synthetic_dialogue_prompt(question: dict,
#                                      num_turns: int) -> str:
#     """
#     Build the self-instruct prompt sent to the LLM to generate
#     a full multi-turn dialogue from a BioASQ entry.
#     This is the core CoQA-style synthesis prompt.
#     """

# def validate_synthetic_turn(turn: dict, 
#                               snippets: list[dict]) -> bool:
#     """
#     Quality check for each generated turn:
#     - Answer is grounded in snippets
#     - Anaphoric references are resolvable from prior turns
#     - Turn is not a repeat of a previous question
#     Returns True if the turn passes all checks.
#     """

# def generate_synthetic_dataset(questions: list[dict],
#                                 llm,
#                                 num_turns: int = 4,
#                                 output_path: str = "data/synthetic/") -> list[dict]:
#     """
#     Full pipeline: iterates over BioASQ questions, generates
#     multi-turn dialogues, validates each, and saves to disk.
#     Targets the 100-scenario Biomedical Dialogue Set from the proposal.
#     """

# def save_synthetic_dataset(dialogues: list[dict], 
#                             filepath: str) -> None:
#     """Serialize synthetic dialogues to JSON."""

# def load_synthetic_dataset(filepath: str) -> list[dict]:
#     """Load previously generated synthetic dialogues."""

# def get_synthetic_dataset_stats(dialogues: list[dict]) -> dict:
#     """
#     Sanity-check report on the generated dataset.
#     Returns: {
#         "total_dialogues": int,
#         "avg_turns": float,
#         "anaphoric_turn_ratio": float,    # % of turns requiring context
#         "question_type_distribution": dict
#     }
#     """
credentials = service_account.Credentials.from_service_account_file("service_key.json")

def prepare_llama_batch_file(input_records, output_path):
    with open(output_path, "w") as f:
        for i, record in enumerate(input_records):
            question = record['body']
            answer = record.get('ideal_answer') or record.get('exact_answer') 
            context = " ".join([s['text'] for s in record['snippets']])
            prompt = f"""
            Convert to 3-turn CoQA: 
            question: {question} - answer: {answer} - context: {context}
            INSTRUCTIONS:
            If the question asks to list items,
            expand the question and then contract it using only the knowledge provided.
            For example:
                Question: List signaling molecules (ligands) that interact with the receptor EGFR?
                Subquestion 1: What is the primary mechanism for receptor activation, and is it strictly limited to EGF?
                Subanswer 1: EGFR activation occurs via ligand binding to the extracellular domain, inducing dimerization. It is not exclusive to EGF; the receptor is pleiotropic and interacts with several ligands within the EGF peptide family.
                Sub question 2: Do these ligands exhibit differential binding affinities or tissue-specific clinical relevance?
                Subanswer 2: Yes. Affinity typically follows the hierarchy: EGF > HB-EGF > TGF-α > BTC > EPR > EPG > AR. Clinically, TGF-α is frequently overexpressed in triple-negative breast cancer, while AREG and HB-EGF are utilized as biomarkers in chemorefractory mCRC and airway epithelial pathologies.
                Original answer: The 7 known EGFR ligands  are: epidermal growth factor (EGF), betacellulin (BTC), epiregulin (EPR), heparin-binding EGF (HB-EGF), transforming growth factor-\u03b1 [TGF-\u03b1], amphiregulin (AREG) and epigen (EPG).
                Avoid fake questions like: What is the receptor that interacts with certain signaling molecules?
            """
            line = {
                "custom_id": f"{record['id']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "llama-4-maverick-17b-128e-instruct-maas",
                    "messages": [
                        {"role": "user", "content": f"{prompt}"}
                    ]
                }
            }
            f.write(json.dumps(line) + "\n")

def get_latest_prediction_prefix(bucket, base_prefix="output/"):
    """Find the most recent prediction folder by sorting folder names."""
    blobs = bucket.list_blobs(prefix=base_prefix, delimiter="/")
    
    _ = list(blobs)
    
    prediction_folders = [
        p for p in blobs.prefixes 
        if "prediction-model-" in p
    ]
    
    if not prediction_folders:
        raise ValueError("No prediction folders found in bucket.")
    
    latest = sorted(prediction_folders)[-1]
    print(f"Latest prediction folder: {latest}")
    return latest


def get_batch_outputs(bucket_name, base_prefix="output/"):
    storage_client = storage.Client(project="agentic-486120", credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    output_prefix = get_latest_prediction_prefix(bucket, base_prefix)

    blobs = bucket.list_blobs(prefix=output_prefix)
    
    all_results = []
    for blob in blobs:
        if blob.name.endswith(".jsonl") and blob.size > 0:
            print(f"Reading: {blob.name} ({blob.size} bytes)")
            content = blob.download_as_text()
            for line in content.splitlines():
                if line.strip():
                    all_results.append(json.loads(line))
    
    print(f"Fetched {len(all_results)} results.")
    return all_results


def save_llama_results(llama_results, output_file="data/corpus/llama_results.jsonl"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for item in llama_results:
            f.write(json.dumps(item) + "\n")

def create_judge_input(llama_results, output_file="judge_input.jsonl"):
    with open(output_file, "w") as f:
        for item in llama_results:
            # Llama outputs usually put the answer in 'response' or 'content'
            # depending on the exact API used. Check your 'results' list first.
            llama_output = item.get('response', {}).get('content', '')
            
            # Construct the Judge Prompt
            judge_prompt = (
                f"Llama generated this CoQA output: {llama_output}. "
                "Rate its accuracy from 1-10 based on medical truth."
            )
            
            # Gemini-specific JSONL structure
            gemini_line = {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": judge_prompt}]}],
                    "generationConfig": {"temperature": 0.0}
                }
            }
            f.write(json.dumps(gemini_line) + "\n")

def judge_conversation(model, original_q, followups):

    prompt = build_judge_prompt(original_q, followups)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0,
            "top_p": 1,
        }
    )

    text = response.text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("Invalid JSON from judge:")
        print(text)
    return None


def build_judge_prompt(original_q, followups):

    return f"""
You are an expert evaluator of medical conversational question decomposition.

You are given:

Original Question - Answer Pairs:
{original_q}

Generated Follow-up Questions:
{followups}

For each question, evaluate the follow-up questions according to:

1. Relevance (1-5)
2. Decomposition Quality (1-5)
3. Non-Redundancy (1-5)
4. Guidance Utility (1-5)
5. Logical Ordering (1-5)
Report the average scores accross all questions and use them to compute
overall score (1-5)

Definitions:
- Relevance: Direct connection to original question.
- Decomposition: Breaks reasoning into meaningful substeps.
- Non-Redundancy: Not restating original question.
- Guidance Utility: Would help a weaker model reach the answer.
- Logical Ordering: Follows a reasoning progression.

Return ONLY valid JSON in this format:

{{
  "relevance": int,
  "decomposition": int,
  "non_redundancy": int,
  "guidance": int,
  "logical_ordering": int,
  "overall": float,
  "justification": string
}}
"""

def run() -> None:
    bucket_name = "bioasq-bucket"
    llama_results = get_batch_outputs(bucket_name)
    save_llama_results(llama_results)
    create_judge_input(llama_results)
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")
    questions = load_bioasq_dataset("data/BioASQ-training14b/training14b.json")
    processed_qs = [parse_question(q, minimized=True) for q in questions[:10]]
    metrics = judge_conversation(model, processed_qs[:10], llama_results[:10])
    print("Evaluation Metrics: ", metrics)
    return metrics

if __name__ == "__main__":
    run()
