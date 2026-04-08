from typing import Optional, Union, List, Dict, Tuple, Any
# Synthetic Data Utilities — Lowami
#
# Pipeline overview:
#   1. prepare_llama_batch_file()   — build JSONL batch input for Llama 4 (GCS/Vertex)
#   2. get_batch_outputs()          — fetch completed batch results from GCS bucket
#   3. parse_llama_batch_output()   — bridge: convert raw Llama text → structured turn schema
#   4. validate_synthetic_turn()    — quality check each turn before saving
#   5. generate_synthetic_dataset() — full orchestration: BioASQ → dialogues → disk
#   6. save/load_synthetic_dataset()
#   7. get_synthetic_dataset_stats()
#   8. judge_conversation()         — Gemini judge for decomposition quality
#
# Turn schema (consumed by ConversationManager and evaluation_utils):
# {
#   "turn_id":          int,
#   "query":            str,
#   "answer":           str,
#   "requires_context": bool   ← drives compute_context_retention_accuracy()
# }

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

from data_utils import load_bioasq_dataset, parse_question

load_dotenv()

# ===========================================================================
# Project Setup
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SERVICE_KEY_PATH = Path(
    os.getenv("SERVICE_KEY_PATH", PROJECT_ROOT / "service_key.json")
).expanduser()

if not SERVICE_KEY_PATH.exists():
    raise FileNotFoundError(
        f"Service account key not found at: {SERVICE_KEY_PATH}. "
        "Set SERVICE_KEY_PATH in your environment or place service_key.json "
        "in the project root."
    )

credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_KEY_PATH)
)


# ===========================================================================
# 1. Batch Input Preparation (Llama 4 via Vertex AI MaaS)
# ===========================================================================

def prepare_llama_batch_file(input_records: list[dict],
                              output_path: str) -> None:
    """
    Build a JSONL batch input file for Llama 4 Maverick on Vertex AI MaaS.
    Each line is one BioASQ question formatted as a CoQA conversion prompt.

    The prompt instructs Llama to generate follow-up subquestions and
    subanswers in a structured, parseable format that _parse_coqa_text_to_turns()
    can reliably extract.

    Args:
        input_records: List of parsed BioASQ question dicts.
        output_path:   Path to write the output JSONL file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for record in input_records:
            question = record.get("body", "")
            answer   = record.get("ideal_answer") or record.get("exact_answer") or ""
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            context  = " ".join(s["text"] for s in record.get("snippets", []))

            prompt = f"""Convert this biomedical QA into a 3-turn CoQA-style dialogue.

Question: {question}
Answer: {answer}
Context: {context}

INSTRUCTIONS:
Generate follow-up sub-questions and sub-answers that break down the reasoning.
Use anaphoric references in follow-ups (e.g. "that gene", "this condition", "those mutations").
Ground all answers strictly in the context provided.

For example:
  Subquestion 1: What is the primary mechanism for receptor activation?
  Subanswer 1: EGFR activation occurs via ligand binding to the extracellular domain.
  Subquestion 2: Do these ligands exhibit differential binding affinities?
  Subanswer 2: Yes. Affinity typically follows the hierarchy: EGF > HB-EGF > TGF-α.

Do NOT ask trivial questions like "What is the topic of this question?"
Generate exactly 2 follow-up pairs (Subquestion 1/Subanswer 1, Subquestion 2/Subanswer 2).
"""
            line = {
                "custom_id": record["id"],
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model": "llama-4-maverick-17b-128e-instruct-maas",
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            f.write(json.dumps(line) + "\n")

    print(f"[INFO] Batch file written: {output_path} ({len(input_records)} records)")


# ===========================================================================
# 2. GCS Batch Output Fetching
# ===========================================================================

def get_latest_prediction_prefix(bucket, base_prefix: str = "output/") -> str:
    """
    Find the most recent Vertex AI prediction folder in the GCS bucket
    by sorting folder names alphabetically (timestamp-based names sort correctly).
    """
    blobs = bucket.list_blobs(prefix=base_prefix, delimiter="/")
    _     = list(blobs)  # must consume iterator to populate blobs.prefixes

    prediction_folders = [
        p for p in blobs.prefixes
        if "prediction-model-" in p
    ]

    if not prediction_folders:
        raise ValueError(f"No prediction folders found under '{base_prefix}' in bucket.")

    latest = sorted(prediction_folders)[-1]
    print(f"[INFO] Latest prediction folder: {latest}")
    return latest


def get_batch_outputs(bucket_name: str,
                      base_prefix: str = "output/") -> list[dict]:
    """
    Download all non-empty JSONL result files from the latest Vertex AI
    prediction folder in the given GCS bucket.

    Returns:
        List of parsed result dicts, one per line across all JSONL files.
    """
    storage_client = storage.Client(
        project="agentic-486120", credentials=credentials
    )
    bucket = storage_client.bucket(bucket_name)

    output_prefix = get_latest_prediction_prefix(bucket, base_prefix)
    blobs         = bucket.list_blobs(prefix=output_prefix)

    all_results = []
    for blob in blobs:
        if blob.name.endswith(".jsonl") and blob.size > 0:
            print(f"[INFO] Reading: {blob.name} ({blob.size} bytes)")
            content = blob.download_as_text()
            for line in content.splitlines():
                if line.strip():
                    all_results.append(json.loads(line))

    print(f"[INFO] Fetched {len(all_results)} results from GCS.")
    return all_results


# ===========================================================================
# 3. Bridge: Raw Llama Output → Structured Turn Schema
# ===========================================================================

def parse_llama_batch_output(llama_results: list[dict],
                              original_questions: list[dict]) -> list[dict]:
    """
    Convert raw Llama batch outputs into the standard dialogue schema
    expected by ConversationManager and evaluation_utils.

    Maps each result back to its source BioASQ question via custom_id,
    parses the free-text CoQA output into structured turns, and sets
    the requires_context flag on every follow-up turn.

    Args:
        llama_results:      Raw output from get_batch_outputs().
        original_questions: List of parsed BioASQ dicts (with "id" field).

    Returns:
        List of dialogue dicts matching the convert_to_multiturn() schema:
        {
            "source_id":      str,
            "question_type":  str,
            "turns":          list[dict],   ← structured, with requires_context
            "snippets":       list[dict],
        }
    """
    # Build lookup: question_id → original question
    q_lookup = {q["id"]: q for q in original_questions}
    dialogues = []

    for result in llama_results:
        qid = result.get("custom_id", "")

        # Extract content from Vertex AI MaaS response structure
        try:
            content = (
                result["response"]["body"]["choices"][0]["message"]["content"]
            )
        except (KeyError, IndexError, TypeError):
            print(f"[WARNING] Could not extract content for id '{qid}' — skipping.")
            continue

        original_q = q_lookup.get(qid, {})
        if not original_q:
            print(f"[WARNING] No matching BioASQ question found for id '{qid}' — skipping.")
            continue

        turns = _parse_coqa_text_to_turns(content, original_q)
        if not turns:
            print(f"[WARNING] No turns parsed for id '{qid}' — skipping.")
            continue

        dialogues.append({
            "source_id":     qid,
            "question_type": original_q.get("type", "summary"),
            "turns":         turns,
            "snippets":      original_q.get("snippets", []),
        })

    print(f"[INFO] Parsed {len(dialogues)} dialogues from {len(llama_results)} results.")
    return dialogues


def _parse_coqa_text_to_turns(text: str,
                               original_q: dict) -> list[dict]:
    """
    Parse Llama's free-text CoQA output into a list of structured turn dicts.

    Turn 1 is always the original BioASQ question (requires_context=False).
    All follow-up turns have requires_context=True since they build on prior turns.

    Handles Llama's output format:
        Subquestion 1: ...
        Subanswer 1: ...
        Subquestion 2: ...
        Subanswer 2: ...
    """
    turns = []

    # Turn 1: original BioASQ question
    original_answer = original_q.get("ideal_answer") or original_q.get("exact_answer") or ""
    if isinstance(original_answer, list):
        original_answer = original_answer[0] if original_answer else ""

    turns.append({
        "turn_id":          1,
        "query":            original_q.get("body", ""),
        "answer":           original_answer,
        "requires_context": False,
    })

    # Parse follow-up pairs from Llama output
    q_blocks = re.split(r"Sub\s*question\s*\d+\s*:", text, flags=re.IGNORECASE)
    a_blocks = re.split(r"Sub\s*answer\s*\d+\s*:",   text, flags=re.IGNORECASE)

    # Extract the first line of each block as the question/answer text
    followup_qs = [b.strip().split("\n")[0].strip() for b in q_blocks[1:]]
    followup_as = [b.strip().split("\n")[0].strip() for b in a_blocks[1:]]

    for i, (q, a) in enumerate(zip(followup_qs, followup_as), start=2):
        q, a = q.strip(), a.strip()
        if q and a:
            turns.append({
                "turn_id":          i,
                "query":            q,
                "answer":           a,
                "requires_context": True,
            })

    return turns


# ===========================================================================
# 4. Turn Validation
# ===========================================================================

def validate_synthetic_turn(turn: dict,
                             snippets: list[dict],
                             all_turns: list[dict]) -> bool:
    """
    Quality check for a single generated turn before it's added to the dataset.

    Checks:
      1. Query and answer are non-empty.
      2. Answer has at least minimal overlap with the snippet pool (grounding proxy).
      3. Query is not a near-duplicate of a previous turn's query.

    Args:
        turn:      The turn dict to validate.
        snippets:  Shared snippet pool for the dialogue.
        all_turns: All prior turns in this dialogue (for duplicate detection).

    Returns:
        True if the turn passes all checks.
    """
    query  = turn.get("query",  "").strip()
    answer = turn.get("answer", "").strip()

    # Check 1: non-empty
    if not query or not answer:
        return False

    # Check 2: grounding proxy — at least one content word from the answer
    # appears in the snippet pool
    snippet_text = " ".join(s.get("text", "") for s in snippets).lower()
    answer_words = set(w.lower() for w in answer.split() if len(w) > 4)
    overlap      = answer_words & set(snippet_text.split())
    if not overlap:
        return False

    # Check 3: not a duplicate of a prior query
    prior_queries = [t["query"].lower().strip() for t in all_turns]
    if query.lower() in prior_queries:
        return False

    return True


# ===========================================================================
# 5. Full Generation Pipeline
# ===========================================================================

def convert_to_multiturn(question: dict,
                          llm,
                          num_turns: int = 3) -> Optional[dict]:
    """
    Convert a single BioASQ question into a validated multi-turn dialogue
    using the loaded Gemini model directly (non-batch path).

    Useful for small runs and testing. For the full 100-dialogue dataset,
    use generate_synthetic_dataset() which uses the Llama batch pipeline.

    Args:
        question:  Parsed BioASQ question dict.
        llm:       Loaded google.generativeai.GenerativeModel instance.
        num_turns: Target number of turns including the original question.

    Returns:
        Validated dialogue dict, or None if generation/validation fails.
    """
    from prompt_templates import build_synthetic_multiturn_prompt

    prompt   = build_synthetic_multiturn_prompt(question, num_turns)
    response = llm.generate_content(
        prompt,
        generation_config={"temperature": 0.7, "max_output_tokens": 1024},
    )

    raw = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        dialogue = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[WARNING] JSON parse failed for question '{question.get('id')}': {e}")
        return None

    # Validate each turn and filter bad ones
    snippets      = question.get("snippets", [])
    valid_turns   = []
    for turn in dialogue.get("turns", []):
        if validate_synthetic_turn(turn, snippets, valid_turns):
            valid_turns.append(turn)

    if len(valid_turns) < 2:
        print(f"[WARNING] Too few valid turns for '{question.get('id')}' — discarding.")
        return None

    dialogue["turns"]    = valid_turns
    dialogue["snippets"] = snippets
    return dialogue


def generate_synthetic_dataset(questions: list[dict],
                                bucket_name: str,
                                batch_input_path: str = None,
                                output_path: str = None,
                                num_turns: int = 3,
                                max_dialogues: int = 100) -> list[dict]:
    """
    Full pipeline: BioASQ questions → Llama batch → parsed dialogues → disk.
    Targets the 100-scenario Biomedical Dialogue Set from the proposal.

    Workflow:
        1. Prepare Llama batch JSONL from the first max_dialogues questions.
        2. Fetch completed batch outputs from GCS.
        3. Parse outputs into structured turn schema.
        4. Validate each turn and filter bad dialogues.
        5. Save to output_path.

    Args:
        questions:        List of parsed BioASQ question dicts.
        bucket_name:      GCS bucket name where batch results are stored.
        batch_input_path: Where to write the batch input JSONL.
                          Defaults to data/synthetic/batch_input.jsonl
        output_path:      Where to save the final dialogues.
                          Defaults to data/synthetic/dialogues.json
        num_turns:        Target turns per dialogue.
        max_dialogues:    Cap on number of dialogues to generate.

    Returns:
        List of validated dialogue dicts.
    """
    batch_input_path = batch_input_path or str(
        PROJECT_ROOT / "data" / "synthetic" / "batch_input.jsonl"
    )
    output_path = output_path or str(
        PROJECT_ROOT / "data" / "synthetic" / "dialogues.json"
    )

    subset = questions[:max_dialogues]

    # Step 1: prepare batch input
    prepare_llama_batch_file(subset, batch_input_path)
    print(f"[INFO] Batch input written. Upload to GCS and run Vertex AI batch job, then re-run.")
    print(f"[INFO] Once the batch completes, call get_batch_outputs('{bucket_name}') manually.")

    # Step 2: fetch batch outputs
    llama_results = get_batch_outputs(bucket_name)

    # Step 3: parse into structured turn schema
    dialogues = parse_llama_batch_output(llama_results, subset)

    # Step 4: validate turns within each dialogue
    validated = []
    for dialogue in dialogues:
        snippets     = dialogue.get("snippets", [])
        valid_turns  = []
        for turn in dialogue.get("turns", []):
            if validate_synthetic_turn(turn, snippets, valid_turns):
                valid_turns.append(turn)

        if len(valid_turns) >= 2:
            dialogue["turns"] = valid_turns
            validated.append(dialogue)

    print(f"[INFO] Validated {len(validated)}/{len(dialogues)} dialogues.")

    # Step 5: save
    save_synthetic_dataset(validated, output_path)

    return validated


# ===========================================================================
# 6. Persistence
# ===========================================================================

def save_synthetic_dataset(dialogues: list[dict],
                            filepath: str) -> None:
    """Serialize synthetic dialogues to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(dialogues, f, indent=2)
    print(f"[INFO] Saved {len(dialogues)} dialogues to {filepath}")


def save_llama_results(llama_results: list[dict],
                       output_file: str = None) -> None:
    """Save raw Llama batch outputs to JSONL for inspection and reuse."""
    output_file = output_file or str(
        PROJECT_ROOT / "data" / "corpus" / "llama_results.jsonl"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for item in llama_results:
            f.write(json.dumps(item) + "\n")
    print(f"[INFO] Saved {len(llama_results)} raw results to {output_file}")


def load_synthetic_dataset(filepath: str) -> list[dict]:
    """Load previously generated synthetic dialogues from JSON."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Synthetic dataset not found at: {filepath}")
    with open(filepath) as f:
        dialogues = json.load(f)
    print(f"[INFO] Loaded {len(dialogues)} dialogues from {filepath}")
    return dialogues


# ===========================================================================
# 7. Dataset Statistics
# ===========================================================================

def get_synthetic_dataset_stats(dialogues: list[dict]) -> dict:
    """
    Sanity-check report on the generated dataset.
    Prints a summary and returns the stats dict.

    Returns:
        {
            "total_dialogues":            int,
            "avg_turns":                  float,
            "anaphoric_turn_ratio":       float,   ← % of turns with requires_context=True
            "question_type_distribution": dict,
        }
    """
    if not dialogues:
        return {"total_dialogues": 0, "avg_turns": 0.0,
                "anaphoric_turn_ratio": 0.0, "question_type_distribution": {}}

    total_turns      = 0
    anaphoric_turns  = 0
    type_dist: dict  = {}

    for d in dialogues:
        turns = d.get("turns", [])
        total_turns     += len(turns)
        anaphoric_turns += sum(1 for t in turns if t.get("requires_context", False))
        qtype            = d.get("question_type", "unknown")
        type_dist[qtype] = type_dist.get(qtype, 0) + 1

    stats = {
        "total_dialogues":            len(dialogues),
        "avg_turns":                  round(total_turns / len(dialogues), 2),
        "anaphoric_turn_ratio":       round(anaphoric_turns / total_turns, 2) if total_turns else 0.0,
        "question_type_distribution": type_dist,
    }

    print("\n[STATS] Synthetic Dataset Report")
    print(f"  Total dialogues:        {stats['total_dialogues']}")
    print(f"  Avg turns per dialogue: {stats['avg_turns']}")
    print(f"  Anaphoric turn ratio:   {stats['anaphoric_turn_ratio']:.0%}")
    print(f"  Question type dist:     {stats['question_type_distribution']}")

    return stats


# ===========================================================================
# 8. Gemini Judge — Decomposition Quality
# (evaluates synthetic DATA quality, not inference-time factuality)
# Note: for inference-time grounding checks, see generation_utils.check_answer_grounded()
# ===========================================================================

def build_decomposition_judge_prompt(original_q: list[dict],
                                      followups: list[dict]) -> str:
    """
    Build the prompt for evaluating the quality of generated follow-up
    question decomposition. This is a DATA QUALITY judge — used during
    synthetic dataset generation to filter low-quality dialogues.

    Distinct from build_judge_prompt() in prompt_templates.py, which
    evaluates medical factuality of answers at inference time.

    Args:
        original_q: List of original BioASQ question dicts.
        followups:  List of Llama result dicts for those questions.

    Returns:
        Prompt string for Gemini.
    """
    return f"""You are an expert evaluator of medical conversational question decomposition.

You are given:

Original Question — Answer Pairs:
{json.dumps(original_q, indent=2)}

Generated Follow-up Questions:
{json.dumps(followups, indent=2)}

For each question, evaluate the follow-up questions according to:

1. Relevance (1-5): Direct connection to original question.
2. Decomposition Quality (1-5): Breaks reasoning into meaningful substeps.
3. Non-Redundancy (1-5): Not restating the original question.
4. Guidance Utility (1-5): Would help a weaker model reach the answer.
5. Logical Ordering (1-5): Follows a clear reasoning progression.

Report the average scores across all questions and compute an overall score (1-5).

Return ONLY valid JSON — no preamble, no markdown:

{{
  "relevance":         <int>,
  "decomposition":     <int>,
  "non_redundancy":    <int>,
  "guidance":          <int>,
  "logical_ordering":  <int>,
  "overall":           <float>,
  "justification":     "<string>"
}}
"""


def judge_conversation(model,
                        original_q: list[dict],
                        followups: list[dict]) -> Optional[dict]:
    """
    Run the Gemini decomposition judge and return parsed scores.

    Args:
        model:      Loaded google.generativeai.GenerativeModel instance.
        original_q: List of original BioASQ question dicts.
        followups:  List of Llama result dicts.

    Returns:
        Parsed dict with decomposition quality scores, or None on failure.
    """
    prompt   = build_decomposition_judge_prompt(original_q, followups)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.0, "top_p": 1},
    )

    raw = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[WARNING] Judge returned invalid JSON:\n{raw}")
        return None


def create_judge_input(llama_results: list[dict],
                        output_file: str = None) -> None:
    """
    Convert raw Llama results into a Gemini batch judge input JSONL.
    Each line is a standalone decomposition judge request.

    Args:
        llama_results: Raw output from get_batch_outputs().
        output_file:   Path to write judge input JSONL.
    """
    output_file = output_file or str(
        PROJECT_ROOT / "data" / "corpus" / "judge_input.jsonl"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for item in llama_results:
            # Extract content from Vertex AI MaaS response structure
            try:
                llama_output = (
                    item["response"]["body"]["choices"][0]["message"]["content"]
                )
            except (KeyError, IndexError, TypeError):
                print(f"[WARNING] Skipping malformed result: {item.get('custom_id')}")
                continue

            judge_prompt = (
                f"Llama generated this CoQA output:\n{llama_output}\n\n"
                "Rate its decomposition quality from 1-5 across: relevance, "
                "decomposition, non_redundancy, guidance, logical_ordering. "
                "Return valid JSON only."
            )

            gemini_line = {
                "request": {
                    "contents": [
                        {"role": "user", "parts": [{"text": judge_prompt}]}
                    ],
                    "generationConfig": {"temperature": 0.0},
                }
            }
            f.write(json.dumps(gemini_line) + "\n")

    print(f"[INFO] Judge input written to {output_file}")


# ===========================================================================
# Entry Point
# ===========================================================================

def run() -> Optional[dict]:
    """
    Full pipeline run:
      1. Fetch Llama batch outputs from GCS.
      2. Save raw results.
      3. Create Gemini judge input.
      4. Parse outputs into structured dialogues.
      5. Save dialogues and print stats.
      6. Run decomposition judge on first 10.
    """
    bucket_name = "bioasq-bucket"

    # Step 1: fetch
    llama_results = get_batch_outputs(bucket_name)

    # Step 2: save raw
    save_llama_results(llama_results)

    # Step 3: create judge input
    create_judge_input(llama_results)

    # Step 4: load BioASQ and parse
    questions      = load_bioasq_dataset(
        str(PROJECT_ROOT / "data" / "BioASQ-training14b" / "training14b.json")
    )
    processed_qs   = [parse_question(q) for q in questions]

    # Step 5: parse outputs → structured dialogues
    dialogues = parse_llama_batch_output(llama_results, processed_qs)
    save_synthetic_dataset(
        dialogues,
        str(PROJECT_ROOT / "data" / "synthetic" / "dialogues.json"),
    )
    get_synthetic_dataset_stats(dialogues)

    # Step 6: decomposition judge on a small sample
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model   = genai.GenerativeModel("gemini-2.5-pro")
    metrics = judge_conversation(model, processed_qs[:10], llama_results[:10])
    print("Decomposition Quality Metrics:", metrics)

    return metrics


if __name__ == "__main__":
    run()