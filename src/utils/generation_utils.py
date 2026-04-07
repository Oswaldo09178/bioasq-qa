from typing import Optional, Union, List, Dict, Tuple, Any
# Generation Utilities (Oswaldo)
# Handles LLM loading, answer generation, question-type routing,
# and strict grounding verification.
#
# Supports three backends:
#   - "huggingface" : local models (MedGemma-4B-it)
#   - "openai"      : GPT-4 via OpenAI API
#   - "google"      : Gemini via Google GenAI SDK (google-genai, NOT google-generativeai)
#
# The grounding check uses the judge prompt from prompt_templates.py
# and is the primary safeguard against hallucination (Risk R1).
#
# Fix history:
#   [FIX-1] _generate_hf: use tokenizer.apply_chat_template() instead of raw
#           tokenizer() call. MedGemma-4B-it is instruction-tuned and requires
#           its chat template — sending a raw string bypasses it entirely.
#   [FIX-2] _generate_hf: replaced greedy decoding (do_sample=False) with
#           low-temperature sampling (temperature=0.1, top_p=0.9).
#           Greedy decoding caused a strong "yes" bias on yesno questions and
#           produced repetitive outputs that parse_answer() could not parse.
#           The original nan/inf GPU error on L40S was caused by passing
#           temperature/top_p with do_sample=False — that combination is now
#           correct and safe.
#   [FIX-3] parse_answer / yesno: removed the hardcoded "yes" fallback default.
#           The old code silently returned "yes" for any unparseable yesno
#           output, making MedGemma appear to predict "yes" for every question
#           it could not parse, corrupting the entire yesno evaluation.
#           Now returns "" so evaluation treats unparseable outputs as missing.
#   [FIX-4] _generate_hf: apply_chat_template return type varies across
#           transformers versions. Calling it with truncation=True/max_length
#           causes it to return a BatchEncoding (dict-like) instead of a raw
#           tensor in newer versions, which breaks model.generate()'s internal
#           shape detection (inputs_tensor.shape[0] raises AttributeError).
#           Fix: call without those kwargs, handle both return types explicitly,
#           truncate manually, and pass input_ids as a keyword arg to generate().

import json
import os
import sys
import time

sys.path.append(os.path.dirname(__file__))
from prompt_templates import (
    build_summary_prompt,
    build_yesno_prompt,
    build_factoid_prompt,
    build_list_prompt,
    build_clarification_prompt,
    build_judge_prompt,
)


# ===========================================================================
# LLM Loading
# ===========================================================================

def load_llm(model_name: str, backend: str = "huggingface") -> dict:
    """
    Load an LLM and return a unified model handle used by generate_answer().

    Args:
        model_name: Model identifier.
                    HuggingFace : "google/medgemma-4b-it"
                    OpenAI      : "gpt-4o", "gpt-4-turbo"
                    Google      : "gemini-2.0-flash"
        backend:    "huggingface" | "openai" | "google"

    Returns:
        A dict {"backend", "model_name", "model", "tokenizer"} where
        model/tokenizer are backend-specific objects.
        generate_answer() unpacks this dict — callers treat it as opaque.
    """
    backend = backend.lower()

    if backend == "huggingface":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"[INFO] Loading HuggingFace model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # device_map="auto" on a CPU-only machine (e.g. Mac without GPU) causes
        # accelerate to offload layers to disk. Gemma3 has tied embeddings
        # (lm_head shares weights with embed_tokens) that break during disk-offload
        # reload, producing a ValueError shape mismatch at forward pass time.
        # On CUDA machines (Babel L40S) "auto" correctly places everything on GPU.
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            # CPU-only: load entirely in RAM, no disk offload, no shape mismatch.
            # Slow for inference but correct — fine for local --limit 10 validation.
            device_map = "cpu"
            torch_dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        model.eval()
        return {
            "backend": "huggingface",
            "model_name": model_name,
            "model": model,
            "tokenizer": tokenizer,
        }

    elif backend == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENAI_API_KEY environment variable.")

        client = OpenAI(api_key=api_key)
        print(f"[INFO] OpenAI client ready: {model_name}")
        return {
            "backend": "openai",
            "model_name": model_name,
            "model": client,
            "tokenizer": None,
        }

    elif backend == "google":
        try:
            from google import genai
        except ImportError:
            raise ImportError("Run: pip install google-genai")

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GOOGLE_API_KEY environment variable.")

        client = genai.Client(api_key=api_key)
        print(f"[INFO] Google GenAI client ready: {model_name}")
        return {
            "backend": "google",
            "model_name": model_name,
            "model": client,
            "tokenizer": None,
        }

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Use 'huggingface', 'openai', or 'google'."
        )


# ===========================================================================
# Core Generation
# ===========================================================================

def generate_answer(
    prompt: str,
    llm: dict,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    retries: int = 3,
) -> str:
    """
    Run inference on the given prompt using the loaded LLM.

    Args:
        prompt:      Full prompt string (from a prompt_templates builder).
        llm:         Model handle returned by load_llm().
        max_tokens:  Maximum tokens in the generated response.
        temperature: Low temperature (0.1) for factual clinical answers.
                     Use 0.7 for synthetic data generation (more variation).
        retries:     Number of retry attempts on transient API errors.

    Returns:
        Raw string response from the model. Parsing (JSON, list extraction,
        yes/no extraction) is handled downstream by parse_answer().
    """
    backend = llm["backend"]

    for attempt in range(1, retries + 1):
        try:
            if backend == "huggingface":
                return _generate_hf(prompt, llm, max_tokens, temperature)
            elif backend == "openai":
                return _generate_openai(prompt, llm, max_tokens, temperature)
            elif backend == "google":
                return _generate_google(prompt, llm, max_tokens, temperature)

        except Exception as e:
            print(f"[WARNING] Generation attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                raise

    return ""


def _generate_hf(prompt: str, llm: dict, max_tokens: int, temperature: float) -> str:
    """
    HuggingFace backend — targets instruction-tuned causal LMs (MedGemma-4B-it).

    FIX-1: apply_chat_template() is required for instruction-tuned models.
    The tokenizer wraps the prompt in the model's expected format
    (system/user/assistant roles, special tokens). Bypassing this with a raw
    tokenizer() call causes the model to receive malformed input and produce
    unstructured, unparseable output.

    FIX-2: Low-temperature sampling replaces greedy decoding.
    do_sample=True + temperature=0.1 + top_p=0.9 produces structured outputs
    that match our prompt templates' expected format. Greedy decoding caused:
      (a) strong "yes" bias on yesno questions (highest-probability first token)
      (b) repetitive, format-breaking outputs on factoid/list/summary questions
    The original nan/inf GPU error on L40S is NOT caused by sampling — it was
    caused by passing temperature/top_p alongside do_sample=False, which
    HuggingFace rejects. That combination is now removed.

    FIX-4: apply_chat_template return type varies across transformers versions.
    With truncation=True/max_length it returns a BatchEncoding (dict-like);
    without those kwargs it returns a raw tensor. We call it without truncation
    kwargs and handle both return types explicitly, then truncate manually.
    input_ids is passed as a keyword argument to model.generate() — passing it
    positionally failed when transformers expected a BatchEncoding at position 0.
    """
    import torch

    tokenizer = llm["tokenizer"]
    model = llm["model"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = [{"role": "user", "content": prompt}]

    # [FIX-4] Call without truncation/max_length to get a consistent tensor.
    # Some transformers versions return BatchEncoding when those kwargs are
    # present, which breaks model.generate()'s shape detection.
    chat_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Handle both return types defensively
    if hasattr(chat_inputs, "input_ids"):
        input_ids = chat_inputs["input_ids"].to(model.device)
    else:
        input_ids = chat_inputs.to(model.device)

    # Manual truncation — keep the last 2048 tokens (preserves the tail,
    # which contains the actual question, not the front padding)
    if input_ids.shape[-1] > 2048:
        input_ids = input_ids[:, -2048:]

    # [FIX-2] Low-temperature sampling — structured, non-degenerate outputs.
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,       # keyword arg — avoids positional ambiguity
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the input prompt)
    generated_ids = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _generate_openai(prompt: str, llm: dict, max_tokens: int, temperature: float) -> str:
    client = llm["model"]
    model_name = llm["model_name"]

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _generate_google(prompt: str, llm: dict, max_tokens: int, temperature: float) -> str:
    from google.genai import types

    client = llm["model"]
    model_name = llm["model_name"]

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )

    # Robustly extract text — response.text or candidates can be None
    # when safety filters block output or response structure differs by model.
    if response.text is not None:
        return response.text.strip()

    candidates = response.candidates or []
    for candidate in candidates:
        try:
            for part in candidate.content.parts or []:
                if hasattr(part, "text") and part.text:
                    return part.text.strip()
        except AttributeError:
            continue

    print(f"[DEBUG] Full response: {response}")
    raise ValueError(
        "Gemini returned no text content. "
        "Check response above for finish_reason or safety block details."
    )


# ===========================================================================
# Question Type Router
# ===========================================================================

def route_by_question_type(
    question: dict,
    snippets: list[dict],
    history: list[dict],
    llm: dict,
    max_tokens: int = 512,
    debug: bool = False,
) -> dict:
    """
    Select the correct prompt template based on question type,
    generate the answer, and return a structured result.

    Args:
        question:   BioASQ question dict with at least {"body", "type", "id"}.
        snippets:   Retrieved and reranked docs from retrieval_utils.
        history:    Context window from ConversationManager.get_context_window().
        llm:        Model handle from load_llm().
        max_tokens: Passed to generate_answer().
        debug:      If True, prints raw model output and parsed answer.
                    Use for validation runs (--limit 10) before full SLURM jobs.

    Returns:
        {
            "question_id":   str,
            "question_type": str,
            "raw_response":  str,         # full model output (for logging/debugging)
            "answer":        str | list,  # parsed answer
        }
    """
    qtype = question.get("type", "summary").lower()
    body = question.get("body", "")
    qid = question.get("id", "")

    prompt_builders = {
        "summary": build_summary_prompt,
        "yesno":   build_yesno_prompt,
        "factoid": build_factoid_prompt,
        "list":    build_list_prompt,
    }

    if qtype not in prompt_builders:
        print(f"[WARNING] Unknown question type '{qtype}', defaulting to summary.")
        qtype = "summary"

    prompt = prompt_builders[qtype](body, snippets, history)
    raw = generate_answer(prompt, llm, max_tokens=max_tokens)
    parsed = parse_answer(raw, qtype)

    if debug:
        print(f"[DEBUG] qid={qid} qtype={qtype}")
        print(f"[DEBUG] raw[:400]={raw[:400]!r}")
        print(f"[DEBUG] parsed={parsed!r}")
        print("---")

    return {
        "question_id":   qid,
        "question_type": qtype,
        "raw_response":  raw,
        "answer":        parsed,
    }


# ===========================================================================
# Answer Parsing
# ===========================================================================

def parse_answer(raw_response: str, qtype: str) -> Union[str, list]:
    """
    Parse the raw model output into the format expected by evaluation_utils.

    Args:
        raw_response: Full string output from generate_answer().
        qtype:        Question type — drives parsing strategy.

    Returns:
        - "summary"  → str   (text after "ANSWER:" marker, or full response)
        - "yesno"    → str   ("yes", "no", or "" if unparseable — see FIX-3)
        - "factoid"  → list[str]  (up to 3 numbered candidates)
        - "list"     → list[str]  (all bullet items)
    """
    if not raw_response:
        return "" if qtype in ("summary", "yesno") else []

    if qtype == "summary":
        if "ANSWER:" in raw_response:
            return raw_response.split("ANSWER:")[-1].strip()
        return raw_response.strip()

    elif qtype == "yesno":
        lower = raw_response.lower()

        # Priority 1: explicit "Answer: yes/no" marker (from prompt template)
        if "answer: yes" in lower:
            return "yes"
        if "answer: no" in lower:
            return "no"

        # Priority 2: standalone yes/no on its own line
        for line in lower.splitlines():
            stripped = line.strip()
            if stripped in ("yes", "no"):
                return stripped

        # Priority 3: first-word heuristic — catches "Yes, ..." / "No, ..."
        # only when it is the very first word of the response
        first_word = lower.split()[0].rstrip(".,;:") if lower.split() else ""
        if first_word in ("yes", "no"):
            return first_word

        # [FIX-3] No hardcoded default. Return "" so the evaluation layer
        # treats this question as unanswered rather than silently inflating
        # yes-counts. The judge will flag it as ungrounded separately.
        print(f"[WARNING] yesno parse failed — no yes/no found in output: {raw_response[:120]!r}")
        return ""

    elif qtype == "factoid":
        # Extract numbered list items: "1. answer text"
        candidates = []
        for line in raw_response.splitlines():
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                candidate = line.split(".", 1)[-1].strip()
                if candidate:
                    candidates.append(candidate)
        if candidates:
            return candidates[:3]
        # Fallback: return the full response as a single candidate rather than
        # an empty list — MRR can still score a correct full-text match.
        stripped = raw_response.strip()
        return [stripped] if stripped else []

    elif qtype == "list":
        # Extract bullet items: "- item text"
        items = []
        for line in raw_response.splitlines():
            line = line.strip()
            if line.startswith("-"):
                item = line.lstrip("-").strip()
                if item:
                    items.append(item)
        if items:
            return items
        # Fallback: return non-empty lines as items
        lines = [l.strip() for l in raw_response.splitlines() if l.strip()]
        return lines if lines else []

    return raw_response.strip()


# ===========================================================================
# Strict Grounding Check
# ===========================================================================

def check_answer_grounded(
    answer: str,
    snippets: list[dict],
    question: str,
    judge_llm: dict,
    score_threshold: int = 3,
) -> dict:
    """
    Verify that the generated answer is supported by retrieved snippets.
    Uses the LLM-as-judge approach (build_judge_prompt) to assess
    factuality and groundedness.

    This is the primary R1 (hallucination) mitigation from the proposal.
    Answers scoring below score_threshold are flagged for review.

    Args:
        answer:          Parsed answer string (from parse_answer()).
        snippets:        The same snippets used to generate the answer.
        question:        Original question body (for judge context).
        judge_llm:       A (possibly different) LLM used as evaluator.
                         Can be the same model or a stronger one (e.g. Gemini).
        score_threshold: Answers with factuality_score < this are flagged.

    Returns:
        {
            "grounded":         bool,
            "flagged":          bool,
            "factuality_score": int (1–5),
            "reasoning":        str,
        }
    """
    # Empty answers are trivially ungrounded — skip judge call
    if not answer or (isinstance(answer, list) and not any(answer)):
        return {
            "grounded":         False,
            "flagged":          True,
            "factuality_score": 0,
            "reasoning":        "Empty answer — nothing to ground.",
        }

    prompt = build_judge_prompt(question, answer, snippets)
    raw = generate_answer(prompt, judge_llm, max_tokens=256, temperature=0.0)

    # Strip markdown fences if model wraps the JSON in code blocks
    clean = (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    try:
        result = json.loads(clean)
        score = int(result.get("factuality_score", 0))
        return {
            "grounded":         bool(result.get("grounded", False)),
            "flagged":          bool(result.get("flagged", False)) or score < score_threshold,
            "factuality_score": score,
            "reasoning":        result.get("reasoning", ""),
        }

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[WARNING] Judge response could not be parsed: {e}\nRaw: {raw}")
        return {
            "grounded":         False,
            "flagged":          True,
            "factuality_score": 0,
            "reasoning":        f"Parse error: {e}",
        }


# ===========================================================================
# Clarification Generator
# ===========================================================================

def generate_clarification(question: str, history: list[dict], llm: dict) -> str:
    """
    Called by rag_system.py when ConversationManager.is_query_underspecified()
    returns True. Generates a clarifying question to ask the user.

    Args:
        question: The underspecified raw query.
        history:  Current context window.
        llm:      Loaded model handle.

    Returns:
        A clarifying question string to surface to the user.
    """
    prompt = build_clarification_prompt(question, history)
    return generate_answer(prompt, llm, max_tokens=80, temperature=0.3)