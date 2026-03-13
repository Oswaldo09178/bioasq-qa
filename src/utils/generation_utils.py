# Generation Utilities (Oswaldo)
# Handles LLM loading, answer generation, question-type routing,
# and strict grounding verification.
#
# Supports three backends:
#   - "huggingface" : local models (PubMedBERT, MedGemma)
#   - "openai"      : GPT-4 via OpenAI API
#   - "google"      : Gemini/MedPaLM via Google API
#
# The grounding check uses the judge prompt from prompt_templates.py
# and is the primary safeguard against hallucination (Risk R1).

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
                    HuggingFace : "ncbi/MedCPT-Query-Encoder",
                                  "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                                  "google/medgemma-4b-it"
                    OpenAI      : "gpt-4o", "gpt-4-turbo"
                    Google      : "gemini-1.5-pro"
        backend:    "huggingface" | "openai" | "google"

    Returns:
        A dict {"backend", "model_name", "model", "tokenizer"} where
        model/tokenizer are backend-specific objects.
        generate_answer() unpacks this dict — callers treat it as opaque.
    """
    backend = backend.lower()

    if backend == "huggingface":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        print(f"[INFO] Loading HuggingFace model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
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
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        print(f"[INFO] Google GenAI client ready: {model_name}")
        return {
            "backend": "google",
            "model_name": model_name,
            "model": model,
            "tokenizer": None,
        }

    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'huggingface', 'openai', or 'google'.")


# ===========================================================================
# Core Generation
# ===========================================================================

def generate_answer(prompt: str,
                    llm: dict,
                    max_tokens: int = 512,
                    temperature: float = 0.1,
                    retries: int = 3) -> str:
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
        yes/no extraction) is handled downstream by the caller.
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


def _generate_hf(prompt: str, llm: dict,
                 max_tokens: int, temperature: float) -> str:
    import torch
    tokenizer = llm["tokenizer"]
    model     = llm["model"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _generate_openai(prompt: str, llm: dict,
                     max_tokens: int, temperature: float) -> str:
    client     = llm["model"]
    model_name = llm["model_name"]

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _generate_google(prompt: str, llm: dict,
                     max_tokens: int, temperature: float) -> str:
    import google.generativeai as genai
    model = llm["model"]

    config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    response = model.generate_content(prompt, generation_config=config)
    return response.text.strip()


# ===========================================================================
# Question Type Router
# ===========================================================================

def route_by_question_type(question: dict,
                            snippets: list[dict],
                            history: list[dict],
                            llm: dict,
                            max_tokens: int = 512) -> dict:
    """
    Select the correct prompt template based on question type,
    generate the answer, and return a structured result.

    Args:
        question:   BioASQ question dict with at least {"body", "type", "id"}.
        snippets:   Retrieved and reranked docs from retrieval_utils.
        history:    Context window from ConversationManager.get_context_window().
        llm:        Model handle from load_llm().
        max_tokens: Passed to generate_answer().

    Returns:
        {
            "question_id":  str,
            "question_type": str,
            "raw_response": str,   # full model output
            "answer":       str | list,  # parsed answer
        }
    """
    qtype  = question.get("type", "summary").lower()
    body   = question.get("body", "")
    qid    = question.get("id", "")

    # Select prompt builder by question type
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
    raw    = generate_answer(prompt, llm, max_tokens=max_tokens)

    return {
        "question_id":   qid,
        "question_type": qtype,
        "raw_response":  raw,
        "answer":        parse_answer(raw, qtype),
    }


# ===========================================================================
# Answer Parsing
# ===========================================================================

def parse_answer(raw_response: str, qtype: str) -> str | list:
    """
    Parse the raw model output into the format expected by evaluation_utils.

    Args:
        raw_response: Full string output from generate_answer().
        qtype:        Question type — drives parsing strategy.

    Returns:
        - "summary"  → str  (text after "ANSWER:" marker)
        - "yesno"    → str  ("yes" or "no")
        - "factoid"  → list[str]  (up to 3 candidates, ordered)
        - "list"     → list[str]  (all bullet items)
    """
    if qtype == "summary":
        if "ANSWER:" in raw_response:
            return raw_response.split("ANSWER:")[-1].strip()
        return raw_response.strip()

    elif qtype == "yesno":
        lower = raw_response.lower()
        if "answer: yes" in lower:
            return "yes"
        elif "answer: no" in lower:
            return "no"
        # Fallback: scan for standalone yes/no
        for line in lower.split("\n"):
            if line.strip() in ("yes", "no"):
                return line.strip()
        return "yes"  # safe default — flagged for review by judge

    elif qtype == "factoid":
        # Extract numbered list items
        candidates = []
        for line in raw_response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                candidate = line.split(".", 1)[-1].strip()
                if candidate:
                    candidates.append(candidate)
        return candidates[:3] if candidates else [raw_response.strip()]

    elif qtype == "list":
        # Extract bullet items
        items = []
        for line in raw_response.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                item = line.lstrip("-").strip()
                if item:
                    items.append(item)
        return items if items else [raw_response.strip()]

    return raw_response.strip()


# ===========================================================================
# Strict Grounding Check
# ===========================================================================

def check_answer_grounded(answer: str,
                           snippets: list[dict],
                           question: str,
                           judge_llm: dict,
                           score_threshold: int = 3) -> dict:
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
                         Can be the same model or a stronger one (e.g. GPT-4).
        score_threshold: Answers with factuality_score < this are flagged.

    Returns:
        {
            "grounded":         bool,
            "flagged":          bool,
            "factuality_score": int (1-5),
            "reasoning":        str,
        }
    """
    prompt   = build_judge_prompt(question, answer, snippets)
    raw      = generate_answer(prompt, judge_llm, max_tokens=256, temperature=0.0)

    # Strip markdown fences if model adds them
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(clean)
        return {
            "grounded":         bool(result.get("grounded", False)),
            "flagged":          bool(result.get("flagged", False))
                                or int(result.get("factuality_score", 0)) < score_threshold,
            "factuality_score": int(result.get("factuality_score", 0)),
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

def generate_clarification(question: str,
                            history: list[dict],
                            llm: dict) -> str:
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