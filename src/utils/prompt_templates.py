# Prompt Templates (Oswaldo)
# One template per BioASQ question type + clarification + synthetic generation.
# All prompts enforce strict grounding — the LLM must only use retrieved snippets.
#
# Template inputs follow a consistent signature:
#   question:  str               — the user's query (anaphora already resolved)
#   snippets:  list[dict]        — retrieved docs: [{"text", "pmid", ...}]
#   history:   list[dict]        — context window: [{"role", "content"}]
#
# The STRICT GROUNDING instruction appears in every clinical prompt.
# This is the primary mitigation for hallucination risk (R1 in the proposal).


# ===========================================================================
# Helpers
# ===========================================================================

def _format_snippets(snippets: list[dict]) -> str:
    """
    Render retrieved snippets as a numbered evidence block.
    Each snippet is prefixed with its PMID for traceability.
    """
    if not snippets:
        return "  [No evidence retrieved]"
    lines = []
    for i, s in enumerate(snippets, 1):
        pmid = s.get("pmid", "unknown")
        text = s.get("text", "").strip()
        lines.append(f"  [{i}] (PMID {pmid}) {text}")
    return "\n".join(lines)


def _format_history(history: list[dict]) -> str:
    """
    Render conversation history as a readable dialogue block.
    Passed from ConversationManager.get_context_window().
    """
    if not history:
        return "  [No prior conversation]"
    lines = []
    for turn in history:
        role = turn["role"].upper()
        lines.append(f"  {role}: {turn['content']}")
    return "\n".join(lines)


GROUNDING_INSTRUCTION = (
    "IMPORTANT: Base your answer primarily on the evidence snippets provided above. "
    "If the snippets are partially relevant, use them to construct the best possible answer — "
    "do not refuse to answer just because the evidence is incomplete. "
    "Only state 'Insufficient evidence in retrieved documents.' if the snippets are "
    "completely unrelated to the question and you cannot derive any useful answer from them."
)


# ===========================================================================
# Question Type Prompts
# ===========================================================================

def build_summary_prompt(question: str,
                          snippets: list[dict],
                          history: list[dict]) -> str:
    """
    Prompt for summary-type questions.
    These require synthesizing information across multiple snippets
    into a coherent, concise clinical paragraph.

    Chain-of-Thought (CoT) is used: the model is asked to reason
    step by step before producing the final answer.

    Example question: "What is the role of RET in Hirschsprung disease?"
    """
    return f"""You are a clinical research assistant helping a medical professional.
Your task is to answer a biomedical question by synthesizing the provided evidence.

--- CONVERSATION HISTORY ---
{_format_history(history)}

--- RETRIEVED EVIDENCE ---
{_format_snippets(snippets)}

--- QUESTION ---
{question}

--- INSTRUCTIONS ---
{GROUNDING_INSTRUCTION}

Think step by step:
1. Identify which snippets are most relevant to the question.
2. Note any agreements or contradictions across snippets.
3. Synthesize a coherent, concise summary answer (2-4 sentences).

Begin your reasoning, then provide your final answer after the line "ANSWER:".
"""


def build_yesno_prompt(question: str,
                        snippets: list[dict],
                        history: list[dict]) -> str:
    """
    Prompt for yes/no questions.
    The model must commit to 'yes' or 'no' and justify with evidence.
    Evaluated by macro F1 and accuracy — so the answer word matters.

    Example question: "Is BRCA1 associated with hereditary breast cancer?"
    """
    return f"""You are a clinical research assistant helping a medical professional.
Your task is to answer a yes/no biomedical question based solely on the evidence provided.

--- CONVERSATION HISTORY ---
{_format_history(history)}

--- RETRIEVED EVIDENCE ---
{_format_snippets(snippets)}

--- QUESTION ---
{question}

--- INSTRUCTIONS ---
{GROUNDING_INSTRUCTION}

Your response must follow this exact format:
ANSWER: yes  (or)  ANSWER: no
JUSTIFICATION: [1-2 sentences citing the specific snippet(s) that support your answer]
"""


def build_factoid_prompt(question: str,
                          snippets: list[dict],
                          history: list[dict]) -> str:
    """
    Prompt for factoid questions.
    Expects a short, exact answer (a named entity, number, or brief phrase).
    Evaluated by Mean Reciprocal Rank (MRR) — the first answer is most important.

    The model is asked to provide up to 3 candidates ranked by confidence,
    which is the standard BioASQ factoid submission format.

    Example question: "Which gene is most commonly mutated in Hirschsprung disease?"
    """
    return f"""You are a clinical research assistant helping a medical professional.
Your task is to extract a precise factoid answer from the provided evidence.

--- CONVERSATION HISTORY ---
{_format_history(history)}

--- RETRIEVED EVIDENCE ---
{_format_snippets(snippets)}

--- QUESTION ---
{question}

--- INSTRUCTIONS ---
{GROUNDING_INSTRUCTION}

Provide up to 3 answer candidates ranked by confidence (most likely first).
Each answer should be a short entity, name, number, or brief phrase — NOT a full sentence.

Format:
1. [most likely answer]
2. [second candidate, if applicable]
3. [third candidate, if applicable]
"""


def build_list_prompt(question: str,
                       snippets: list[dict],
                       history: list[dict]) -> str:
    """
    Prompt for list questions.
    Expects an exhaustive enumeration of items supported by the evidence.
    Evaluated by mean F1 — both precision and recall matter.

    Example question: "What genes are involved in Hirschsprung disease?"
    """
    return f"""You are a clinical research assistant helping a medical professional.
Your task is to extract a complete list of items that answer the biomedical question.

--- CONVERSATION HISTORY ---
{_format_history(history)}

--- RETRIEVED EVIDENCE ---
{_format_snippets(snippets)}

--- QUESTION ---
{question}

--- INSTRUCTIONS ---
{GROUNDING_INSTRUCTION}

Return a bullet-point list. Include every item supported by the evidence.
Do not include items not mentioned in the snippets.
Do not add explanations — list items only.

ANSWER:
- [item 1]
- [item 2]
- ...
"""


# ===========================================================================
# Clarification Prompt
# ===========================================================================

def build_clarification_prompt(question: str,
                                history: list[dict]) -> str:
    """
    Prompt used when ConversationManager.is_query_underspecified() returns True.
    Instead of attempting retrieval on a vague query, the system asks
    the user to clarify. The clarifying question should be specific and
    clinically relevant.

    Example trigger: user says "What about it?" after a long exchange.
    """
    return f"""You are a clinical research assistant helping a medical professional.
The user's latest question is ambiguous and cannot be answered without clarification.

--- CONVERSATION HISTORY ---
{_format_history(history)}

--- AMBIGUOUS QUESTION ---
{question}

--- TASK ---
Generate one short, specific clarifying question to ask the user.
The question should help you understand what clinical concept, drug, gene,
or condition they are referring to.

Respond with only the clarifying question — no preamble.
"""


# ===========================================================================
# LLM-as-Judge Prompt
# ===========================================================================

def build_judge_prompt(question: str,
                        answer: str,
                        snippets: list[dict]) -> str:
    """
    Prompt for the LLM-as-judge evaluation step.
    Used in evaluation_utils.llm_as_judge() and in the strict grounding
    check in generation_utils.check_answer_grounded().

    The judge evaluates:
      - factuality_score: 1-5 (is the answer medically accurate per snippets?)
      - grounded:         True/False (is every claim traceable to a snippet?)
      - flagged:          True if the answer contains a claim contradicted by evidence

    Response must be valid JSON for reliable parsing.
    """
    return f"""You are an expert biomedical reviewer evaluating the quality of an AI-generated answer.

--- QUESTION ---
{question}

--- RETRIEVED EVIDENCE (ground truth source) ---
{_format_snippets(snippets)}

--- ANSWER TO EVALUATE ---
{answer}

--- TASK ---
Evaluate the answer strictly against the retrieved evidence above.
Respond ONLY with a valid JSON object — no preamble, no markdown, no explanation outside the JSON.

{{
  "factuality_score": <integer 1-5>,
  "grounded": <true|false>,
  "flagged": <true|false>,
  "reasoning": "<1-2 sentences explaining your scores>"
}}

Scoring guide for factuality_score:
  5 = Fully accurate, all claims supported by evidence
  4 = Mostly accurate, minor omissions
  3 = Partially accurate, some unsupported claims
  2 = Mostly inaccurate or misleading
  1 = Contradicts the evidence or fabricated
"""