# Conversational Biomedical QA System

A multi-turn Retrieval-Augmented Generation (RAG) system for medical practitioners, built on the BioASQ challenge dataset.

**Team:**
- Oswaldo (Intelligence Lead) — Conversation memory, LLM prompting, multi-turn logic
- Joel (Retrieval Lead) — PubMed indexing, hybrid retrieval pipeline, snippet extraction
- Lowami (Data & Evaluation Lead) — Synthetic data generation, BioASQ harness, user study

---

## Project Structure

```
BioASQ/
├── README.md
├── requirements.txt
├── configs/
│   ├── retrieval_config.yaml      # BM25 / BGE-M3 / MedCPT settings
│   ├── generation_config.yaml     # LLM backend, prompting strategy
│   └── evaluation_config.yaml     # Metrics, thresholds, output paths
├── data/
│   ├── BioASQ-training14b/        # Raw BioASQ 14b training JSON
│   └── corpus/
│       ├── pubmed/                # PubMed abstracts fetched via E-utilities
│       └── bioasq_snippets/       # Preprocessed BioASQ gold snippets
├── notebook/
│   └── example_notebook.ipynb
├── output/
│   ├── evaluation/                # Scored results (MAP, F1, ROUGE, etc.)
│   └── prediction/                # Raw model outputs
├── scripts/
│   ├── index_corpus.py            # One-off: index PubMed corpus
│   └── fetch_pubmed.py            # One-off: fetch abstracts via E-utilities
└── src/
    ├── rag_system.py              # Central orchestrator
    ├── evaluation.py              # Evaluation entry point
    └── utils/
        ├── conversation_manager.py
        ├── data_utils.py
        ├── evaluation_utils.py
        ├── generation_utils.py
        └── retrieval_utils.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.rag_system import BioASQRAGSystem

system = BioASQRAGSystem(config_path="configs/")
response = system.answer(query="What are the side effects of metformin?")
```

## Research Questions

- **RQ1**: Does conversational context-awareness improve answer accuracy?
- **RQ2**: Which hybrid retrieval architecture is optimal for multi-turn medical queries?
- **RQ3**: Does the system reduce Time-to-Answer vs. traditional literature search?