#!/bin/bash
#SBATCH --job-name=bioasq_experiments
#SBATCH --output=logs/bioasq_%j.out
#SBATCH --error=logs/bioasq_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# =============================================================================
# BioASQ Conversational RAG — Full Experiment Suite
# Covers H1 (retriever ablation) and H3 (generator ablation)
# =============================================================================

echo "=============================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=============================="

# --- Environment ---
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# Load .env (GOOGLE_API_KEY etc.)
set -a && source .env && set +a

mkdir -p logs output/prediction output/evaluation output/indices

DATA="data/BioASQ-training14b/training14b.json"

# =============================================================================
# H1 — Retriever Ablation (same generator, swap retriever)
# Run order: none → bm25 → dense → hybrid
# Dense builds the embedding index — subsequent hybrid run reuses it
# =============================================================================

echo ""
echo "[H1] Retriever ablation — generator=gemini"
echo "-------------------------------"

echo "[$(date)] Starting: none + gemini"
python src/rag_system.py \
    --retriever none \
    --generator gemini \
    --k 5 \
    --data $DATA \
    --output_dir output/ \
    --eval
echo "[$(date)] Done: none + gemini"

echo "[$(date)] Starting: bm25 + gemini"
python src/rag_system.py \
    --retriever bm25 \
    --generator gemini \
    --k 5 \
    --data $DATA \
    --output_dir output/ \
    --eval
echo "[$(date)] Done: bm25 + gemini"

echo "[$(date)] Starting: dense + gemini"
python src/rag_system.py \
    --retriever dense \
    --generator gemini \
    --k 5 \
    --data $DATA \
    --output_dir output/ \
    --eval
echo "[$(date)] Done: dense + gemini"

echo "[$(date)] Starting: hybrid + gemini"
python src/rag_system.py \
    --retriever hybrid \
    --generator gemini \
    --k 5 \
    --data $DATA \
    --output_dir output/ \
    --eval
echo "[$(date)] Done: hybrid + gemini"

# =============================================================================
# H3 — Generator Ablation (best retriever=hybrid, swap generator)
# MedGemma requires the GPU — keep on same node
# =============================================================================

echo ""
echo "[H3] Generator ablation — retriever=hybrid"
echo "-------------------------------"

echo "[$(date)] Starting: hybrid + medgemma"
python src/rag_system.py \
    --retriever hybrid \
    --generator medgemma \
    --k 5 \
    --data $DATA \
    --output_dir output/ \
    --eval
echo "[$(date)] Done: hybrid + medgemma"

echo ""
echo "=============================="
echo "All experiments complete: $(date)"
echo "Results in output/evaluation/"
echo "=============================="