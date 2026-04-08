cat > scripts/run_h2_multiturn.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=h2_multiturn
#SBATCH --output=logs/h2_multiturn_%j.out
#SBATCH --error=logs/h2_multiturn_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
# No GPU needed — Gemini is API-based

source .venv/bin/activate
python scripts/test_multiturn.py
EOF