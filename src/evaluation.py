# --- evaluation.py -- Lowami ---

def run_phase_a_evaluation(predictions: list[dict], 
                            ground_truth: list[dict]) -> dict:
    """Full retrieval evaluation: MAP + Snippet F-measure."""

def run_phase_b_evaluation(predictions: list[dict], 
                            ground_truth: list[dict]) -> dict:
    """Full generation evaluation split by question type."""

def run_full_evaluation(predictions: list[dict], 
                        ground_truth: list[dict]) -> dict:
    """Combined Phase A + B report saved to output/evaluation/."""