import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.scripts.init_db import setup_db
from core.configs.research_config import ResearchConfig
from core.optimizers.mutator import insert_candidate

import sqlite3

def run_researcher(prompt: str, threshold: float = 1e11):
    """
    Parses a research prompt, establishes compute limits, and seeds a baseline.
    In a fully AI-driven environment, this would call an LLM to generate the AST baseline.
    Here we simulate "fetching parameters from papers" by creating a strong baseline.
    """
    print(f"🔬 Starting Research Agent\nPrompt: '{prompt}'")
    print(f"📊 Compute Threshold: {threshold:e} FLOPs/Step")
    
    # Simulate API interaction / literature review for math ops
    print("📚 Simulating literature review for optimization operators...")
    
    # Baseline: A simple AdamW-like or SGD-like AST representation
    # W_t - alpha * grad_L
    baseline_ast = ["sub", "W_t", ["mul", "alpha", "grad_L"]]
    
    # Save to database
    db_path = ResearchConfig.DB_PATH
    conn = sqlite3.connect(db_path)
    
    # Check if there's already a baseline
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM optimizers WHERE generation = 0 AND parent_id IS NULL")
    existing = cursor.fetchone()
    
    if existing:
        print(f"ℹ️ Found existing baseline (ID: {existing[0]}). Proceeding with existing campaign.")
        baseline_id = existing[0]
    else:
        print("🌱 Seeding Baseline AST into blackboard...")
        # Since insert_candidate does not take flops_threshold natively, we update it right after
        inserted = insert_candidate(
            conn,
            parent_id=None,
            generation=0,
            raw_ast=baseline_ast,
            latex_formula="W_t - \\alpha \\nabla L"
        )
        
        # Get the ID of the inserted baseline
        cursor.execute("SELECT id FROM optimizers WHERE ast_hash = (SELECT ast_hash FROM optimizers ORDER BY id DESC LIMIT 1)")
        baseline_id = cursor.fetchone()[0]
        
        # Mark it as done so mutator can branch from it (or leave it pending to be run as a baseline)
        # We need to run it to get baseline metrics! So we keep it pending.
        cursor.execute("UPDATE optimizers SET flops_threshold = ?, status = 'pending' WHERE id = ?", (threshold, baseline_id))
        conn.commit()
        print(f"✅ Baseline seeded successfully (ID: {baseline_id}). Waiting for workers to execute it.")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Agent Entry Point")
    parser.add_argument("prompt", type=str, help="Research prompt (e.g. 'research matrix operations in AI')")
    parser.add_argument("--threshold", type=float, default=1e11, help="Max FLOPs per step allowed")
    
    args = parser.parse_args()
    
    setup_db() # Ensure DB is initialized
    run_researcher(args.prompt, args.threshold)
