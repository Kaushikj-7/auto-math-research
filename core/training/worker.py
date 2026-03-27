from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any
from pathlib import Path

import torch
import sys

# Root and Guide Paths
project_root = Path(__file__).parent.parent.parent
guide_path = project_root / "muon-optimizer-guide"

# Add project root and guide path to sys.path
# We insert them at the front to ensure they are searchable
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(guide_path))

try:
    # Explicitly import from the guide files by filename if needed, 
    # but we'll try the sys.path first.
    # To avoid collision with 'core.training', we can try to use the guide_path directly
    from training import trainer, evaluation
    from configs.llm_config import LLMConfig
    from data.loader import get_dataloaders
    from utils.gpu_monitor import GPUMonitor
    
    train_minimal_llm = trainer.train_minimal_llm
except ImportError as e:
    print(f"Warning: Muon guide components not found: {e}")
    # Fallback to local mocks if needed for demonstration
    class LLMConfig: 
        def __init__(self): 
            self.device = "cpu"
            self.train_tokens = 100
            self.batch_size = 1
            self.max_seq_len = 8
            self.vocab_size = 100
            self.gradient_accumulation_steps = 1
            self.use_amp = False
            self.compile_model = False
            self.eval_milestones = []
            self.eval_every = 10
            self.grad_clip = 1.0
    def get_dataloaders(cfg): return [None], [None]
    def train_minimal_llm(**kwargs): 
        return {
            'metrics': {'val_loss': 5.0, 'val_accuracy': 0.1, 'val_perplexity': 100.0},
            'history': {'steps': [0], 'val_losses': [5.0]},
            'steps': 1
        }
    def GPUMonitor(*args, **kwargs): pass

from core.data.queue import claim_next_pending, mark_done, mark_failed
from core.configs.research_config import ResearchConfig
from core.utils.compute_estimator import estimate_ast_flops, is_feasible
from core.utils.logging import save_experiment_results
from core.fitness import compute_fitness

DB_PATH = ResearchConfig.DB_PATH
SLEEP_SECONDS = ResearchConfig.TPU_WORKER_SLEEP

def _worker_id(device: str) -> str:
    return f"{device}-worker-{os.getpid()}"

def run_worker(
    device: str = "cuda" if torch.cuda.is_available() else "cpu", db_path: str = DB_PATH
):
    print(f"🚀 Starting Autonomous Worker on {device}")
    worker_id = _worker_id(device)
    
    # Initialize GPU Monitor if on CUDA
    gpu_mon = None
    if device == "cuda":
        try:
            gpu_mon = GPUMonitor(interval=5)
            gpu_mon.start()
        except:
            print("GPU Monitoring unavailable.")

    while True:
        try:
            conn = sqlite3.connect(db_path)
            row = claim_next_pending(conn, worker_id)
            conn.close()

            if not row:
                time.sleep(SLEEP_SECONDS)
                continue

            print(f"🛠️ Claimed experiment {row['id']} (Gen {row['generation']})")
            
            candidate_ast = json.loads(row["raw_ast_json"])
            threshold = row["flops_threshold"] or 1e11
            baseline_id = row["baseline_id"]
            
            # 1. Feasibility Check
            flops_est = estimate_ast_flops(candidate_ast)
            if not is_feasible(candidate_ast, threshold):
                print(f"⚠️ AST exceeding compute threshold ({flops_est:e} > {threshold:e}). Quarantining.")
                conn = sqlite3.connect(db_path)
                mark_failed(conn, row["id"], "COMPUTE_EXCEEDED", f"Estimated FLOPs {flops_est:e} exceeds threshold {threshold:e}")
                conn.close()
                continue

            # 2. Execution Setup
            config = LLMConfig()
            config.device = device
            # Adjust config for research (extremely small for demonstration)
            config.train_tokens = 10_000 
            config.use_amp = (device == "cuda") 
            config.compile_model = False 
            
            # 3. Train & Evaluate
            print(f"📉 Running training for exp {row['id']}...")
            try:
                # Load Data
                train_loader, val_loader = get_dataloaders(config)
                
                results = train_minimal_llm(
                    config=config,
                    train_loader=train_loader,
                    val_loader=val_loader
                )
                
                loss_sum = results['metrics']['val_loss']
                steps_run = results['steps']
                
                # 4. Fitness & Artifacts
                obj_val = compute_fitness(loss_sum, flops=flops_est)
                
                # Save artifacts (metrics + graphs)
                artifact_path = save_experiment_results(
                    experiment_id=row['id'],
                    metrics=results['metrics'],
                    history=results['history'],
                    baseline_id=baseline_id
                )
                
                conn = sqlite3.connect(db_path)
                mark_done(
                    conn,
                    row["id"],
                    obj_val,
                    loss_sum,
                    flops_est,
                    steps_run,
                    artifact_path=artifact_path
                )
                conn.close()
                print(f"✅ Experiment {row['id']} complete. Objective: {obj_val:.4f}")

            except Exception as train_err:
                print(f"❌ Training failed: {train_err}")
                conn = sqlite3.connect(db_path)
                mark_failed(conn, row["id"], "TRAINING_FAILED", str(train_err))
                conn.close()

        except Exception as e:
            print(f"💥 Worker critical error: {e}")
            time.sleep(SLEEP_SECONDS)

    if gpu_mon:
        gpu_mon.stop()

if __name__ == "__main__":
    run_worker()
