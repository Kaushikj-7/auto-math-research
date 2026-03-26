from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from typing import Any

import torch

from training.evaluation import compute_fitness
from training.kernels import update_weights, fused_step
from training.execute_ast import execute_ast, verify_cuda_stack
from data.queue import claim_next_pending, mark_done, mark_failed
from configs.research_config import ResearchConfig
from utils.cuda_monitor import CudaEnforcer, ExperimentMonitor

DB_PATH = ResearchConfig.DB_PATH
SLEEP_SECONDS = ResearchConfig.TPU_WORKER_SLEEP
DEFAULT_STEPS = ResearchConfig.DEFAULT_STEPS


def _worker_id(device: str) -> str:
    return f"{device}-worker-{os.getpid()}"


def _ensure_tensor(value: Any, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, dtype=torch.float32, device=device)


def run_worker(
    device: str = "cuda" if torch.cuda.is_available() else "cpu", db_path: str = DB_PATH
):
    print(f"Starting Worker on {device} using {db_path}")

    # Enforce strict CUDA requirements
    CudaEnforcer.enforce_env()
    verify_cuda_stack()

    monitor = ExperimentMonitor()
    worker_id = _worker_id(device)

    while True:
        try:
            conn = sqlite3.connect(db_path)
            row = claim_next_pending(conn, worker_id)
            conn.close()

            if not row:
                time.sleep(SLEEP_SECONDS)
                continue

            print(f"Claimed task {row['id']}")
            start_time = time.time()

            # Retrieve AST
            candidate_ast = json.loads(row["raw_ast_json"])

            # Setup context for execution (Mocking a training step)
            # In a real loop, this would run for DEFAULT_STEPS
            # using the candidate_ast to update weights.

            # 1. Init Data (using cuRAND, cuBLAS)
            W = torch.randn(ResearchConfig.MATRIX_SHAPE, device=device)
            grad = torch.randn(ResearchConfig.MATRIX_SHAPE, device=device)
            M_t = torch.zeros(ResearchConfig.MATRIX_SHAPE, device=device)
            V_t = torch.zeros(ResearchConfig.MATRIX_SHAPE, device=device)
            context = {
                "W_t": W,
                "grad_L": grad,
                "M_t": M_t,
                "V_t": V_t,
                "alpha": 0.01,
                "epsilon": 1e-8,
            }

            # 2. Execute Graph (using cuBLAS, cuSOLVER, cuFFT etc via AST)
            # This ensures the candidate code actually runs on GPU
            try:
                # Dry run to verify executable and measure overhead
                _ = execute_ast(candidate_ast, context)
            except Exception as ast_err:
                raise RuntimeError(f"AST Execution failed: {ast_err}")

            # 3. Simulate Training Loop
            # (Replace this with actual model training in future)
            time.sleep(0.1)  # Simulating work

            duration_ms = (time.time() - start_time) * 1000

            # Mock fitness
            loss_sum = 10.0  # Placeholder
            objective_value = compute_fitness(loss_sum, flops=0.0)  # Placeholder
            flops_estimate = 1e9  # Placeholder

            # Collect Metrics
            gpu_metrics = CudaEnforcer.get_gpu_metrics()
            metrics = {
                "loss": loss_sum,
                "flops": flops_estimate,
                "duration_ms": duration_ms,
                "cuda_peak_memory": gpu_metrics.get("cuda_peak_memory", 0),
            }

            # Log to CSV for day-to-day progress
            monitor.log_experiment(row["id"], row["generation"], metrics, gpu_metrics)

            conn = sqlite3.connect(db_path)
            mark_done(
                conn,
                row["id"],
                objective_value,
                loss_sum,
                flops_estimate,
                DEFAULT_STEPS,
            )
            conn.close()
            print(f"Completed task {row['id']} - Logged metrics.")

        except Exception as e:
            print(f"Worker error: {e}")
            if "row" in locals() and row:
                conn = sqlite3.connect(db_path)
                mark_failed(conn, row["id"], "WORKER_ERROR", str(e))
                conn.close()
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    run_worker()
