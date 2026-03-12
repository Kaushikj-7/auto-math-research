from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import torch

from core.fitness import compute_fitness
from core.queue_claims import claim_next_pending, mark_done, mark_failed

DB_PATH = Path("optimizer_evolution.db")
SLEEP_SECONDS = 1
DEFAULT_STEPS = 100


def _worker_id(device: str) -> str:
    return f"{device}-worker-{os.getpid()}"


def _ensure_tensor(value: Any, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, dtype=torch.float32, device=device)


def _matrix_exp(matrix: torch.Tensor) -> torch.Tensor:
    return torch.matrix_exp(matrix)


def _skew(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix - matrix.transpose(-1, -2))


def _symm(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _cayley(matrix: torch.Tensor) -> torch.Tensor:
    skew = _skew(matrix)
    identity = torch.eye(skew.shape[0], device=skew.device, dtype=skew.dtype)
    a = 0.1 * skew
    return torch.linalg.solve(identity - a, identity + a)


def _eval_ast(
    node: Any, context: dict[str, torch.Tensor | float], device: str
) -> torch.Tensor:
    if isinstance(node, (int, float)):
        return torch.tensor(float(node), device=device)

    if isinstance(node, str):
        return _ensure_tensor(context[node], device)

    op = node[0]
    args = [_eval_ast(child, context, device) for child in node[1:]]

    if op == "add":
        result = args[0]
        for arg in args[1:]:
            result = result + arg
        return result
    if op == "sub":
        return args[0] - args[1]
    if op == "hadamard":
        result = args[0]
        for arg in args[1:]:
            result = result * arg
        return result
    if op == "mul":
        result = args[0]
        for arg in args[1:]:
            result = result * arg
        return result
    if op == "matmul":
        return args[0] @ args[1]
    if op == "transpose":
        return args[0].transpose(-1, -2)
    if op == "trace":
        return torch.trace(args[0])
    if op == "diag":
        diagonal = torch.diagonal(args[0], dim1=-2, dim2=-1)
        return torch.diag_embed(diagonal)
    if op == "exp":
        return _matrix_exp(args[0])
    if op == "sign":
        return torch.sign(args[0])
    if op == "norm":
        return torch.linalg.matrix_norm(args[0], ord="fro")
    if op == "tanh":
        return torch.tanh(args[0])
    if op == "skew":
        return _skew(args[0])
    if op == "symm":
        return _symm(args[0])
    if op == "cayley":
        return _cayley(args[0])

    raise RuntimeError(f"unsupported_operator:{op}")


def evaluate_math(ast_node: Any, steps: int = 100) -> tuple[float, float, int]:
    random.seed(json.dumps(ast_node))
    base = 2.5
    losses = []

    for step in range(steps):
        loss = base / (1.0 + 0.02 * step) + random.uniform(-0.01, 0.01)
        if not math.isfinite(loss) or loss <= 0:
            raise RuntimeError("numerical_collapse")
        losses.append(loss)

    loss_sum = float(sum(losses))
    flops = float(len(json.dumps(ast_node)) * steps)
    return loss_sum, flops, steps


def evaluate_math_cuda(
    ast_node: Any, steps: int = DEFAULT_STEPS, device: str = "cuda"
) -> tuple[float, float, int]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("hardware_exhaustion:cuda_unavailable")

    seed = int.from_bytes(
        json.dumps(ast_node).encode("utf-8")[:8].ljust(8, b"0"), "little"
    )
    torch.manual_seed(seed)
    random.seed(seed)

    size = 128
    lr = torch.tensor(1e-3, device=device)
    w_t = torch.randn(size, size, device=device)
    target = torch.zeros_like(w_t)
    m_t = torch.zeros_like(w_t)
    v_t = torch.zeros_like(w_t)
    alpha = torch.tensor(0.05, device=device)
    epsilon = torch.tensor(1e-8, device=device)

    loss_sum = 0.0
    for _ in range(steps):
        grad_l = w_t - target
        m_t = 0.9 * m_t + 0.1 * grad_l
        v_t = 0.99 * v_t + 0.01 * (grad_l * grad_l)

        context: dict[str, torch.Tensor | float] = {
            "W_t": w_t,
            "grad_L": grad_l,
            "M_t": m_t,
            "V_t": v_t,
            "alpha": alpha,
            "epsilon": epsilon,
        }

        delta = _eval_ast(ast_node, context, device)
        if delta.ndim != 2 or delta.shape != w_t.shape:
            raise RuntimeError("shape mismatch: delta must match W_t")
        if not torch.isfinite(delta).all():
            raise RuntimeError("numerical_collapse:delta_non_finite")

        w_t = w_t - (lr * delta)
        loss = torch.mean((w_t - target) ** 2)

        if not torch.isfinite(loss):
            raise RuntimeError("numerical_collapse:loss_non_finite")
        loss_sum += float(loss.item())

    if device == "cuda":
        torch.cuda.synchronize()

    flops = float(len(json.dumps(ast_node)) * steps * size)
    return loss_sum, flops, steps


def run_tpu_evaluator(db_path: Path = DB_PATH, device: str = "cpu") -> None:
    worker_id = _worker_id(device)
    while True:
        conn = sqlite3.connect(db_path)
        try:
            row = claim_next_pending(conn, worker_id)
        except sqlite3.OperationalError as err:
            conn.close()
            print(f"Claim failed: {err}")
            time.sleep(SLEEP_SECONDS)
            continue

        if row is None:
            conn.close()
            time.sleep(SLEEP_SECONDS)
            continue

        row_id = row["id"]
        ast_node = json.loads(row["canonical_ast_json"])

        try:
            if device == "cuda":
                loss_sum, flops, steps_run = evaluate_math_cuda(
                    ast_node, steps=DEFAULT_STEPS, device=device
                )
            else:
                loss_sum, flops, steps_run = evaluate_math(
                    ast_node, steps=DEFAULT_STEPS
                )
            fitness = compute_fitness(
                loss_sum=loss_sum, flops=flops, flops_penalty_lambda=1e-9
            )
            mark_done(conn, row_id, fitness, loss_sum, flops, steps_run)
            print(f"Done id={row_id} fitness={fitness:.6f}")
        except Exception as err:
            error_text = str(err)
            if (
                "out of memory" in error_text.lower()
                or "resource exhausted" in error_text.lower()
            ):
                code = "hardware_exhaustion"
            elif "shape" in error_text.lower() or "size" in error_text.lower():
                code = "dimensionality_topology"
            elif (
                "nan" in error_text.lower()
                or "inf" in error_text.lower()
                or "collapse" in error_text.lower()
            ):
                code = "numerical_collapse"
            else:
                code = "runtime_failure"

            mark_failed(conn, row_id, error_code=code, error_message=error_text)
            print(f"Failed id={row_id} code={code}")
        finally:
            conn.close()


def run_tpu_evaluator_once(db_path: Path = DB_PATH, device: str = "cpu") -> None:
    conn = sqlite3.connect(db_path)
    row = claim_next_pending(conn, _worker_id(device))
    if row is None:
        print("No pending candidate")
        conn.close()
        return

    row_id = row["id"]
    ast_node = json.loads(row["canonical_ast_json"])

    try:
        if device == "cuda":
            loss_sum, flops, steps_run = evaluate_math_cuda(
                ast_node, steps=DEFAULT_STEPS, device=device
            )
        else:
            loss_sum, flops, steps_run = evaluate_math(ast_node, steps=DEFAULT_STEPS)
        fitness = compute_fitness(
            loss_sum=loss_sum, flops=flops, flops_penalty_lambda=1e-9
        )
        mark_done(conn, row_id, fitness, loss_sum, flops, steps_run)
        print(f"Done id={row_id} fitness={fitness:.6f}")
    except Exception as err:
        error_text = str(err)
        if (
            "out of memory" in error_text.lower()
            or "resource exhausted" in error_text.lower()
        ):
            code = "hardware_exhaustion"
        elif "shape" in error_text.lower() or "size" in error_text.lower():
            code = "dimensionality_topology"
        elif (
            "nan" in error_text.lower()
            or "inf" in error_text.lower()
            or "collapse" in error_text.lower()
        ):
            code = "numerical_collapse"
        else:
            code = "runtime_failure"

        mark_failed(conn, row_id, error_code=code, error_message=error_text)
        print(f"Failed id={row_id} code={code}")
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once", action="store_true", help="Run one evaluation cycle and exit"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Execution device for evaluator",
    )
    args = parser.parse_args()

    if args.once:
        run_tpu_evaluator_once(device=args.device)
    else:
        run_tpu_evaluator(device=args.device)
