from __future__ import annotations

import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any
import google.generativeai as genai

# Correct imports for root-level execution
from core.models.ast_canonicalize import canonical_json, canonicalize_ast
from core.models.ast_validation import ASTValidationError, ensure_matrix_update, validate_ast
from core.models.hash_id import ast_hash
from core.models.symbolic_sanitizer import sanitize_ast
from core.configs.research_config import ResearchConfig
from core.utils.compute_estimator import is_feasible, estimate_ast_flops

DB_PATH = ResearchConfig.DB_PATH
SLEEP_SECONDS = ResearchConfig.LLM_WORKER_SLEEP

# Initialize Gemini
if hasattr(ResearchConfig, "GEMINI_API_KEY"):
    genai.configure(api_key=ResearchConfig.GEMINI_API_KEY)
    # Primary and fallback models
    _MODELS = [
        genai.GenerativeModel("gemini-2.5-flash"),
        genai.GenerativeModel("gemini-2.0-flash"),
        genai.GenerativeModel("gemini-1.5-flash"),
        genai.GenerativeModel("gemini-flash-latest"),
        genai.GenerativeModel("gemini-1.5-pro"),
        genai.GenerativeModel("gemini-pro-latest"),
    ]
else:
    _MODELS = []
    print(
        "WARNING: Gemini API Key not found in ResearchConfig. Using random mutations."
    )


def get_best_parent(conn: sqlite3.Connection) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT id, canonical_ast_json, generation, flops_threshold, baseline_id
        FROM optimizers
        WHERE status = 'done' AND objective_value IS NOT NULL
        ORDER BY objective_value DESC
        LIMIT 1
        """
    ).fetchone()


def mutate_via_llm(parent_ast: Any) -> Any:
    if not _MODELS:
        # Fallback to random if no LLM
        return _random_mutation()

    prompt = f"""
    You are an expert mathematician designing optimization algorithms for deep learning.
    We represent update rules as JSON-based Abstract Syntax Trees (ASTs).
    
    The current best optimizer has this AST:
    {json.dumps(parent_ast)}

    Your goal is to propose a NOVEL, improved variation of this optimizer.
    
    Allowed Variables: "W_t", "grad_L", "M_t", "V_t", "alpha", "epsilon"
    Allowed Unary Ops: "transpose", "trace", "diag", "exp", "sign", "cayley", "skew", "symm", "tanh", "norm", "fft", "ifft", "random_normal"
    Allowed Binary Ops: "add", "sub", "hadamard", "matmul", "mul"
    
    The output must utilize matrix operations effectively on GPU.
    The AST format is: [OP, child1, child2, ...] or "VARIABLE" or NUMBER.
    Example: ["sub", "W_t", ["mul", "alpha", "grad_L"]] corresponds to W_t - alpha * grad_L.
    
    Respond ONLY with the JSON of the new AST. Do not utilize Markdown code blocks.
    """

    for model in _MODELS:
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Clean up if the model wraps in markdown
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            candidate = json.loads(text.strip())
            return candidate
        except Exception as e:
            print(f"LLM Mutation Failed for {model.model_name}: {e}")
            continue

    print("All LLM Mutations Failed. Falling back to random proposal.")
    return _random_mutation()


def _random_mutation() -> Any:
    # Improved random fallback with more variety to avoid immediate duplicates
    v = random.choice(["M_t", "V_t", "grad_L"])
    op = random.choice(["add", "sub", "hadamard"])
    scale = random.uniform(0.001, 0.1)
    proposals = [
        ["hadamard", ["sign", "grad_L"], v],
        [op, ["hadamard", "grad_L", v], ["mul", scale, "W_t"]],
        ["add", "W_t", ["mul", -scale, "grad_L"]],
        ["sub", "W_t", ["mul", "alpha", ["sign", "grad_L"]]],
    ]
    return random.choice(proposals)


def insert_candidate(
    conn: sqlite3.Connection,
    parent_id: int | None,
    generation: int,
    raw_ast: Any,
    latex_formula: str = "",
) -> bool:
    validate_ast(raw_ast)
    ensure_matrix_update(raw_ast)
    sanitize_ast(raw_ast, expected_matrix_shape=ResearchConfig.MATRIX_SHAPE)

    canonical_ast = canonicalize_ast(raw_ast)
    row = (
        parent_id,
        generation,
        json.dumps(raw_ast, separators=(",", ":")),
        canonical_json(canonical_ast),
        ast_hash(canonical_ast),
        latex_formula,
        "pending",
    )

    try:
        conn.execute(
            """
            INSERT INTO optimizers (
                parent_id, generation, raw_ast_json, canonical_ast_json, ast_hash,
                latex_formula, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def run_mutator(db_path: Path = DB_PATH) -> None:
    print(f"🔄 Starting Recursive Mutator on {db_path}")
    while True:
        try:
            conn = sqlite3.connect(db_path)
            parent = get_best_parent(conn)

            if parent:
                parent_ast = json.loads(parent["canonical_ast_json"])
                threshold = parent["flops_threshold"] or 1e11
                baseline_id = parent["baseline_id"] or parent["id"] if parent["generation"] == 0 else parent["baseline_id"]

                try:
                    candidate = mutate_via_llm(parent_ast)
                    
                    if is_feasible(candidate, threshold):
                        inserted = insert_candidate(
                            conn,
                            parent_id=parent["id"],
                            generation=parent["generation"] + 1,
                            raw_ast=candidate,
                            latex_formula="auto-generated",
                        )
                        if inserted:
                            conn.execute(
                                "UPDATE optimizers SET flops_threshold = ?, baseline_id = ? WHERE id = (SELECT max(id) FROM optimizers)",
                                (threshold, baseline_id)
                            )
                            conn.commit()
                            print(f"✨ Inserted Gen {parent['generation'] + 1} (Baseline: {baseline_id})")
                        else:
                            print("Duplicate candidate skipped")
                    else:
                        print(f"🚫 Proposed AST too complex ({estimate_ast_flops(candidate):.1e} flops). Discarding.")
                except ASTValidationError as err:
                    print(f"Rejected mutation: {err}")
            else:
                print("No parent found, waiting for researcher to seed...", end="\r")
            conn.close()
        except Exception as e:
            print(f"Mutator error: {e}")

        time.sleep(SLEEP_SECONDS)
