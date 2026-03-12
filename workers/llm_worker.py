from __future__ import annotations

import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

from core.ast_canonicalize import canonical_json, canonicalize_ast
from core.ast_validation import ASTValidationError, ensure_matrix_update, validate_ast
from core.hash_id import ast_hash
from core.symbolic_sanitizer import sanitize_ast

DB_PATH = Path("optimizer_evolution.db")
SLEEP_SECONDS = 2


def get_best_parent(conn: sqlite3.Connection) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT id, canonical_ast_json, generation
        FROM optimizers
        WHERE status = 'done' AND objective_value IS NOT NULL
        ORDER BY objective_value DESC
        LIMIT 1
        """
    ).fetchone()


def mutate_via_llm(parent_ast: Any) -> Any:
    proposals = [
        ["hadamard", ["sign", "grad_L"], "M_t"],
        ["add", ["hadamard", "grad_L", "M_t"], ["mul", "alpha", "W_t"]],
        ["exp", ["symm", ["add", "W_t", ["mul", "alpha", "grad_L"]]]],
    ]
    return random.choice(proposals)


def insert_candidate(
    conn: sqlite3.Connection,
    parent_id: int,
    generation: int,
    raw_ast: Any,
    latex_formula: str = "",
) -> bool:
    validate_ast(raw_ast)
    ensure_matrix_update(raw_ast)
    sanitize_ast(raw_ast, expected_matrix_shape=(128, 128))

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
    while True:
        conn = sqlite3.connect(db_path)
        parent = get_best_parent(conn)

        if parent:
            parent_ast = json.loads(parent["canonical_ast_json"])
            try:
                candidate = mutate_via_llm(parent_ast)
                inserted = insert_candidate(
                    conn,
                    parent_id=parent["id"],
                    generation=parent["generation"] + 1,
                    raw_ast=candidate,
                    latex_formula="auto-generated",
                )
                if inserted:
                    print("Inserted candidate")
                else:
                    print("Duplicate candidate skipped")
            except ASTValidationError as err:
                print(f"Rejected mutation: {err}")
        else:
            print("No completed parent available")

        conn.close()
        time.sleep(SLEEP_SECONDS)


def run_mutator_once(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    parent = get_best_parent(conn)
    if not parent:
        print("No completed parent available")
        conn.close()
        return

    parent_ast = json.loads(parent["canonical_ast_json"])
    try:
        candidate = mutate_via_llm(parent_ast)
        inserted = insert_candidate(
            conn,
            parent_id=parent["id"],
            generation=parent["generation"] + 1,
            raw_ast=candidate,
            latex_formula="auto-generated",
        )
        print("Inserted candidate" if inserted else "Duplicate candidate skipped")
    except ASTValidationError as err:
        print(f"Rejected mutation: {err}")
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once", action="store_true", help="Run one mutation cycle and exit"
    )
    args = parser.parse_args()

    if args.once:
        run_mutator_once()
    else:
        run_mutator()
