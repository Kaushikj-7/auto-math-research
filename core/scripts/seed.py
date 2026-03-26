import json
import sqlite3
from pathlib import Path
from models.ast_canonicalize import canonical_json, canonicalize_ast
from models.hash_id import ast_hash
from configs.research_config import ResearchConfig

DB_PATH = ResearchConfig.DB_PATH

INITIAL_OPTIMIZER = ["sub", "W_t", ["mul", "alpha", "grad_L"]]  # SGD


def seed_db(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)

    # Check if already seeded
    count = conn.execute("SELECT COUNT(*) FROM optimizers").fetchone()[0]
    if count > 0:
        print("Database already contains optimizers. Skipping seed.")
        conn.close()
        return

    raw_ast = INITIAL_OPTIMIZER
    canonical_ast = canonicalize_ast(raw_ast)

    row = (
        None,  # parent_id
        0,  # generation
        json.dumps(raw_ast, separators=(",", ":")),
        canonical_json(canonical_ast),
        ast_hash(canonical_ast),
        "W_t - alpha * grad_L",  # latex_formula
        "done",  # Mark as done so it can be a parent immediately
        0.1,  # Initial objective_value (baseline)
    )

    try:
        conn.execute(
            """
            INSERT INTO optimizers (
                parent_id, generation, raw_ast_json, canonical_ast_json, ast_hash,
                latex_formula, status, objective_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        conn.commit()
        print("Seeded database with initial SGD optimizer.")
    except sqlite3.IntegrityError as e:
        print(f"Error seeding database: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    seed_db()
