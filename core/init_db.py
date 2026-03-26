import sqlite3
from pathlib import Path

from core.models.ast_canonicalize import canonical_json
from core.models.hash_id import ast_hash

DB_PATH = Path("optimizer_evolution.db")


def setup_db(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS optimizers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER,
            generation INTEGER NOT NULL,
            raw_ast_json TEXT NOT NULL,
            canonical_ast_json TEXT NOT NULL,
            ast_hash TEXT NOT NULL UNIQUE,
            latex_formula TEXT,
            status TEXT NOT NULL CHECK (status IN ('pending', 'claimed', 'done', 'failed', 'quarantined')),
            strike_count INTEGER NOT NULL DEFAULT 0,
            objective_value REAL,
            loss_sum REAL,
            flops_estimate REAL,
            steps_run INTEGER,
            error_code TEXT,
            error_message TEXT,
            worker_id TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            claimed_at DATETIME,
            completed_at DATETIME,
            FOREIGN KEY(parent_id) REFERENCES optimizers(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_optimizers_status
        ON optimizers(status)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_optimizers_generation
        ON optimizers(generation)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_optimizers_objective
        ON optimizers(objective_value)
        """
    )

    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_optimizers_updated_at
        AFTER UPDATE ON optimizers
        FOR EACH ROW
        BEGIN
            UPDATE optimizers SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        """
    )

    count = cursor.execute("SELECT COUNT(*) FROM optimizers").fetchone()[0]
    if count == 0:
        lbga_seed_ast = [
            "exp",
            [
                "add",
                ["cayley", "M_t"],
                [
                    "mul",
                    ["tanh", ["mul", "alpha", ["norm", "grad_L"]]],
                    ["symm", "W_t"],
                ],
            ],
        ]
        cursor.execute(
            """
            INSERT INTO optimizers (
                parent_id, generation, raw_ast_json, canonical_ast_json, ast_hash,
                latex_formula, status, objective_value, loss_sum, steps_run
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                0,
                0,
                canonical_json(lbga_seed_ast),
                canonical_json(lbga_seed_ast),
                ast_hash(lbga_seed_ast),
                "K_new = exp(cayley(M_t) + tanh(alpha * norm(grad_L)) * symm(W_t))",
                "done",
                0.5,
                2.0,
                100,
            ),
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup_db()
    print(f"Initialized {DB_PATH}")
