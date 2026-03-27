import sqlite3
from pathlib import Path

from core.models.ast_canonicalize import canonical_json
from core.models.hash_id import ast_hash
from core.configs.research_config import ResearchConfig

DB_PATH = ResearchConfig.DB_PATH


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
            flops_threshold REAL,
            steps_run INTEGER,
            baseline_id INTEGER,
            error_code TEXT,
            error_message TEXT,
            artifact_path TEXT,
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


if __name__ == "__main__":
    setup_db()
