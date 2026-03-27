from __future__ import annotations

import sqlite3
from typing import Optional


CLAIM_QUERY = """
SELECT id, generation, raw_ast_json, canonical_ast_json, flops_threshold, baseline_id
FROM optimizers
WHERE status = 'pending'
ORDER BY created_at ASC
LIMIT 1
"""


def claim_next_pending(
    conn: sqlite3.Connection, worker_id: str
) -> Optional[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(CLAIM_QUERY).fetchone()
    if row is None:
        conn.execute("COMMIT")
        return None

    conn.execute(
        """
        UPDATE optimizers
        SET status = 'claimed', worker_id = ?, claimed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (worker_id, row["id"]),
    )
    conn.execute("COMMIT")
    return row


def mark_done(
    conn: sqlite3.Connection,
    row_id: int,
    objective_value: float,
    loss_sum: float,
    flops_estimate: float,
    steps_run: int,
    artifact_path: str = None,
) -> None:
    conn.execute(
        """
        UPDATE optimizers
        SET status = 'done', objective_value = ?, loss_sum = ?, flops_estimate = ?, steps_run = ?, artifact_path = ?, completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (objective_value, loss_sum, flops_estimate, steps_run, artifact_path, row_id),
    )
    conn.commit()


def mark_failed(
    conn: sqlite3.Connection,
    row_id: int,
    error_code: str,
    error_message: str,
) -> None:
    conn.execute(
        """
        UPDATE optimizers
        SET status = CASE WHEN strike_count + 1 >= 3 THEN 'quarantined' ELSE 'failed' END,
            objective_value = 0.0, error_code = ?, error_message = ?,
            strike_count = strike_count + 1, completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (error_code, error_message[:500], row_id),
    )
    conn.commit()
