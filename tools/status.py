from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("optimizer_evolution.db")


def main(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Queue status:")
    for status in ["pending", "claimed", "done", "failed", "quarantined"]:
        count = cursor.execute(
            "SELECT COUNT(*) FROM optimizers WHERE status = ?", (status,)
        ).fetchone()[0]
        print(f"  {status:11s}: {count}")

    row = cursor.execute(
        """
        SELECT id, objective_value, generation, latex_formula
        FROM optimizers
        WHERE status = 'done'
        ORDER BY objective_value DESC
        LIMIT 1
        """
    ).fetchone()

    if row:
        print("\nBest candidate:")
        print(f"  id={row[0]} gen={row[2]} objective={row[1]:.6f}")
        print(f"  formula={row[3]}")
    else:
        print("\nBest candidate: none")

    conn.close()


if __name__ == "__main__":
    main()
