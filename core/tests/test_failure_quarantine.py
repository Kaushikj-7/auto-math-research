import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.ast_canonicalize import canonical_json
from core.hash_id import ast_hash
from core.queue_claims import mark_failed
from init_db import setup_db


class TestFailureQuarantine(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        setup_db(self.db_path)

        conn = sqlite3.connect(self.db_path)
        ast = ["hadamard", ["sign", "grad_L"], "M_t"]
        conn.execute(
            """
            INSERT INTO optimizers (parent_id, generation, raw_ast_json, canonical_ast_json, ast_hash, latex_formula, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                0,
                1,
                json.dumps(ast),
                canonical_json(ast),
                ast_hash(ast),
                "seed",
                "claimed",
            ),
        )
        self.row_id = conn.execute("SELECT MAX(id) FROM optimizers").fetchone()[0]
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_quarantine_on_third_strike(self):
        conn = sqlite3.connect(self.db_path)
        mark_failed(conn, self.row_id, "numerical_collapse", "nan")
        mark_failed(conn, self.row_id, "numerical_collapse", "nan")
        mark_failed(conn, self.row_id, "numerical_collapse", "nan")

        status, strikes = conn.execute(
            "SELECT status, strike_count FROM optimizers WHERE id = ?", (self.row_id,)
        ).fetchone()
        self.assertEqual(status, "quarantined")
        self.assertEqual(strikes, 3)
        conn.close()


if __name__ == "__main__":
    unittest.main()
