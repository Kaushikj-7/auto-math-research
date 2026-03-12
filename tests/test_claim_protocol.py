import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.ast_canonicalize import canonical_json
from core.hash_id import ast_hash
from core.queue_claims import claim_next_pending
from init_db import setup_db


class TestClaimProtocol(unittest.TestCase):
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
                "pending",
            ),
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_single_claim(self):
        conn1 = sqlite3.connect(self.db_path)
        row1 = claim_next_pending(conn1, "w1")
        self.assertIsNotNone(row1)

        conn2 = sqlite3.connect(self.db_path)
        row2 = claim_next_pending(conn2, "w2")
        self.assertIsNone(row2)

        conn1.close()
        conn2.close()


if __name__ == "__main__":
    unittest.main()
