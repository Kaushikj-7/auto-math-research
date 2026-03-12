import unittest

from core.ast_canonicalize import canonical_hash
from core.ast_validation import ASTValidationError, ensure_matrix_update, validate_ast


class TestCanonicalization(unittest.TestCase):
    def test_commutative_mul_hash_equal(self):
        a = ["mul", "grad_L", "M_t"]
        b = ["mul", "M_t", "grad_L"]
        self.assertEqual(canonical_hash(a), canonical_hash(b))

    def test_non_commutative_sub_hash_not_equal(self):
        a = ["sub", "grad_L", "M_t"]
        b = ["sub", "M_t", "grad_L"]
        self.assertNotEqual(canonical_hash(a), canonical_hash(b))

    def test_ast_validation_blocks_unknown_symbol(self):
        with self.assertRaises(ASTValidationError):
            validate_ast(["add", "unknown_tensor", "M_t"])

    def test_update_must_be_matrix(self):
        with self.assertRaises(ASTValidationError):
            ensure_matrix_update(["trace", "W_t"])


if __name__ == "__main__":
    unittest.main()
