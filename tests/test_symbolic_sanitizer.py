import unittest

from core.symbolic_sanitizer import SymbolicSanitizationError, sanitize_ast


class TestSymbolicSanitizer(unittest.TestCase):
    def test_accepts_valid_matrix_update(self):
        ast = ["hadamard", ["sign", "grad_L"], "M_t"]
        sanitize_ast(ast, expected_matrix_shape=(64, 64))

    def test_rejects_scalar_terminal_update(self):
        ast = ["trace", "W_t"]
        with self.assertRaises(SymbolicSanitizationError):
            sanitize_ast(ast, expected_matrix_shape=(64, 64))

    def test_rejects_non_square_for_matrix_exp(self):
        ast = ["exp", "W_t"]
        with self.assertRaises(SymbolicSanitizationError):
            sanitize_ast(ast, expected_matrix_shape=(64, 32))


if __name__ == "__main__":
    unittest.main()
