from typing import Any

def estimate_ast_flops(ast: Any, shape: tuple[int, int] = (1024, 1024)) -> float:
    """
    Recursively estimates the number of FLOPs required to execute an AST node.
    This is a heuristic for feasibility checking before running expensive operations.
    """
    if isinstance(ast, (str, int, float)):
        return 0.0
    
    if isinstance(ast, list) and len(ast) > 0:
        op = ast[0]
        n, m = shape
        # Base dimensions heuristics
        matrix_size = n * m
        
        flops = 0.0
        
        # Calculate children flops
        for child in ast[1:]:
            flops += estimate_ast_flops(child, shape)
            
        if op in ["add", "sub", "hadamard", "mul"]:
            flops += matrix_size
        elif op == "matmul":
            # Assuming n x m * m x n (rough estimate: 2 * n * m^2)
            flops += 2 * n * (m ** 2)
        elif op == "transpose":
            flops += 0 # Mem op, not strictly FLOPs, but could add penalty if needed
        elif op in ["trace", "diag"]:
            flops += min(n, m)
        elif op in ["exp", "sign", "tanh"]:
            flops += matrix_size * 10 # transcendental/branching approx
        elif op == "norm":
            flops += matrix_size * 2 # sq + sum + sqrt
        elif op in ["svd"]:
            # O(min(m n^2, m^2 n)) roughly
            flops += min(m * (n**2), (m**2) * n) * 10
        elif op in ["fft", "ifft"]:
            flops += matrix_size * 5 * 10 # N log N approx
            
        return flops
        
    return 0.0

def is_feasible(ast: Any, threshold: float = 1e11, shape: tuple[int, int] = (1024, 1024)) -> bool:
    """
    Checks if an AST is computationally feasible given a FLOP threshold.
    """
    est = estimate_ast_flops(ast, shape)
    return est <= threshold
