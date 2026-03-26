import torch
import torch.nn.functional as F
from configs.research_config import ResearchConfig

# Enforce strict CUDA ops
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CudaExecutionError(Exception):
    pass


def verify_cuda_stack():
    """Fail-fast check for all required libraries"""
    if DEVICE != "cuda":
        # If strict requirement
        if ResearchConfig.REQUIRE_CUDA:
            raise CudaExecutionError("CUDA required but not available.")
        return

    try:
        # Check BLAS (cuBLAS) - trivial matmul
        torch.matmul(
            torch.randn(10, 10, device=DEVICE), torch.randn(10, 10, device=DEVICE)
        )

        # Check Solver (cuSOLVER) - linear solve
        A = torch.randn(10, 10, device=DEVICE) + torch.eye(10, device=DEVICE)
        b = torch.randn(10, 1, device=DEVICE)
        torch.linalg.solve(A, b)

        # Check FFT (cuFFT)
        torch.fft.fft2(
            torch.complex(
                torch.randn(10, 10, device=DEVICE), torch.randn(10, 10, device=DEVICE)
            )
        )

        # Check Rand (cuRAND)
        torch.randn(100, device=DEVICE)

        # Check Sparse (cuSPARSE) - creation
        i = torch.tensor([[0, 1, 1], [2, 0, 2]], device=DEVICE)
        v = torch.tensor([3, 4, 5], dtype=torch.float32, device=DEVICE)
        torch.sparse_coo_tensor(i, v, [2, 4], device=DEVICE)

        # Check DNN (cuDNN) - convolution
        conv = torch.nn.Conv2d(1, 1, 3).to(DEVICE)
        conv(torch.randn(1, 1, 10, 10).to(DEVICE))

    except Exception as e:
        raise CudaExecutionError(f"CUDA library check failed: {e}")


def execute_ast(ast, context):
    """
    Executes an AST node in the given context (dict of variables).
    Enforces CUDA usage for all ops.
    """
    if isinstance(ast, (int, float)):
        return torch.tensor(ast, device=DEVICE, dtype=torch.float32)

    if isinstance(ast, str):
        if ast in context:
            val = context[ast]
            if (
                isinstance(val, torch.Tensor)
                and val.device.type != "cuda"
                and DEVICE == "cuda"
            ):
                val = val.to(DEVICE)
            return val
        raise ValueError(f"Unknown variable: {ast}")

    op = ast[0]
    args = [execute_ast(arg, context) for arg in ast[1:]]

    # Ensure args are on device
    args = [arg.to(DEVICE) if isinstance(arg, torch.Tensor) else arg for arg in args]

    # Map ops to optimized CUDA implementations
    if op == "add":
        return torch.add(args[0], args[1])  # tensor op
    elif op == "sub":
        return torch.sub(args[0], args[1])
    elif op == "mul":
        return torch.mul(args[0], args[1])
    elif op == "hadamard":  # same as mul
        return torch.mul(args[0], args[1])
    elif op == "matmul":
        return torch.matmul(args[0], args[1])  # cuBLAS
    elif op == "transpose":
        return args[0].transpose(-2, -1)
    elif op == "trace":
        return torch.trace(args[0])
    elif op == "diag":
        return torch.diag(args[0])
    elif op == "exp":
        return torch.matrix_exp(args[0])  # Optimized matrix exp
    elif op == "sign":
        return torch.sign(args[0])
    elif op == "tanh":
        return torch.tanh(args[0])
    elif op == "norm":
        return torch.norm(args[0])

    # Advanced Ops
    elif op == "cayley":
        # Cayley transform: (I - 0.5 A)^-1 (I + 0.5 A)
        # Using cuSOLVER via linalg.solve
        A = args[0]
        I = torch.eye(A.shape[-1], device=DEVICE, dtype=A.dtype)
        denom = I - 0.5 * A
        numer = I + 0.5 * A
        return torch.linalg.solve(denom, numer)

    elif op == "skew":
        return 0.5 * (args[0] - args[0].transpose(-1, -2))
    elif op == "symm":
        return 0.5 * (args[0] + args[0].transpose(-1, -2))

    elif op == "fft":
        return torch.fft.fft2(
            args[0]
        ).real  # Keeping real part for optimizer update context
    elif op == "ifft":
        return torch.fft.ifft2(args[0]).real

    elif op == "random_normal":
        # cuRAND
        shape = ResearchConfig.MATRIX_SHAPE
        return torch.randn(shape, device=DEVICE)

    else:
        raise ValueError(f"Unknown op: {op}")
