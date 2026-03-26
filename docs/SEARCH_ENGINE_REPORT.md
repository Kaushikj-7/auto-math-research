# Implementation Report: Automated Math Research System

**Date:** March 26, 2026
**System:** Auto Math Research w/ CUDA & Gemini Integration

## 1. System Architecture Refactoring

The codebase was refactored from a flat structure into a modular Python package `auto-math-research` to support scalability and separation of concerns.

### New Directory Structure
- **`configs/`**: Centralized configuration (`ResearchConfig`) for DB paths, hyperparameters, and API keys.
- **`models/`**: Core data structures.
  - `ast_canonicalize.py`: Ensures `a + b` and `b + a` hash to the same ID.
  - `ast_validation.py`: Strict validation of allowed operators and variables.
  - `symbolic_sanitizer.py`: Shape inference to ensure all generated optimizers are valid for `(128, 128)` matrices.
- **`training/`**: Execution environment.
  - `execute_ast.py`: The runtime engine that maps AST nodes to PyTorch CUDA operations.
  - `worker.py`: The process that claims tasks, executes them, and logs metrics.
  - `kernels.py`: Custom CUDA kernels (placeholder for future C++ extensions).
- **`optimizers/`**: Search logic.
  - `mutator.py`: Interfaces with Gemini API to evolve optimizer ASTs.
  - `muon.py`: Reference implementation.
- **`data/`**: Database interactions (`queue.py`).
- **`utils/`**: Monitoring and logging (`cuda_monitor.py`).

## 2. Strict CUDA Enforcement & Efficiency

To meet the efficiency requirements, a strict enforcement layer was implemented to ensure no operations fallback to CPU.

### CudaEnforcer (`utils/cuda_monitor.py`)
- **Pre-flight Check**: Raises `RuntimeError` if `torch.cuda.is_available()` is False.
- **Optimization Flags**:
  - `cudnn.benchmark = True`: Enables auto-tuner for convolution algorithms.
  - `allow_tf32 = True`: Enables TensorFloat-32 on Ampere+ GPUs (like the RTX 2050) for faster matrix math.
- **Monitoring**: Real-time logging of GPU memory peaks and execution duration.

### Library Integration (`training/execute_ast.py`)
All mathematical operations in the AST are mapped to their specific CUDA-accelerated PyTorch counterparts:
- **BLAS**: `matmul` $\rightarrow$ `torch.matmul` (cuBLAS).
- **Solver**: `cayley` $\rightarrow$ `torch.linalg.solve` (cuSOLVER).
- **FFT**: `fft`, `ifft` $\rightarrow$ `torch.fft` (cuFFT).
- **Rand**: `random_normal` $\rightarrow$ `torch.randn` (cuRAND).
- **DNN**: Convolution operations verified against cuDNN backend.

## 3. LLM-Driven Mutation (Gemini)

The mutation engine was upgraded from a random baseline to a Large Language Model driven approach using **Google Gemini 1.5 Flash**.

### Implementation (`optimizers/mutator.py`)
1.  **Context**: The LLM is provided with the JSON AST of the currently best-performing optimizer.
2.  **Prompt Engineering**: A specialized system prompt defines the "Search Space" (allowed variables, unary ops, binary ops) and restricts output to valid JSON only.
3.  **Process**:
    - Query DB for best `done` task.
    - Send AST to Gemini.
    - Validate & Sanitize the returned AST.
    - Canonicalize & Hash.
    - Insert into DB as `pending`.

## 4. Execution Workflow

The system operates as an asynchronous producer-consumer loop backed by SQLite.

1.  **Initialization**:
    ```bash
    python auto-math-research/run_search.py init
    python auto-math-research/scripts/seed.py # Seeds SGD
    ```
2.  **Producer (Mutator)**:
    - Running in background.
    - Continuously reads best solution -> Prompts Gemini -> Writes new candidate to DB.
3.  **Consumer (Worker)**:
    - Running in background (CUDA device).
    - Claims `pending` row -> Reconstructs Pytorch Graph -> Executes on GPU -> Updates DB with Loss/FLOPS.

## 5. Current Status

- **Hardware Verified**: NVIDIA GeForce RTX 2050 (Driver 581.86).
- **Database**: Initialized and seeded with SGD.
- **Mutator**: Successfully generating candidates (some duplicates are efficiently skipped via unique hash constraints).
- **Worker**: Configured to run strict CUDA workloads.

### Next Steps
1.  Monitor `logs/experiment_metrics.csv` for convergence.
2.  Visualize the "Family Tree" of the evolved optimizers using the `parent_id` links in SQLite.
