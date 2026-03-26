# Auto-Math Research: Evolutionary Optimizer Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![CUDA: 13.0](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)

An automated research system for discovering novel mathematical optimizers using **Google Gemini 2.5 Flash** for mutation and evolutionary search, accelerated by NVIDIA CUDA.

---

## 🔬 Research Findings & Experimental Iterations

This repository includes a specialized experimentation suite (`experiments/`) to evaluate the impact of different matrix operations on Transformer training.

### **Iteration 1: Orthogonalization Techniques**
- **SVD (Singular Value Decomposition):** Provided exact orthogonalization but suffered from high compute overhead (slowest execution).
- **Newton-Schulz (NS):** Matched SVD performance with significantly lower overhead. Identified as the optimal trade-off.
- **QR Decomposition & Sign Methods:** Resulted in training instability and poor convergence for deep Transformers.

### **Iteration 2: Newton-Schulz Optimization**
- **Iterations (Steps):** 7 steps provided the most stable convergence profile compared to 3 or 5 steps.
- **Hybrid Approaches:** SVD-initialization followed by NS steps did not significantly improve over pure NS.
- **Momentum:** High momentum (0.99) led to overshooting; **0.95** remains the robust default.

---

## 📂 Project Structure

- **`core/`**: The evolutionary search engine (mutators, workers, and database management).
- **`experiments/`**: Standalone research scripts for matrix operation benchmarking.
- **`results/`**: Logs, metrics, and visualization curves from research iterations.
- **`docs/`**: Detailed implementation reports and scaling guides.
- **`configs/`**: Centralized configuration for LLM and training parameters.

---

## 🚀 Portability & Getting Started

To ensure error-free execution across different environments:

1. **Clone & Setup:**
   ```bash
   git clone <repo_url>
   cd math_auto_research
   pip install -r requirements.txt
   ```

2. **Initialize Research Database:**
   ```bash
   python core/run_search.py init
   ```

3. **Run Experiments (Iterative Matrix Ops):**
   ```bash
   python experiments/matrix_ops_v1.py
   python experiments/matrix_ops_v2.py
   ```

4. **Start the Evolutionary Search Engine:**
   - **Terminal 1 (Mutator):** `python core/run_search.py mutator`
   - **Terminal 2 (Worker):** `python core/run_search.py worker --device cuda`

---

## 📈 Scaling for Professional Research

To scale this experimentation for large-scale language models (100M+ parameters):

1. **Dataset Scaling:** Replace synthetic data with **FineWeb-Edu** or **OpenWebText**. Use the `muon-optimizer-guide/data/` scripts for efficient chunking.
2. **Precision:** Transition from `float32` to **`bfloat16`** (using `torch.autocast`) to double throughput on Tensor Cores.
3. **Architecture:** Use **DistributedDataParallel (DDP)** or **Fully Sharded Data Parallel (FSDP)** for multi-GPU training.
4. **Optimized Kernels:** Compile the matrix operations using `torch.compile(mode='max-autotune')` to fuse the Newton-Schulz steps.
5. **Evolutionary Width:** Run multiple `worker` instances across different GPU nodes to explore the mutation space faster.

---

## 📜 Citations & Acknowledgments

This research incorporates and builds upon the excellent work found in the [**muon-optimizer-guide**](https://github.com/vukrosic/muon-optimizer-guide.git) repository by **vukrosic**. The foundational Muon implementations, Newton-Schulz steps, and LLM training configurations provided in that guide were instrumental in developing the automated search and matrix operation benchmarks within this project.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
