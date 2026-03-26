# Auto Math Research

This project implements an automated research system for discovering new mathematical optimizers using **Google Gemini 1.5 Flash** for mutation and evolutionary search, accelerated by NVIDIA CUDA.

## System Architecture

- **`auto-math-research`**: Main package
  - **`configs`**: Centralized configuration and API keys.
  - **`models`**: AST definitions, canonicalization (structure-invariant hashing), and symbolic validation.
  - **`training`**: Execution engine mapping ASTs to PyTorch operations.
  - **`optimizers`**: Interface to Gemini API (`mutator.py`) and evolutionary strategy logic.
  - **`utils`**: CUDA monitoring and enforcement tools.

## Setup & Requirements

### Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (Tested on RTX 2050).
- **Google Gemini API Key** (Set in `configs/research_config.py`).

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Initialize the SQLite database:
   ```bash
   python auto-math-research/run_search.py init
   ```

3. Seed the initial optimizer (SGD):
   ```bash
   python auto-math-research/scripts/seed.py
   ```

## Usage

### 1. Run the Mutator (Producer)
The mutator continuously polls the database for the best performing optimizer and uses **Gemini 1.5 Flash** to propose novel variations.
```bash
python auto-math-research/run_search.py mutator
```

### 2. Run the Worker (Consumer)
The worker polls for pending tasks and evaluates them on the GPU, enforcing strict CUDA execution for performance.
```bash
python auto-math-research/run_search.py worker --device cuda
```

### 3. Check Status
Monitor the queue size, best loss, and recent discoveries.
```bash
python auto-math-research/run_search.py status
```

## Implementation Details

For a deep dive into the code changes, refactoring decisions, and specific implementation of the CUDA enforcement and Gemini integration, please refer to:
[**IMPLEMENTATION_REPORT.md**](./IMPLEMENTATION_REPORT.md)

### Demo
Open `auto-math-research/demo.ipynb` to interact with the system in a notebook.
