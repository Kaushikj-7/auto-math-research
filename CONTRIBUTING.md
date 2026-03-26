# Contributing to Auto-Math Research

Thank you for your interest in contributing! This project aims to automate the discovery of novel mathematical optimizers using evolutionary search and LLMs.

## 🚀 Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/math-auto-research.git
    cd math-auto-research
    ```
3.  **Create a branch** for your changes:
    ```bash
    git checkout -b feature/your-feature-name
    ```
4.  **Install development dependencies**:
    ```bash
    pip install -e ".[test]"
    ```

## 🛠️ Development Workflow

### Project Structure
- `core/`: The main evolutionary engine.
- `experiments/`: Benchmarking and research scripts.
- `muon-optimizer-guide/`: Reference implementations and training guides.

### Testing
We use `pytest` for unit testing. All new features should include corresponding tests in `core/tests/`.
Run tests locally before submitting a PR:
```bash
python -m pytest core/tests/
```

### Code Style
- Follow PEP 8 for Python code.
- Ensure your code is well-documented, especially for complex mathematical operations.
- Use descriptive commit messages.

## 📬 Submitting Changes

1.  **Push your changes** to your fork:
    ```bash
    git push origin feature/your-feature-name
    ```
2.  **Open a Pull Request** against the `prod` branch.
3.  **Describe your changes** clearly in the PR description, including the "why" behind any mathematical or architectural changes.
4.  **Ensure CI passes**: Our GitHub Actions pipeline will automatically run tests and check for coverage.

## 🧪 Research Contributions
If you are contributing new mutation strategies or optimizer templates:
1.  Verify the symbolic validity of your ASTs.
2.  Provide benchmark results (if possible) from the `experiments/` suite.

## 📜 License
By contributing, you agree that your contributions will be licensed under the project's MIT License.
