# Autonomous AI Scientist (MVP)

This repository implements a bounded, asynchronous optimizer-discovery engine with:

- Strict AST-only math mutation flow
- Symbolic sanitizer guardrails before queue insertion
- Canonical commutative dedup via AST hashing
- SQLite ledger with claim-safe evaluator queue
- Three-strike quarantine transition for repeated failures

## Quick start

1. Initialize DB and seed generation-0 baseline:

```powershell
E:/project-llm/.venv/Scripts/python.exe init_db.py
```

2. Run tests:

```powershell
E:/project-llm/.venv/Scripts/python.exe -m unittest discover -s tests -v
```

3. Run one-shot producer/evaluator cycles:

```powershell
E:/project-llm/.venv/Scripts/python.exe -m workers.llm_worker --once
E:/project-llm/.venv/Scripts/python.exe -m workers.tpu_worker --once
E:/project-llm/.venv/Scripts/python.exe tools/status.py
```

4. Run continuous workers:

```powershell
E:/project-llm/.venv/Scripts/python.exe -m workers.llm_worker
E:/project-llm/.venv/Scripts/python.exe -m workers.tpu_worker
```

## Guardrails implemented

- Grammar/operator and variable whitelist checks (`core/ast_validation.py`)
- Shape/domain symbolic preflight checks (`core/symbolic_sanitizer.py`)
- Canonical dedup for commutative/associative operators (`core/ast_canonicalize.py`)
- Atomic queue claim protocol (`core/queue_claims.py`)
- Three-strike quarantine on repeated failures (`core/queue_claims.py`)
