from pathlib import Path


class ResearchConfig:
    DB_PATH = Path("optimizer_evolution.db")
    LLM_WORKER_SLEEP = 2
    TPU_WORKER_SLEEP = 1
    DEFAULT_STEPS = 100

    # Validation
    MATRIX_SHAPE = (128, 128)

    # CUDA & Efficiency MANDATES
    REQUIRE_CUDA = True
    ENABLE_CUDNN_BENCHMARK = True
    ALLOW_TF32 = True  # efficient tensor ops

    # Logging
    LOG_DIR = Path("logs")
    METRICS_FILE = LOG_DIR / "monitoring_metrics.csv"

    # LLM Config
    GEMINI_API_KEY = "AIzaSyBZBi8F2I2beBq_Sxyp__IRrcOUVYyPy_k"
