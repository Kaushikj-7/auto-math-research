import torch
import torch.backends.cudnn as cudnn
import torch.backends.cuda as cuda
import os
import psutil
import time
import subprocess
from typing import Dict, Any


class CudaEnforcer:
    @staticmethod
    def enforce_env():
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CRITICAL: CUDA is mandatory for this research but not available."
            )

        # Enforce performance flags
        cudnn.benchmark = True
        cudnn.enabled = True
        cuda.matmul.allow_tf32 = True  # Utilize Tensor Cores
        cudnn.allow_tf32 = True

        # Check for sparse support (implicit in PyTorch but verify version)
        if not hasattr(torch, "sparse"):
            print("WARNING: Sparse tensor support not found.")

        print(f"CUDA Enforced: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN Version: {cudnn.version()}")
        print(f"PyTorch Version: {torch.__version__}")

    @staticmethod
    def get_gpu_metrics(device_index=0) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {}

        stats = {}
        try:
            # Memory Stats
            mem_stats = torch.cuda.memory_stats(device_index)
            stats["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device_index)
            stats["cuda_reserved_bytes"] = torch.cuda.memory_reserved(device_index)
            stats["cuda_peak_memory"] = mem_stats.get("allocated_bytes.all.peak", 0)

            # Utilization (via nvidia-smi if available, parsed quickly)
            # This is overhead-heavy so use sparingly or use pynvml if installed.
            # Fallback to simple memory checks for low overhead.
        except Exception as e:
            stats["error"] = str(e)

        return stats


class ExperimentMonitor:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "experiment_metrics.csv")
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                f.write(
                    "timestamp,optimizer_id,generation,loss,flops,time_ms,gpu_mem_peak,library_usage\n"
                )

    def log_experiment(
        self,
        optimizer_id,
        generation,
        metrics: Dict[str, Any],
        library_metadata: Dict[str, str],
    ):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        loss = metrics.get("loss", 0.0)
        flops = metrics.get("flops", 0.0)
        duration = metrics.get("duration_ms", 0.0)
        mem = metrics.get("cuda_peak_memory", 0)
        libs = ";".join([f"{k}:{v}" for k, v in library_metadata.items()])

        with open(self.csv_path, "a") as f:
            f.write(
                f"{timestamp},{optimizer_id},{generation},{loss},{flops},{duration},{mem},{libs}\n"
            )
