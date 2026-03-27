import os
import json
from pathlib import Path
from datetime import datetime

# Assume muon-optimizer-guide is mapped directly or relative
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "muon-optimizer-guide"))

try:
    from utils.plot_loss import plot_loss
except ImportError:
    # Fallback if path doesn't resolve cleanly
    def plot_loss(*args, **kwargs):
        pass

def save_experiment_results(
    experiment_id: int,
    metrics: dict,
    history: dict,
    baseline_id: int = None,
    output_dir: str = "results"
) -> str:
    """
    Saves metrics and generates a researcher graph comparing with baseline.
    Returns the artifact path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"exp_{experiment_id}_{timestamp}"
    
    metrics_file = out_path / f"{base_name}_metrics.json"
    plot_file = out_path / f"{base_name}_plot.png"
    
    full_data = {
        "experiment_id": experiment_id,
        "baseline_id": baseline_id,
        "metrics": metrics,
        "history": history
    }
    
    with open(metrics_file, "w") as f:
        json.dump(full_data, f, indent=2)
        
    baseline_file = None
    if baseline_id is not None:
        # Simplistic lookup for the baseline metrics file
        # In a robust setup, we'd fetch the artifact_path from the DB
        # This assumes baselines are in the same folder
        matches = list(out_path.glob(f"exp_{baseline_id}_*_metrics.json"))
        if matches:
            baseline_file = str(matches[0])
            
    try:
        plot_loss(
            str(metrics_file),
            str(plot_file),
            title=f"Experiment {experiment_id} vs Baseline {baseline_id}",
            baseline_file=baseline_file
        )
    except Exception as e:
        print(f"Failed to plot graph for exp {experiment_id}: {e}")
        
    return str(metrics_file)
