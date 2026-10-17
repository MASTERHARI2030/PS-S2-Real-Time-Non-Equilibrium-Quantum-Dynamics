import csv
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from qiskit import QuantumCircuit


def extract_circuit_metrics(transpiled_qc: QuantumCircuit, raw_cx_count: int = 30) -> Dict[str, Any]:
    """
    Extracts depth, CX count, total gates, and estimated SWAP count.
    Each SWAP inserted by the transpiler contributes 3 additional CX gates.
    """
    ops = dict(transpiled_qc.count_ops())
    depth = transpiled_qc.depth()
    cx_count = ops.get("cx", 0)
    total_gates = sum(ops.values())
    swap_count = max(0, (cx_count - raw_cx_count) // 3)
    
    return {
        "depth": depth,
        "cx_count": cx_count,
        "swap_count": swap_count,
        "total_gates": total_gates,
        "ops": ops
    }


def compute_mae(exact_vals: np.ndarray, noisy_vals: np.ndarray) -> float:
    """Computes Mean Absolute Error between exact baseline and noisy simulation."""
    return float(np.mean(np.abs(exact_vals - noisy_vals)))


def export_metrics_table(records: List[Dict[str, Any]], output_csv_path: str = "results/tables/transpilation_metrics.csv") -> None:
    """Exports metrics list to CSV using standard library."""
    path = Path(output_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    headers = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)


def export_dynamics_data(
    times: np.ndarray,
    exact_vals: np.ndarray,
    ideal_trotter: np.ndarray,
    noisy_a: np.ndarray,
    noisy_b: np.ndarray,
    output_csv_path: str = "results/tables/dynamics_data.csv"
) -> None:
    """Exports time-series trajectory values to CSV."""
    path = Path(output_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "exact_ground_truth", "ideal_trotter", "noisy_processor_A", "noisy_processor_B"])
        for t, ex, idl, na, nb in zip(times, exact_vals, ideal_trotter, noisy_a, noisy_b):
            writer.writerow([round(t, 4), round(ex, 6), round(idl, 6), round(na, 6), round(nb, 6)])
