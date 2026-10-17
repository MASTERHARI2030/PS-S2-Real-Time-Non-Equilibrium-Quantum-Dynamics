import json
import numpy as np
from qiskit_aer import AerSimulator

from src.problem import build_tfim_hamiltonian, compute_exact_magnetization
from src.processors import load_processor_spec, get_processor_coupling_map, build_synthetic_noise_model
from src.routing import build_full_trotter_circuit, transpile_for_processor
from src.metrics import extract_circuit_metrics, compute_mae, export_metrics_table, export_dynamics_data
from src.visualization import plot_magnetization_comparison, plot_hardware_overhead, plot_error_growth


def expectation_from_counts(counts: dict, shots: int = 4096) -> float:
    # Target qubit 0 measured into classical bit 0
    c0 = counts.get("0", 0)
    c1 = counts.get("1", 0)
    return (c0 - c1) / shots


def main():
    print("=" * 60)
    print("Running PS-S2 End-to-End Simulation Pipeline")
    print("=" * 60)

    # 1. Classical Exact Ground Truth
    num_spins = 4
    j_val, g_val = 1.0, 1.0
    times = np.linspace(0.0, 2.0, 11)
    
    print("[1/5] Calculating exact theoretical baseline using scipy.linalg.expm...")
    hamiltonian = build_tfim_hamiltonian(num_spins, j_coupling=j_val, g_field=g_val)
    exact_vals = compute_exact_magnetization(hamiltonian, initial_state_str="0000", time_points=times)

    # 2. Setup Processors & Noise Models
    print("[2/5] Loading Processor A (5Q) and Processor B (9Q) specifications...")
    spec_a = load_processor_spec("processors/processor_A.json")
    spec_b = load_processor_spec("processors/processor_B.json")
    cmap_a = get_processor_coupling_map(spec_a)
    cmap_b = get_processor_coupling_map(spec_b)
    
    noise_model = build_synthetic_noise_model(p1q=0.001, p2q=0.005, p_ro=0.01)
    sim_ideal = AerSimulator()
    sim_noisy = AerSimulator(noise_model=noise_model)
    shots = 4096

    # 3. Simulate Dynamics over Time
    print(f"[3/5] Executing quantum circuits across {len(times)} time steps (shots={shots})...")
    ideal_trotter = []
    noisy_a = []
    noisy_b = []
    
    # Store benchmark transpiled circuits at t=1.0 for metrics
    bench_qc_raw = None
    bench_qc_a = None
    bench_qc_b_opt = None
    bench_qc_b_naive = None

    for idx, t in enumerate(times):
        steps = max(1, int(t * 5)) if t > 0 else 0
        raw_qc = build_full_trotter_circuit(num_spins=4, total_time=t, steps=steps, measure_target_qubit=0)
        
        # Ideal simulation
        t_ideal = transpile_for_processor(raw_qc, cmap_a, optimization_level=0)
        res_ideal = sim_ideal.run(t_ideal, shots=shots).result()
        ideal_trotter.append(expectation_from_counts(res_ideal.get_counts(), shots))
        
        # Processor A (Optimal linear mapping: [0, 1, 2, 3])
        t_a = transpile_for_processor(raw_qc, cmap_a, initial_layout=[0, 1, 2, 3], optimization_level=1)
        res_a = sim_noisy.run(t_a, shots=shots).result()
        noisy_a.append(expectation_from_counts(res_a.get_counts(), shots))
        
        # Processor B (Layout: [0, 1, 2, 3])
        t_b = transpile_for_processor(raw_qc, cmap_b, initial_layout=[0, 1, 2, 3], optimization_level=1)
        res_b = sim_noisy.run(t_b, shots=shots).result()
        noisy_b.append(expectation_from_counts(res_b.get_counts(), shots))
        
        if idx == 5: # t = 1.0 s reference point
            bench_qc_raw = raw_qc
            bench_qc_a = t_a
            bench_qc_b_opt = t_b
            # Naive/scattered layout on Processor B forcing SWAP routing
            bench_qc_b_naive = transpile_for_processor(raw_qc, cmap_b, initial_layout=[0, 7, 4, 3], optimization_level=1)

    ideal_trotter = np.array(ideal_trotter)
    noisy_a = np.array(noisy_a)
    noisy_b = np.array(noisy_b)

    # 4. Compute Metrics & Error Bounds
    print("[4/5] Computing architectural metrics and exporting tables...")
    raw_metrics = extract_circuit_metrics(bench_qc_raw, raw_cx_count=bench_qc_raw.count_ops().get("cx", 0))
    metrics_a = extract_circuit_metrics(bench_qc_a, raw_cx_count=raw_metrics["cx_count"])
    metrics_b_opt = extract_circuit_metrics(bench_qc_b_opt, raw_cx_count=raw_metrics["cx_count"])
    metrics_b_naive = extract_circuit_metrics(bench_qc_b_naive, raw_cx_count=raw_metrics["cx_count"])

    mae_ideal = compute_mae(exact_vals, ideal_trotter)
    mae_a = compute_mae(exact_vals, noisy_a)
    mae_b = compute_mae(exact_vals, noisy_b)

    records = [
        {"configuration": "Raw Trotter (Ideal)", "layout": "Virtual [0-3]", "depth": raw_metrics["depth"], "cx_count": raw_metrics["cx_count"], "swap_count": 0, "mae_vs_exact": round(mae_ideal, 4)},
        {"configuration": "Processor A (5Q)", "layout": "[0, 1, 2, 3] Optimal", "depth": metrics_a["depth"], "cx_count": metrics_a["cx_count"], "swap_count": metrics_a["swap_count"], "mae_vs_exact": round(mae_a, 4)},
        {"configuration": "Processor B (9Q)", "layout": "[0, 1, 2, 3] Direct Path", "depth": metrics_b_opt["depth"], "cx_count": metrics_b_opt["cx_count"], "swap_count": metrics_b_opt["swap_count"], "mae_vs_exact": round(mae_b, 4)},
        {"configuration": "Processor B (9Q)", "layout": "[0, 7, 4, 3] Sparse Bottleneck", "depth": metrics_b_naive["depth"], "cx_count": metrics_b_naive["cx_count"], "swap_count": metrics_b_naive["swap_count"], "mae_vs_exact": "N/A (Routing Benchmark)"}
    ]
    export_metrics_table(records)
    export_dynamics_data(times, exact_vals, ideal_trotter, noisy_a, noisy_b)

    # 5. Render Figures
    print("[5/5] Generating publication figures in results/figures/...")
    plot_magnetization_comparison(times, exact_vals, ideal_trotter, noisy_a, noisy_b)
    
    plot_hardware_overhead(
        labels=["Raw Circuit", "Proc A (Opt)", "Proc B (Direct)", "Proc B (Sparse)"],
        depths=[raw_metrics["depth"], metrics_a["depth"], metrics_b_opt["depth"], metrics_b_naive["depth"]],
        cx_counts=[raw_metrics["cx_count"], metrics_a["cx_count"], metrics_b_opt["cx_count"], metrics_b_naive["cx_count"]],
        swap_counts=[0, metrics_a["swap_count"], metrics_b_opt["swap_count"], metrics_b_naive["swap_count"]]
    )
    
    error_a = np.abs(exact_vals - noisy_a)
    error_b = np.abs(exact_vals - noisy_b)
    plot_error_growth(times, error_a, error_b)

    print("=" * 60)
    print("SUCCESS: All benchmarks completed, figures and tables saved.")
    print(f"Processor A MAE: {mae_a:.4f} | Processor B MAE: {mae_b:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
