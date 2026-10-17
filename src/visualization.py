from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot_magnetization_comparison(
    times: np.ndarray,
    exact_vals: np.ndarray,
    ideal_vals: np.ndarray,
    noisy_a: np.ndarray,
    noisy_b: np.ndarray,
    save_path: str = "results/figures/magnetization_comparison.png"
) -> None:
    """Generates the primary trajectory plot: Exact vs Ideal vs Processor A vs Processor B."""
    plt.figure(figsize=(9, 5.5))
    plt.plot(times, exact_vals, label="Exact Baseline (scipy.linalg.expm)", color="black", linewidth=2.5, zorder=4)
    plt.plot(times, ideal_vals, "--", label="Ideal Trotter Circuit (No Noise)", color="#2ca02c", linewidth=2, zorder=3)
    plt.plot(times, noisy_a, "o-", label="Processor A (5Q Linear, p=0.005 CX noise)", color="#1f77b4", markersize=5, zorder=2)
    plt.plot(times, noisy_b, "s-", label="Processor B (9Q Heavy-Hex, p=0.005 CX noise)", color="#d62728", markersize=5, zorder=1)
    
    plt.xlabel("Evolution Time $ (seconds)", fontsize=12)
    plt.ylabel(r"Expectation Value $\langle Z_0(t) \rangle$", fontsize=12)
    plt.title("PS-S2: Real-Time TFIM Spin Dynamics (4 Spins, J=1.0, g=1.0)", fontsize=13, fontweight="bold")
    plt.ylim(-1.1, 1.1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_hardware_overhead(
    labels: List[str],
    depths: List[int],
    cx_counts: List[int],
    swap_counts: List[int],
    save_path: str = "results/figures/circuit_overhead.png"
) -> None:
    """Generates bar charts comparing Depth, CX count, and SWAP overhead across architectures."""
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 5))
    
    rects1 = ax.bar(x - width, depths, width, label="Circuit Depth", color="#4c72b0")
    rects2 = ax.bar(x, cx_counts, width, label="Two-Qubit CX Count", color="#c44e52")
    rects3 = ax.bar(x + width, swap_counts, width, label="Injected SWAPs", color="#dd8452")
    
    ax.set_ylabel("Gate / Layer Count", fontsize=12)
    ax.set_title("Architectural Overhead: Processor A vs Processor B Transpilation", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    for rects in [rects1, rects2, rects3]:
        for r in rects:
            height = r.get_height()
            if height > 0:
                ax.annotate(f"{int(height)}",
                            xy=(r.get_x() + r.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)
                            
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_growth(
    times: np.ndarray,
    error_a: np.ndarray,
    error_b: np.ndarray,
    save_path: str = "results/figures/error_growth.png"
) -> None:
    """Plots absolute deviation over time to visualize noise-induced decoherence."""
    plt.figure(figsize=(8.5, 4.5))
    plt.plot(times, error_a, "o-", label="Processor A Absolute Error", color="#1f77b4")
    plt.plot(times, error_b, "s-", label="Processor B Absolute Error", color="#d62728")
    plt.xlabel("Evolution Time $ (seconds)", fontsize=12)
    plt.ylabel(r"$|\langle Z_0 \rangle_{exact} - \langle Z_0 \rangle_{noisy}|$", fontsize=12)
    plt.title("Error Accumulation Across Sliced Time Evolution", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
