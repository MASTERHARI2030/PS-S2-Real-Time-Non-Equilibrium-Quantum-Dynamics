"""
src/problem.py
Defines the 4-spin Transverse-Field Ising Model (TFIM) Hamiltonian
and computes the exact classical ground truth using scipy.linalg.expm.
"""

import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import SparsePauliOp


def build_tfim_hamiltonian(num_spins: int = 4, j_coupling: float = 1.0, g_field: float = 1.0) -> SparsePauliOp:
    """
    Constructs the TFIM Hamiltonian:
    H = -J * sum(Z_i Z_{i+1}) - g * sum(X_i)
    """
    pauli_list = []
    
    # Nearest-neighbor ZZ interactions
    for i in range(num_spins - 1):
        pauli_str = ["I"] * num_spins
        pauli_str[i] = "Z"
        pauli_str[i + 1] = "Z"
        # Reverse string for Qiskit endianness (qubit 0 is rightmost)
        pauli_list.append(("".join(reversed(pauli_str)), -j_coupling))
        
    # Transverse magnetic field X
    for i in range(num_spins):
        pauli_str = ["I"] * num_spins
        pauli_str[i] = "X"
        pauli_list.append(("".join(reversed(pauli_str)), -g_field))
        
    return SparsePauliOp.from_list(pauli_list)


def compute_exact_magnetization(
    hamiltonian: SparsePauliOp,
    initial_state_str: str = "0000",
    time_points: np.ndarray = np.linspace(0.0, 2.0, 21),
    target_qubit: int = 0
) -> np.ndarray:
    """
    Computes exact <Z_target(t)> using scipy.linalg.expm.
    """
    h_matrix = hamiltonian.to_matrix()
    dim = 2 ** len(initial_state_str)
    
    # Statevector preparation
    idx = int(initial_state_str, 2)
    psi_0 = np.zeros(dim, dtype=complex)
    psi_0[idx] = 1.0
    
    # Target observable Z_0
    z_op = ["I"] * len(initial_state_str)
    z_op[target_qubit] = "Z"
    obs_matrix = SparsePauliOp.from_list([("".join(reversed(z_op)), 1.0)]).to_matrix()
    
    exact_z = []
    for t in time_points:
        # Unitary propagation U(t) = exp(-i * H * t)
        u_t = expm(-1j * h_matrix * t)
        psi_t = u_t @ psi_0
        expectation_val = np.real(np.vdot(psi_t, obs_matrix @ psi_t))
        exact_z.append(expectation_val)
        
    return np.array(exact_z)


if __name__ == "__main__":
    h = build_tfim_hamiltonian(num_spins=4, j_coupling=1.0, g_field=1.0)
    times = np.linspace(0.0, 2.0, 11)
    z_vals = compute_exact_magnetization(h, initial_state_str="0000", time_points=times)
    print("Exact <Z_0(t)> values:")
    for t, z in zip(times, z_vals):
        print(f"t = {t:.2f} s -> <Z_0> = {z:.4f}")
        