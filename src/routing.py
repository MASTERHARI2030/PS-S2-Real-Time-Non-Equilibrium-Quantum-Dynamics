import sys
from pathlib import Path
from typing import List, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.processors import load_processor_spec, get_processor_coupling_map
except ImportError:
    from processors import load_processor_spec, get_processor_coupling_map


def build_trotter_step_circuit(
    num_spins: int = 4,
    dt: float = 0.1,
    j_coupling: float = 1.0,
    g_field: float = 1.0
) -> QuantumCircuit:
    qc = QuantumCircuit(num_spins, name="Trotter_Step")
    for i in range(num_spins - 1):
        qc.cx(i, i + 1)
        qc.rz(2.0 * j_coupling * dt, i + 1)
        qc.cx(i, i + 1)
    for i in range(num_spins):
        qc.rx(2.0 * g_field * dt, i)
    return qc


def build_full_trotter_circuit(
    num_spins: int = 4,
    total_time: float = 1.0,
    steps: int = 5,
    j_coupling: float = 1.0,
    g_field: float = 1.0,
    initial_state_str: str = "0000",
    measure_target_qubit: Optional[int] = 0
) -> QuantumCircuit:
    qc = QuantumCircuit(num_spins, 1 if measure_target_qubit is not None else 0)
    for idx, bit in enumerate(initial_state_str):
        if bit == "1":
            qc.x(idx)
    if steps > 0 and total_time > 0:
        dt = total_time / steps
        step_circuit = build_trotter_step_circuit(num_spins, dt, j_coupling, g_field)
        for _ in range(steps):
            qc.compose(step_circuit, inplace=True)
    if measure_target_qubit is not None:
        qc.measure(measure_target_qubit, 0)
    return qc


def transpile_for_processor(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    basis_gates: Optional[List[str]] = None,
    initial_layout: Optional[List[int]] = None,
    optimization_level: int = 1,
    seed: int = 42
) -> QuantumCircuit:
    if basis_gates is None:
        basis_gates = ["rz", "sx", "x", "cx"]
    return transpile(
        circuit,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        initial_layout=initial_layout,
        optimization_level=optimization_level,
        seed_transpiler=seed
    )


if __name__ == "__main__":
    spec_a = load_processor_spec("processors/processor_A.json")
    spec_b = load_processor_spec("processors/processor_B.json")
    cmap_a = get_processor_coupling_map(spec_a)
    cmap_b = get_processor_coupling_map(spec_b)

    raw_qc = build_full_trotter_circuit(num_spins=4, total_time=1.0, steps=5)
    print(f"Raw circuit depth: {raw_qc.depth()}, gate count: {dict(raw_qc.count_ops())}")

    qc_a = transpile_for_processor(raw_qc, cmap_a, initial_layout=[0, 1, 2, 3])
    print(f"Processor A transpiled depth: {qc_a.depth()}, ops: {dict(qc_a.count_ops())}")

    qc_b = transpile_for_processor(raw_qc, cmap_b, initial_layout=[0, 1, 2, 3])
    print(f"Processor B transpiled depth: {qc_b.depth()}, ops: {dict(qc_b.count_ops())}")
