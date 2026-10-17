import json
from pathlib import Path
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


def load_processor_spec(json_path: str) -> dict:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {json_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_processor_coupling_map(spec: dict) -> CouplingMap:
    return CouplingMap(spec["coupling_map"])


def build_synthetic_noise_model(
    p1q: float = 0.001,
    p2q: float = 0.005,
    p_ro: float = 0.01
) -> NoiseModel:
    noise_model = NoiseModel(basis_gates=["rz", "sx", "x", "cx"])
    err_1q = depolarizing_error(p1q, 1)
    noise_model.add_all_qubit_quantum_error(err_1q, ["sx", "x", "rz"])
    err_2q = depolarizing_error(p2q, 2)
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx"])
    ro_matrix = [[1.0 - p_ro, p_ro], [p_ro, 1.0 - p_ro]]
    err_ro = ReadoutError(ro_matrix)
    noise_model.add_all_qubit_readout_error(err_ro)
    return noise_model


if __name__ == "__main__":
    spec_a = load_processor_spec("processors/processor_A.json")
    spec_b = load_processor_spec("processors/processor_B.json")
    cmap_a = get_processor_coupling_map(spec_a)
    cmap_b = get_processor_coupling_map(spec_b)
    print(f"Loaded {spec_a['name']}: {spec_a['num_qubits']} qubits, {len(cmap_a.get_edges())} directed edges")
    print(f"Loaded {spec_b['name']}: {spec_b['num_qubits']} qubits, {len(cmap_b.get_edges())} directed edges")
    noise = build_synthetic_noise_model()
    print("Synthetic noise model created successfully.")
