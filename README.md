# hybrid-qpinn-cost

Code and result data comparing a classical physics-informed neural network (PINN) with hybrid
quantum-classical PINNs on the 1D diffusion equation `dT/dt = kz * d²T/dz²`, `z ∈ [0, 2]`, `t ∈ [0, 5]`.
The quantum circuit is simulated with PennyLane.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU only. No GPU is needed.

## Code

| File | What it does |
|---|---|
| `qcpinn_seawater.py` | Model definitions (`ClassicalDNN`, `QuantumLayer`, `HybridQNN`, `PINN`) and analytical solutions. The other scripts import it. |
| `qcpinn_parameter_count.py` | Counts trainable parameters per configuration, split into quantum and classical. |
| `qcpinn_timing_benchmark.py` | Stage timing, quantum vs. classical time share, qubit / depth / batch sweeps, simulator backend comparison, size-matched classical controls, drift check. |
| `qcpinn_convergence_benchmark.py` | Trains the classical PINN and one hybrid configuration on six scenarios (Dirichlet / Neumann / Robin, forward / inverse); records loss history and test error. |
| `plot_convergence_figures.py` | Loss-curve figures from the convergence runs. |
| `draw_circuits.py` | Circuit diagram and hybrid-network overview figures. |

Hybrid configurations:

| Name | Qubits | Variational layers |
|---|---|---|
| narrow | 2 | 2 |
| baseline | 4 | 2 |
| wide | 6 | 2 |
| deep | 4 | 4 |

## Reproducing the results

Run from the repository root.

```bash
python qcpinn_parameter_count.py

python qcpinn_timing_benchmark.py --tag my_run
python qcpinn_timing_benchmark.py --quick --out-root /tmp/tb_test   # 1-2 minute smoke test

python qcpinn_convergence_benchmark.py --config baseline --tag baseline_seed0
python qcpinn_convergence_benchmark.py --config narrow   --tag narrow_seed0 --paper-figures-dir ""
python qcpinn_convergence_benchmark.py --config wide     --tag wide_seed0   --paper-figures-dir ""
python qcpinn_convergence_benchmark.py --config deep     --tag deep_seed0   --paper-figures-dir ""
python qcpinn_convergence_benchmark.py --classical-adam-only --tag classical_adam_seed0 --paper-figures-dir ""
python plot_convergence_figures.py

python draw_circuits.py
```

Figures are written to `figures/`. Timing depends on the machine; the hardware and package versions
of each run are recorded in its `env.json`.

## Results

| Folder | Contents |
|---|---|
| `results/parameter_count/` | `parameter_count.csv` |
| `results/timing_benchmark/win_py314_run6/` | `summary.csv`, `report.md`, every timed call in `raw/`, environment in `env.json` |
| `results/accuracy_convergence/<config>_seed0/` | `summary.csv`, `report.md`, per-step loss histories in `raw/`, environment in `env.json` |

The `git_commit` field in `env.json` refers to the development repository the run was made in.
