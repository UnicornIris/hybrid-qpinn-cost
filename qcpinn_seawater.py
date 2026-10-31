"""
QCPINN for Seawater Temperature Diffusion Equation
Extends: Han et al. (2026), Journal of Oceanology and Limnology
Architecture: Classical pre-processor → Variational Quantum Circuit → Classical post-processor
Reference quantum circuit design: Afrah et al. (2025), arXiv:2503.16678

PDE:  dT/dt = kz * d²T/dz²
Domain: z ∈ [0, 2], t ∈ [0, 5]
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import time

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")

Z_MIN, Z_MAX = 0.0, 2.0
T_MIN, T_MAX = 0.0, 5.0
KZ_TRUE = 0.05

# Normalize (z,t) → [-1, 1] for AngleEmbedding (avoids RX angle aliasing)
def normalize_input(zt: torch.Tensor) -> torch.Tensor:
    z_norm = 2.0 * (zt[:, 0:1] - Z_MIN) / (Z_MAX - Z_MIN) - 1.0
    t_norm = 2.0 * (zt[:, 1:2] - T_MIN) / (T_MAX - T_MIN) - 1.0
    return torch.cat([z_norm, t_norm], dim=1)


# ── Analytical solutions (same as classical PINN) ─────────────────────────────
def T_dirichlet(z, t):
    return (np.exp(-KZ_TRUE * (np.pi / 2) ** 2 * t) * np.sin(np.pi * z / 2)
            + np.exp(-KZ_TRUE * np.pi ** 2 * t) * np.sin(np.pi * z))

def T_neumann(z, t):
    return (np.exp(-KZ_TRUE * (np.pi / 2) ** 2 * t) * np.cos(np.pi * z / 2)
            + np.exp(-KZ_TRUE * np.pi ** 2 * t) * np.cos(np.pi * z))

def T_robin(z, t):
    return (np.exp(KZ_TRUE * t - z)
            + np.exp(-KZ_TRUE * (np.pi / 2) ** 2 * t)
              * (np.pi / 2 * np.cos(np.pi * z / 2) - np.sin(np.pi * z / 2))
            + np.exp(-KZ_TRUE * np.pi ** 2 * t)
              * (np.pi * np.cos(np.pi * z) - np.sin(np.pi * z)))


# ── Quantum Layer ──────────────────────────────────────────────────────────────
class QuantumLayer(nn.Module):
    """
    Variational Quantum Circuit using the 'layered' ansatz from Afrah et al.

    Circuit structure per layer:
        AngleEmbedding(x)          — encode classical features as RX rotations
        RZ(θ) RX(θ) on each qubit — single-qubit rotations (pre-entanglement)
        CNOT(i → i+1 mod n)       — ring entanglement
        RX(θ) RZ(θ) on each qubit — single-qubit rotations (post-entanglement)
    Output: PauliZ expectation value <Z_i> ∈ [-1, 1] for each qubit
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, init_std: float = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Each layer: n_qubits × 4 parameters (RZ, RX, RX, RZ per qubit)
        self.params = nn.Parameter(
            torch.empty(n_layers, n_qubits * 4, dtype=torch.float32)
        )
        if init_std is None:
            nn.init.xavier_normal_(self.params)
        else:
            # Small-scale init to avoid barren-plateau-like flat gradients
            # from angles spread too wide by xavier_normal_.
            nn.init.normal_(self.params, mean=0.0, std=init_std)

        dev = qml.device("default.qubit", wires=n_qubits)
        self._qnode = qml.QNode(
            self._circuit, dev, interface="torch", diff_method="backprop"
        )

    def _circuit(self, x):
        # x shape: [batch, n_qubits] — AngleEmbedding handles batched input
        qml.templates.AngleEmbedding(x, wires=range(self.n_qubits), rotation="X")
        for layer_idx in range(self.n_layers):
            p = self.params[layer_idx]
            idx = 0
            for q in range(self.n_qubits):
                qml.RZ(p[idx], wires=q); idx += 1
                qml.RX(p[idx], wires=q); idx += 1
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
            for q in range(self.n_qubits):
                qml.RX(p[idx], wires=q); idx += 1
                qml.RZ(p[idx], wires=q); idx += 1
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_qubits]
        # _qnode returns list of n_qubits tensors, each shape [batch]
        out = self._qnode(x)                          # list of n_qubits tensors
        return torch.stack(out, dim=1).float()        # [batch, n_qubits]


# ── Hybrid Quantum-Classical Network ──────────────────────────────────────────
class HybridQNN(nn.Module):
    """
    Architecture (mirrors DVPDESolver from Afrah et al., with improvements):

        Input (z, t)   [batch, 2]
            ↓ normalize_input → [-1,1]
            ↓ preprocessor: Linear(2→hidden) + Tanh + Linear(hidden→hidden) + Tanh + Linear(hidden→n_qubits)
        [batch, n_qubits]
            ↓ QuantumLayer (VQC)
        [batch, n_qubits]  — PauliZ expectation values ∈ [-1,1]
            ↓ output_scale  — learnable per-qubit scale (breaks [-1,1] saturation)
            ↓ postprocessor: Linear(n_qubits→hidden) + Tanh + Linear(hidden→hidden) + Tanh + Linear(hidden→1)
        T̂  [batch, 1]
    """
    def __init__(self, n_qubits: int = 4, n_qlayers: int = 2, hidden: int = 32,
                 quantum_init_std: float = None):
        super().__init__()
        self.preprocessor = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_qubits),
        )
        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_qlayers,
                                     init_std=quantum_init_std)
        # Learnable scale: breaks PauliZ ∈ [-1,1] saturation
        self.output_scale = nn.Parameter(torch.ones(n_qubits))
        self.postprocessor = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in list(self.preprocessor) + list(self.postprocessor):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = normalize_input(x)            # map to [-1,1] before angle embedding
        h = self.preprocessor(x_norm)          # [batch, n_qubits]
        h = self.quantum(h)                    # [batch, n_qubits], values in [-1,1]
        h = h * self.output_scale              # learnable rescale
        return self.postprocessor(h)           # [batch, 1]


# ── Classical baseline (same as pinn_seawater.py) ─────────────────────────────
class ClassicalDNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(5):
            layers += [nn.Linear(in_dim, 32), nn.Tanh()]
            in_dim = 32
        layers.append(nn.Linear(32, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── PINN (network-agnostic) ───────────────────────────────────────────────────
class PINN:
    """
    Identical physics logic to pinn_seawater.py.
    Accepts any nn.Module as `network` — classical DNN or HybridQNN.
    """
    def __init__(self, bc_type="dirichlet", inverse=False,
                 network=None, w_pde=1.0, w_bc=1.0, w_ic=1.0, w_data=1.0):
        self.bc_type  = bc_type
        self.inverse  = inverse
        self.w_pde, self.w_bc, self.w_ic, self.w_data = w_pde, w_bc, w_ic, w_data

        self.net = network if network is not None else ClassicalDNN()
        self.net = self.net.to(DEVICE)

        if inverse:
            self.log_kz = nn.Parameter(torch.tensor([np.log(0.1)]))
        else:
            self.log_kz = None

        self._build_training_data()

    @property
    def kz(self):
        if self.inverse:
            return torch.exp(self.log_kz)
        return torch.tensor(KZ_TRUE, dtype=torch.float32)

    def _T_exact(self, z, t):
        if self.bc_type == "dirichlet": return T_dirichlet(z, t)
        elif self.bc_type == "neumann":  return T_neumann(z, t)
        else:                            return T_robin(z, t)

    def _tensor(self, z, t, requires_grad=False):
        zt = np.stack([z, t], axis=1).astype(np.float32)
        return torch.tensor(zt, requires_grad=requires_grad)

    def _build_training_data(self):
        z_f = np.random.uniform(Z_MIN, Z_MAX, 2540)
        t_f = np.random.uniform(T_MIN, T_MAX, 2540)
        self.zt_f = self._tensor(z_f, t_f, requires_grad=True)

        t_bc0 = np.random.uniform(T_MIN, T_MAX, 40)
        t_bcL = np.random.uniform(T_MIN, T_MAX, 40)
        z_bc  = np.concatenate([np.zeros(40), np.full(40, Z_MAX)])
        t_bc  = np.concatenate([t_bc0, t_bcL])
        T_bc  = self._T_exact(z_bc, t_bc).astype(np.float32)
        self.zt_bc = self._tensor(z_bc, t_bc,
                                   requires_grad=(self.bc_type != "dirichlet"))
        self.T_bc  = torch.tensor(T_bc).unsqueeze(1)

        z_ic = np.random.uniform(Z_MIN, Z_MAX, 160)
        t_ic = np.zeros(160)
        T_ic = self._T_exact(z_ic, t_ic).astype(np.float32)
        self.zt_ic = self._tensor(z_ic, t_ic)
        self.T_ic  = torch.tensor(T_ic).unsqueeze(1)

        if self.inverse:
            z_d = np.linspace(Z_MIN, Z_MAX, 4)
            t_d = np.linspace(T_MIN, T_MAX, 3)
            zz, tt = np.meshgrid(z_d, t_d)
            T_data = self._T_exact(zz.ravel(), tt.ravel()).astype(np.float32)
            self.zt_data = self._tensor(zz.ravel(), tt.ravel())
            self.T_data  = torch.tensor(T_data).unsqueeze(1)

    def _pde_residual(self):
        zt = self.zt_f
        zt.requires_grad_(True)
        T_pred = self.net(zt)
        dT     = torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred),
                                     create_graph=True)[0]
        dT_dz, dT_dt = dT[:, 0:1], dT[:, 1:2]
        d2T_dz2 = torch.autograd.grad(dT_dz, zt, torch.ones_like(dT_dz),
                                      create_graph=True)[0][:, 0:1]
        return dT_dt - self.kz * d2T_dz2

    def _bc_residual(self):
        if self.bc_type == "dirichlet":
            return self.net(self.zt_bc) - self.T_bc
        zt = self.zt_bc.clone().requires_grad_(True)
        T_pred = self.net(zt)
        dT_dz  = torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred),
                                     create_graph=True)[0][:, 0:1]
        if self.bc_type == "neumann":
            return dT_dz
        return dT_dz + T_pred   # robin: dT/dz + T = 0

    def loss(self):
        L = (self.w_pde * torch.mean(self._pde_residual() ** 2)
           + self.w_bc  * torch.mean(self._bc_residual()  ** 2)
           + self.w_ic  * torch.mean((self.net(self.zt_ic) - self.T_ic) ** 2))
        if self.inverse:
            L += self.w_data * torch.mean((self.net(self.zt_data) - self.T_data) ** 2)
        return L

    def train(self, adam_steps=5000, lbfgs_iter=10000, eta_min=1e-5, optimizer="auto"):
        """optimizer="auto": quantum nets run Adam with cosine decay for adam_steps + lbfgs_iter steps,
        classical nets run Adam then L-BFGS. optimizer="adam_cosine" forces the Adam-only schedule on
        any net (the optimiser control for classical-vs-hybrid comparisons)."""
        if optimizer not in ("auto", "adam_cosine"):
            raise ValueError(f"unknown optimizer {optimizer!r}")
        is_quantum = isinstance(self.net, HybridQNN)
        adam_only = is_quantum or optimizer == "adam_cosine"

        if is_quantum:
            # Separate LR: quantum params need higher LR; classical pre/post lower
            quantum_params  = list(self.net.quantum.parameters()) + [self.net.output_scale]
            classical_params = (list(self.net.preprocessor.parameters())
                               + list(self.net.postprocessor.parameters()))
            param_groups = [
                {"params": quantum_params,  "lr": 5e-3},
                {"params": classical_params, "lr": 1e-3},
            ]
            if self.inverse:
                param_groups.append({"params": [self.log_kz], "lr": 1e-3})
            flat_params = quantum_params + classical_params
            if self.inverse:
                flat_params = flat_params + [self.log_kz]
        else:
            all_params = list(self.net.parameters())
            if self.inverse:
                all_params.append(self.log_kz)
            param_groups = [{"params": all_params, "lr": 1e-3}]
            flat_params = all_params

        def grad_norm():
            grads = [p.grad.detach().norm() for p in flat_params if p.grad is not None]
            return float(torch.norm(torch.stack(grads))) if grads else 0.0

        # Circuit-only gradient norm (excludes output_scale and the classical
        # pre/post-processor) — the quantity a barren plateau actually collapses,
        # tracked separately from grad_norm() so circuit-depth/qubit-count
        # ablations can see it directly instead of it being diluted by the
        # classical params' gradients.
        circuit_params = list(self.net.quantum.parameters()) if is_quantum else []

        def circuit_grad_norm():
            grads = [p.grad.detach().norm() for p in circuit_params if p.grad is not None]
            return float(torch.norm(torch.stack(grads))) if grads else 0.0

        self.loss_history = []
        self.grad_norm_history = []
        self.circuit_grad_norm_history = []
        # Index into the histories where L-BFGS begins; None if this run never
        # switches to L-BFGS (the Adam-only path stays on Adam throughout).
        self.lbfgs_start = None if adam_only else adam_steps
        t0 = time.time()

        # For quantum: skip L-BFGS (too expensive per QNode eval), run more Adam steps
        # with cosine LR decay; for classical: keep original Adam + L-BFGS unless forced Adam-only
        if adam_only:
            total_steps = adam_steps + lbfgs_iter  # equivalent total budget
            adam = torch.optim.Adam(param_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                adam, T_max=total_steps, eta_min=eta_min
            )
            for _ in range(total_steps):
                adam.zero_grad()
                L = self.loss()
                L.backward()
                self.loss_history.append(L.item())
                self.grad_norm_history.append(grad_norm())
                self.circuit_grad_norm_history.append(circuit_grad_norm())
                adam.step()
                scheduler.step()
        else:
            adam = torch.optim.Adam(param_groups)
            for _ in range(adam_steps):
                adam.zero_grad()
                L = self.loss()
                L.backward()
                self.loss_history.append(L.item())
                self.grad_norm_history.append(grad_norm())
                self.circuit_grad_norm_history.append(circuit_grad_norm())
                adam.step()

            all_params_flat = list(self.net.parameters())
            if self.inverse:
                all_params_flat.append(self.log_kz)
            lbfgs = torch.optim.LBFGS(
                all_params_flat, lr=1.0, max_iter=lbfgs_iter,
                tolerance_grad=1e-9, tolerance_change=1e-11,
                history_size=100, line_search_fn="strong_wolfe",
            )
            def closure():
                lbfgs.zero_grad()
                L = self.loss()
                L.backward()
                self.loss_history.append(L.item())
                self.grad_norm_history.append(grad_norm())
                self.circuit_grad_norm_history.append(circuit_grad_norm())
                return L
            lbfgs.step(closure)

        return time.time() - t0

    def evaluate(self, n_test=200):
        z_t = np.linspace(Z_MIN, Z_MAX, n_test)
        t_t = np.linspace(T_MIN, T_MAX, n_test)
        zz, tt = np.meshgrid(z_t, t_t)
        zt_test = torch.tensor(
            np.stack([zz.ravel(), tt.ravel()], axis=1), dtype=torch.float32)
        with torch.no_grad():
            T_pred = self.net(zt_test).numpy().reshape(n_test, n_test)
        T_true = self._T_exact(zz, tt)
        mae = np.mean(np.abs(T_pred - T_true))
        l2  = np.linalg.norm(T_pred - T_true) / (np.linalg.norm(T_true) + 1e-12)
        kz_err = None
        if self.inverse:
            kz_err = abs(float(torch.exp(self.log_kz).detach()) - KZ_TRUE)
        return dict(mae=mae, l2=l2, T_pred=T_pred, T_true=T_true,
                    z=z_t, t=t_t, kz_err=kz_err)

    def mean_pde_residual(self):
        with torch.enable_grad():
            zt = self.zt_f.detach().clone().requires_grad_(True)
            T  = self.net(zt)
            dT = torch.autograd.grad(T, zt, torch.ones_like(T),
                                     create_graph=True)[0]
            dT_dz = dT[:, 0:1]
            dT_dt = dT[:, 1:2]
            d2T_dz2 = torch.autograd.grad(dT_dz, zt, torch.ones_like(dT_dz),
                                           create_graph=False)[0][:, 0:1]
            r = (dT_dt - self.kz * d2T_dz2).detach().numpy()
        return float(np.mean(np.abs(r)))
