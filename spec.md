# 📦 `sge-labs`: From-Scratch DSGE Toolkit in Python

Build a Python-native, from-scratch DSGE system that:
- Parses `.mod` files (Dynare-like DSL)
- Linearizes around steady state
- Solves linear RE models via GenSys (Sims, 2001)
- Computes IRFs, FEVDs, and Historical Decompositions
- Uses only core libraries (NumPy, SciPy, Numba, SymPy)

## ✅ Goals

- Full `.mod` parser for core Dynare subset
- Typed intermediate representation (IR)
- First-order linearization (symbolic or finite diff)
- GenSys solver (QZ decomposition)
- IRF/FEVD computation
- Historical decomposition from structural shocks
- Kalman filter/smoother to recover shocks (optional)

## 🚫 Non-Goals

- Higher-order perturbations
- Dynare/PyDSGE/GEconPy/Julia dependency
- Full-blown Bayesian estimation
- GUI or interactive tools

---

## 📁 Directory Structure

```
sge-labs/
  sgelabs/
    io/             # parser, AST, mod file loader
    ir/             # model IR and linearization
    solve/          # GenSys implementation
    inference/      # Kalman filter, smoother, historical decomp
    analysis/       # IRFs, FEVDs
    plotting/       # IRF/FEVD/Historical plots
    utils/          # algebra, naming, validation
  examples/
  tests/
  pyproject.toml
```

---

## ⚙️ CLI Interface

```bash
sgelabs load   --mod model.mod          # parse mod
sgelabs solve  --ir model.ir.json       # linearize + solve
sgelabs irf    --ss model.ss.json       # IRF compute + plot
sgelabs fevd   --ss model.ss.json       # FEVD compute + plot
sgelabs hist   --ss model.ss.json --data y.csv
```

---

## 🧠 `.mod` Subset (Dynare-like)

Supported blocks:
- `var`, `varexo`, `parameters`
- `model` ... `end`
- `initval` (optional)
- `shocks` (for Σ_ε)
- `varobs` (observables)

Example:

```mod
var y c k i a;
varexo e_a;
parameters alpha beta delta rho sigma;

model;
  y = c + i;
  c = (1 - alpha)*y;
  i = y - c;
  k = (1 - delta)*k(-1) + i;
  a = rho*a(-1) + e_a;
end;

initval;
  a = 0; k = 1;
end;

shocks;
  var e_a; stderr sigma;
end;

varobs y c i;
```

---

## 📄 Intermediate Representation

```python
@dataclass
class Variable: name: str
@dataclass
class Shock:    name: str
@dataclass
class Equation: lhs: Expr; rhs: Expr
@dataclass
class ModelIR:
    endo: List[Variable]
    exo: List[Shock]
    params: Dict[str, float]
    equations: List[Equation]
    shocks: Dict[str, float]  # varname -> stderr
    initvals: Dict[str, float]
    varobs: List[str]
```

---

## 🔧 Linearization

- Linearize around steady state
- Use `SymPy` if enabled; else finite differences
- Output Sims form:

\[
\text{GAM0} y_t = \text{GAM1} y_{t-1} + C + \text{PSI} \varepsilon_t + \Pi \eta_t
\]

Return:

```python
@dataclass
class Linearization:
    GAM0, GAM1, C, PSI, Pi: np.ndarray
    y_ss: np.ndarray
    Sigma_e: np.ndarray  # structural shock cov
    names: List[str]
    shock_names: List[str]
```

---

## 🧮 GenSys Solver

- From scratch implementation using `scipy.linalg.ordqz`
- Perform:
  - QZ decomposition
  - Blanchard–Kahn check
  - Construct law of motion:

\[
y_t = G y_{t-1} + C_0 + H \varepsilon_t
\]

---

## 📊 Analysis

### IRF

```python
def compute_irfs(G, H, h=40) -> np.ndarray:
    # Returns (h+1, n, k) array
```

### FEVD

```python
def compute_fevd(G, H, Sigma_e, h=40) -> np.ndarray:
    # Returns (n, k) contribution shares
```

### Historical Decomposition

```python
def historical_decomp(G, H, eps: np.ndarray) -> np.ndarray:
    # eps: (T, k)
    # Returns (k, T, n)
```

---

## 🧾 Kalman Filter / Smoother

From-scratch implementation:
- Input: measurement eq \( z_t = Z y_t + d + v_t \)
- Output: smoothed states + shocks
- Provide log-likelihood if desired

---

## 📈 Plotting

- IRFs: per variable, lines per shock
- FEVD: stacked area / lines
- Historical: area contribution + fitted overlay
- Use Matplotlib; clean deterministic style

---

## 🧪 Testing

- Round-trip parser test
- Jacobian validation (symbolic vs numeric)
- Manufactured systems for GenSys
- Known IRFs vs closed-form
- Historical recon vs sum of shock contribs
- KF recovery on simulated data

---

## 📚 Config: `config.yaml`

```yaml
measurement:
  observables: [y, c, i]
  Z: null
  R_diag: [0.0, 0.0, 0.0]

linearization:
  method: sympy
  fd_step: 1e-6

outputs:
  horizons: 40
```

---

## 🚀 Performance

- Use Numba on all core loops: IRF, FEVD, KF, smoothing
- Avoid Python loops inside algebra
- Use contiguous, float64 arrays only
- Cache matrix powers where needed

---

## ✅ Acceptance Criteria

- Runs on `examples/rbc_basic/rbc.mod`
- Produces IRF + FEVD plots
- Reconstructs fitted path from shocks
- All unit tests pass

---

## 🪜 Future Extensibility

- Higher-order perturbation
- Bayesian estimation (Priors + MCMC)
- Markov-switching models
- Particle filters (non-Gaussian)

---

## ✅ Deliverables

- Complete Python codebase
- Working CLI + Python API
- Example `.mod`, config, and plots
- 100% test coverage for core components

