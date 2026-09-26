# Coordination Matters: A Real-Data QUBO Benchmark for Multi-Satellite Collision Avoidance

Code, data and benchmark instances for the paper

> **Coordination Matters: A Real-Data QUBO Benchmark for Multi-Satellite Collision-Avoidance Maneuver Selection with Constraint-Preserving Classical and Quantum Solvers**
> Arunabha *[Surname]*, Angshuman Jana — Indian Institute of Information Technology Guwahati

When several maneuverable satellites share conjunctions, their avoidance decisions are coupled: if both dodge the same way the danger remains, and a satellite that dodges one neighbour can drift into another. This repository builds a validated pipeline from the public satellite catalogue to **quadratic unconstrained binary optimization (QUBO)** problems for *joint* maneuver selection, solves every instance exactly, and benchmarks operator-style greedy planning, classical annealers and QAOA against the exact optima.

---

## Key results (snapshot of 25 Sep 2026, 16:38 UTC)

| Finding | Result |
|---|---|
| Catalogue → decision problems | 19,283 objects → 3,229-satellite shell → 195 risky encounters (P_c ≥ 1e-4) → **158 independent QUBOs** of ≤ 7 satellites |
| Coordination is common | The optimal plan needs ≥ 2 coordinated burns in **60–67 %** of coupled instances |
| Greedy planning fails there | Uncoordinated planning is sub-optimal in **94–95 %**, first-come-first-served in **85 %** of those instances (Fisher p ≤ 8e-5) |
| Classical solvers | Constraint-preserving **SA-swap** solves 100 % of instances; median TTS₉₉ **0.56–0.62 ms**, 11–54× faster than bit-flip SA |
| QAOA | **XY-mixer** QAOA beats penalty (X-mixer) QAOA on **89/93** (M3) and **102/105** (M5) instances at depth p = 3 |
| Why XY wins | An exact decomposition shows the advantage comes from searching only feasible plans, not from circuit amplification; at p ≤ 3 neither mixer's advantage over random guessing grows with size |
| Verification | Linear TCA vs Newton refinement: 0.2 m median; P_c formula vs exact integral: ≤ 3.4e-4 relative; CW vs J2 integrator: ≤ 0.18 %; QUBO vs direct objective: ≤ 1.3e-13; simulator vs CUDA-Q: \|ΔP\| ≤ 7.4e-6 |

No quantum advantage is claimed: at this scale a classical swap-move annealer solves every instance in under a millisecond. The contribution is the real-data benchmark, the coordination result, and the evidence that **encoding the one-choice-per-satellite constraint in the move set (swap moves / XY mixer) matters more than the solver family**.

---

## Repository layout

> Adjust folder names below if yours differ.

```
.
├── code/
│   ├── cell01_setup_and_data.py          # download CelesTrak GP data (OMM/CSV), build SGP4 propagators
│   ├── cell02_shell_and_screening.py     # find the shell, propagate, GPU conjunction screening, graph
│   ├── cell03_pc_and_episodes.py         # TCA refinement, collision probability, temporal episodes
│   ├── cell04_options_and_qubo.py        # maneuver options (CW model), risk tables, QUBO assembly
│   ├── cell05_calibration_benchmark.py   # lambda sweep, hard-cap QUBO, final benchmark set
│   ├── cell06_classical_solvers.py       # greedy baselines, SA-flip, SA-swap, PT-flip
│   ├── cell07_qaoa_mixers.py             # QAOA with X vs XY mixers (+ CUDA-Q cross-check)
│   └── cell08_statistics_and_paper.py    # statistics, tables, paper figures
├── data/
│   ├── raw/                              # CelesTrak group CSVs used for the snapshot
│   └── processed/
│       ├── omm_snapshot_20260925T1638Z.csv
│       ├── primaries.csv, secondaries.csv, meta.json
│       ├── encounters.csv, components.json
│       ├── encounters_pc.csv, episodes.csv, components_v2.json
│       ├── qubo_instances.pkl, benchmark.pkl
│       ├── solver_results.csv, qaoa_results.csv
│       └── qubo_export/                  # 224 instances: M3_cXXX.qubo, M5_cXXX.qubo
├── figures/                              # fig01 ... fig07 (pipeline figures)
├── paper/                                # main.tex, refs.bib, tables (.csv/.tex), key_numbers.json
└── runs/d20260925_S1/                    # archived results of the reported run
```

---

## Reproducing the pipeline

The code is written as eight notebook cells that run in order in **one** session. It was developed on **Kaggle** with an **NVIDIA Tesla T4** GPU.

1. Create a Kaggle notebook. Set **Accelerator → GPU T4** and **Internet → On**.
2. Paste each file from `code/` into its own cell, in order (`cell01` → `cell08`), and run them one after another.
3. Outputs are written to `/kaggle/working/data/processed/`, `/kaggle/working/figures/` and `/kaggle/working/paper/`.

| Cell | Purpose | Runtime (T4) |
|---|---|---|
| 1 | Download GP data, build propagators, select shell | < 1 min |
| 2 | Shell discovery, propagation (5,690 objects × 2,881 steps), GPU screening | ~1 min |
| 3 | TCA refinement, P_c, sensitivity, temporal episodes | ~4 s |
| 4 | Maneuver options, CW validation, QUBO assembly, exact optima | ~3 s |
| 5 | Objective calibration, hard-cap QUBO, benchmark set | ~3 s |
| 6 | Greedy baselines and classical heuristics | ~1 min |
| 7 | QAOA, X vs XY mixers, p = 1…3, CUDA-Q cross-check | ~7 min |
| 8 | Statistics, LaTeX tables, paper figures | < 1 min |

All tunable parameters sit in the `CFG`, `CFG2`, …, `CFG8` dictionaries at the top of each cell (thresholds, uncertainty model, maneuver menu, solver budgets, QAOA depths).

### Reproducing the exact numbers in the paper (frozen snapshot)

Running Cell 1 with Internet on downloads **today's** catalogue, so results will differ from the paper. To reproduce the reported run:

1. Upload `data/raw/*.csv` as a Kaggle dataset, attach it to the notebook, and set **Internet → Off**. Cell 1 then falls back to the files under `/kaggle/input/`.
2. In Cell 1, replace the line that sets the current time
   ```python
   now_utc = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
   ```
   with the snapshot time
   ```python
   now_utc = dt.datetime(2026, 9, 25, 16, 38)
   ```
   Otherwise every element set is older than the 3-day freshness limit and is discarded.
3. Run Cells 1–8 as above. Stochastic solvers use fixed seeds in their `CFG` dictionaries.

### Robustness runs

Cell 8 archives each run under `RUN_TAG` in `runs/` and analyses all archived runs together. Examples:

- stricter threshold: `CFG3["pc_threshold"] = 1e-5`, re-run Cells 3–8;
- half the assumed uncertainty: halve `CFG3["sigma0_km"]` and `CFG3["sigma_rate_km_per_day"]`, re-run Cells 3–8;
- another day: re-run Cells 1–8 with a new tag.

For larger scenarios, drop instances whose exhaustive enumeration would be too large before Cell 5:

```python
INSTANCES = {m: [I for I in inst if np.prod(I["M_e"]) <= 200_000] for m, inst in INSTANCES.items()}
```

---

## The benchmark instances

### Plain-text QUBO files (`data/processed/qubo_export/*.qubo`)

One file per coupled instance: 112 for menu **M3** (options {0, +3, −3}) and 112 for **M5** (options {0, +1, −1, +3, −3}; the number is the burn lead time in orbits).

```
n  nnz  const  E_opt          # header: qubits, non-zeros, constant offset, exact optimum energy
i  j  Q_ij                    # nnz lines, 0-based indices, upper triangle (i <= j)
```

The energy of a bit-string `x` is `E(x) = xᵀ Q x + const`, and the exact minimum equals `E_opt`.

```python
import numpy as np

def read_qubo(path):
    with open(path) as f:
        n, nnz, const, e_opt = f.readline().split()
        n, nnz, const, e_opt = int(n), int(nnz), float(const), float(e_opt)
        Q = np.zeros((n, n))
        for _ in range(nnz):
            i, j, v = f.readline().split()
            Q[int(i), int(j)] = float(v)
    return Q, const, e_opt

Q, const, e_opt = read_qubo("data/processed/qubo_export/M3_c008.qubo")
energy = lambda x: float(x @ Q @ x + const)
```

Variables come in consecutive **blocks**, one block per satellite decision; a valid plan has exactly one `1` per block. The one-hot constraint is already in `Q` as a penalty, so the minimum is always a valid plan. The block sizes are stored in `benchmark.pkl` (next section) and are needed for constraint-preserving solvers such as swap moves or XY-mixer QAOA.

### Full instance data (`data/processed/benchmark.pkl`)

```python
import pickle
# Only unpickle files you trust.
with open("data/processed/benchmark.pkl", "rb") as f:
    B = pickle.load(f)

inst = B["bench"]["M3"]          # list of instances ("M5" for the 5-option menu)
I = inst[0]
I["Q"], I["const"]               # QUBO matrix (upper triangular) and constant
I["M_e"]                         # options per satellite = block sizes
I["labels"]                      # (episode id, option label) for every variable
I["opt_energy"], I["opt_choice"] # exact optimum and the chosen option per satellite
I["n_maneuvers"]                 # burns in the optimum (>= 2 means coordination)
I["worst_energy"]                # worst feasible energy (for approximation ratios)
I["fuel"], I["lin"], I["Pq"]     # fuel costs, linear risks, pairwise risk matrix
```

The objective is `H = fuel + λ·risk + B·(couplings still above threshold) + A·(one-hot penalty)` with λ = 1000 (m/s per unit P_c). See Sec. IV of the paper.

---

## Requirements

| Package | Used for |
|---|---|
| `sgp4 >= 2.23` | SGP4 propagation from OMM data (installed automatically by Cell 1) |
| `numpy`, `pandas`, `scipy`, `matplotlib`, `networkx`, `requests` | Everything else (pre-installed on Kaggle) |
| `torch` (CUDA) | GPU screening and QAOA simulation (optional; NumPy fallback) |
| `cudaq` 0.16 | Gate-level cross-check of the QAOA circuits (optional; installed by Cell 7) |

---

## Modelling assumptions and limitations

- Public GP data carry **no covariance**. The position-uncertainty model is assumed (see `CFG3`), and trigger counts are sensitive to it.
- Satellites are assumed to **return to their slot** one orbit after closest approach.
- Only **along-track impulsive** burns of 5 cm/s are considered; these are weak against near-head-on encounters.
- Results come from **one snapshot**; heuristic budgets were fixed rather than tuned; QAOA was simulated without noise.
- This is a **benchmark model**, not a replica of any operator's conjunction-assessment process.

---

## Data source and acknowledgment

Orbital data are general-perturbations (GP) element sets from **[CelesTrak](https://celestrak.org)**, downloaded in the OMM CSV format. Please follow the [CelesTrak usage policy](https://celestrak.org/usage-policy.php) if you re-download data, and avoid repeated downloads (Cell 1 caches files for 2 hours).

## Citation

If you use the code or the benchmark instances, please cite:

```bibtex
@misc{arunabha2026coordination,
  author = {Arunabha [Surname] and Jana, Angshuman},
  title  = {Coordination Matters: A Real-Data {QUBO} Benchmark for Multi-Satellite
            Collision-Avoidance Maneuver Selection with Constraint-Preserving
            Classical and Quantum Solvers},
  year   = {2026},
  note   = {Preprint. Code and data: https://github.com/Arunav1/[repository-name]}
}
```

## License

Code: *[choose a license, e.g. MIT]*. Derived data files are provided for research reproducibility; the underlying orbital data remain subject to CelesTrak's terms.

## Contact

Arunabha Dutta — Department of CSE, IIIT Guwahati — arunavdutta20@gmail.com
