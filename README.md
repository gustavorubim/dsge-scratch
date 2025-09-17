# sge-labs Plotting Utilities

This repository implements a from-scratch DSGE toolkit with a lightweight script for generating impulse response (IRF) and forecast error variance decomposition (FEVD) plots directly from Dynare-style `.mod` files.

## Requirements

- Python 3.11+
- Dependencies listed in `pyproject.toml`
- The repo root must be on `PYTHONPATH` (the bundled script takes care of this automatically).

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Run the unit tests to ensure everything is wired correctly:

```bash
pytest -q
```

## Script: `scripts/generate_plots.py`

This helper reads a Dynare `.mod` file, parses and solves the model with `sgelabs`, and writes IRF/FEVD PNGs plus a diagnostics NPZ bundle to an output folder.

Basic usage:

```bash
python scripts/generate_plots.py <model.mod> <output_dir>
```

Example (RBC toy model):

```bash
python scripts/generate_plots.py examples/rbc_basic/rbc.mod output
```

Example (Smets–Wouters 2007 replication):

```bash
python scripts/generate_plots.py model_base/US_SW07/US_SW07_rep/US_SW07_rep.mod output
```

### Optional arguments

- `--horizon N` (default `40`): number of periods for IRF/FEVD computations.
- `--include-lags`: include companion lag states in plots (default skips them)

Examples:

```bash
# Short horizon, results saved under output_short/rbc/...
python scripts/generate_plots.py examples/rbc_basic/rbc.mod output_short --horizon 12

# Long horizon analysis for SW07 (output_long/US_SW07_rep/...)
python scripts/generate_plots.py model_base/US_SW07/US_SW07_rep/US_SW07_rep.mod output_long --horizon 80
python scripts/generate_plots.py model_base/US_SW07/US_SW07_rep/US_SW07_rep.mod output --horizon 80

### Output

For each run the script writes:

- `<output_dir>/<model_name>/irfs/<variable>/<shock>.png`: IRF for each shock/variable pair (one figure per response)
- `<output_dir>/<model_name>/fevds/<variable>.png`: FEVD bar chart per endogenous variable
- `<output_dir>/<model_name>/diagnostics.npz`: NumPy archive containing transition matrices, IRFs, FEVD, state/shock names, Blanchard-Kahn flags, and the horizon used

File names are sanitized to keep only alphanumeric characters, dashes, and underscores.
By default, lagged companion states are omitted; pass \\--include-lags\\ if you need the stacked state vector plots.

You can inspect the diagnostics bundle, e.g.:

```python
import numpy as np
payload = np.load("output/rbc/diagnostics.npz", allow_pickle=True)
print(payload["state_names"])
print(payload["irfs"].shape)
```

### Batch generation tips

The script is self-contained; to process multiple models you can loop over files:

```bash
for mod in examples/*/*.mod; do
  python scripts/generate_plots.py "$mod" output --horizon 40
done
```

On Windows PowerShell:

```powershell
Get-ChildItem examples -Filter *.mod -Recurse | ForEach-Object {
    python scripts/generate_plots.py $_.FullName output --horizon 60
    python scripts/generate_plots.py $_.FullName $dest --horizon 60
}
```

## CLI (`sgelabs` entrypoint)

The package also exposes a CLI for advanced workflows:

```bash
sgelabs load --mod model.mod --output model.ir.json
sgelabs solve --mod model.mod --output model_solution.npz
sgelabs irf --solution model_solution.npz --horizon 40 --output model.irf.npz
sgelabs fevd --solution model_solution.npz --horizon 40 --output model.fevd.npy
```

Refer to `sgelabs/__main__.py` for the full command set.

## Troubleshooting

- Ensure `matplotlib` can access a backend (the default `Agg` works headlessly).
- If you see `ModuleNotFoundError: sgelabs`, confirm you run the script from the repo root or install the package with `pip install -e .`.
- Large models may require additional horizons or numerical tolerance adjustments; the solver reports Blanchard–Kahn flags in the diagnostics bundle.

## License

See `LICENSE` (if provided) for terms and conditions.
