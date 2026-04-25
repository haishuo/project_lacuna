# Lacuna — class demo

A small Streamlit app that runs Lacuna on a CSV you drag in, then shows a
green / yellow / red verdict for the missingness mechanism plus the
intermediate work (per-column missing rates, cross-column correlation,
posterior, Bayes risk).

## What you need on the demo machine

- Python 3.11+
- The Lacuna repo (this folder lives inside it)
- The `pystatistics` sibling package importable on the same Python (this
  is a dependency of `lacuna`; if you already trained a model on this
  machine you have it)
- A trained checkpoint at `demo/model.pt` (or pointed to via
  `LACUNA_DEMO_CHECKPOINT=/path/to/model.pt`)

## Install

```bash
cd <repo root>
pip install -r demo/requirements.txt
pip install -e .          # makes the lacuna package importable
```

If `pystatistics` isn't already on the path:

```bash
pip install -e ../pystatistics
```

## Run

```bash
./demo/run.sh
# or:
streamlit run demo/app.py
```

The browser opens on <http://localhost:8501>. Drop a CSV (or pick one
from `demo/sample_data/`) and hit **Analyze**.

## Pointing at a different checkpoint

```bash
LACUNA_DEMO_CHECKPOINT=/path/to/another.pt ./demo/run.sh
```

If a sidecar JSON sits next to the checkpoint (e.g. `model.json` next to
`model.pt`) with keys `trained_at`, `val_accuracy`, `config`, the sidebar
will show them.

## Architecture assumption

The app assumes the checkpoint was trained with the
`semisynthetic_full.yaml` defaults (hidden=128, evidence=64, layers=4,
heads=4, max_cols=48, max_rows=128). If you trained with different
hyperparameters, edit `MODEL_DEFAULTS` at the top of `app.py`.
