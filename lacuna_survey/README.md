# Lacuna-Survey

The collection-process specialisation of Lacuna for self-administered
or interviewer-administered surveys. See ADR 0005
(`docs/decisions/0005-lacuna-survey-iteration-arc.md`) for the v1–v8
iteration arc and empirical motivation.

## Where things live

| Asset | Location | Notes |
|---|---|---|
| Generator registry | `configs/generators/lacuna_survey.yaml` | Lives under `configs/` because base Lacuna's loader expects it there |
| Training config | `configs/training/survey.yaml` | Same reason |
| Training catalog (X-bases) | `/mnt/data/lacuna/raw/survey_*.csv` | Auto-discovered by `lacuna.data.catalog.create_default_catalog()`; complete-case only |
| Real-data fetch script | `lacuna_survey/fetch_data.py` | Run with `python -m lacuna_survey.fetch_data` |
| Real evaluation data | `lacuna_survey/evaluation_data/survey_*_real.csv` | Native missingness preserved (this is the test signal); gitignored |
| Real-data diagnostic | `lacuna_survey/diagnostic.py` | Run with `python -m lacuna_survey.diagnostic` |
| Deployment-layer artefacts | `lacuna_survey/deployment/` | Future home for calibration scalars, OOD detector weights, ensemble configs, etc. |
| Trained checkpoint | `/mnt/artifacts/project_lacuna/runs/lacuna_survey_v*/checkpoints/best_model.pt` | Standard Lacuna checkpoint location |

## What's variant-specific vs base Lacuna

Lacuna-Survey diverges from base Lacuna in exactly one architectural
choice: `model.learn_evidence_attenuation` is `True` for survey
training (set in `configs/training/survey.yaml`). Base Lacuna's
default is `False`. Everything else — value-conditional features,
generator family classes, training pipeline, model architecture — is
identical. Branching the codebase is unnecessary; future variants
(Lacuna-Records, Lacuna-Instrument, …) follow the same pattern with
their own registry / catalog / fetch / diagnostic / deployment assets.

## Typical workflow

```bash
# Once: fetch the real survey datasets (training + evaluation)
python -m lacuna_survey.fetch_data

# Train a new variant version
python scripts/train.py \
    --config configs/training/survey.yaml \
    --name lacuna_survey_v9 \
    --seed 42 \
    --class-balanced-prior \
    --report

# Install the trained checkpoint into the demo bundle
python demo/install_checkpoint.py /mnt/artifacts/project_lacuna/runs/lacuna_survey_v9

# Run the diagnostic against the installed checkpoint
python -m lacuna_survey.diagnostic
```

## Open work (per ADR 0005)

Four candidate next directions to push within-domain MAR detection
from 3/5 toward 5/5, in order of expected leverage:

1. Real-data calibration (temperature scaling on verified bfi /
   chile / cars93 labels) → `deployment/calibration.json`.
2. Encoder-side regularization (information bottleneck on evidence).
3. Out-of-distribution detection (so cross-domain Pima / airquality
   return abstain rather than confident-wrong) →
   `deployment/ood_detector.pt`.
4. Active learning on real data (small verified-label real-survey
   set as training anchors).

The Little's-test ensemble direction is explicitly NOT on the list;
see ADR 0004 and the `mcar-alternatives-bakeoff` arc for the
empirical reasons (six classical MCAR-test families tested as gate
features, none cleared the bar).
