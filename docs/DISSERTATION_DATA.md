# Dissertation Data Package — Project Lacuna

This folder contains all data, tables, figures, and context needed to write the dissertation chapter(s) for Project Lacuna.

## Project Summary

**Lacuna** is a transformer-based classifier for missing data mechanisms. Given a dataset with missing values, it classifies the generating mechanism as:
- **MCAR** (Missing Completely at Random) → safe for complete-case analysis
- **MAR** (Missing at Random) → requires multiple imputation
- **MNAR** (Missing Not at Random) → requires sensitivity analysis

### Architecture
- Row-level tokenization: each cell → [value, observed_indicator, mask_indicator]
- Transformer encoder (4 layers, 128-dim, 4 heads) with attention-based pooling
- Reconstruction heads for self-supervised pretraining (1 per expert)
- Mixture of Experts gating network (3 experts in 1/1/1 symmetric config)
- Bayes-optimal decision rule with asymmetric loss matrix
- Post-hoc temperature scaling (T=1.96)

### Training Setup
- **Semi-synthetic data:** Real datasets from UCI/OpenML with synthetic missingness injected by 110 parameterized generators (32 MCAR, 36 MAR, 42 MNAR at runtime after dimension filtering)
- **24 training datasets**, 7 validation datasets
- Cross-entropy loss, uniform generator prior, no label smoothing
- 100 epochs, batch size 16, lr=0.0003, hidden=128

### Final Model Performance (Experiment 10)
| Metric | Value |
|--------|-------|
| Overall accuracy | 82.6% |
| MCAR recall / precision / F1 | 94.5% / 79.4% / 86.3% |
| MAR recall / precision / F1 | 73.6% / 91.8% / 81.7% |
| MNAR recall / precision / F1 | 84.5% / 78.7% / 81.5% |
| ECE | 0.0383 |
| Parameters | 901,130 |
| Temperature | 1.96 (post-hoc calibrated) |

---

## Key Findings (for the Narrative)

### 1. Expert Asymmetry Causes MAR Underdetection
The original architecture used 5 experts mapped as [MCAR, MAR, MNAR, MNAR, MNAR]. Despite mean aggregation correcting the class prior, the 3 MNAR logits created gradient asymmetry during training. The gating network learned to route borderline MAR→MNAR because it reduced loss across 3 expert outputs rather than 1.

**Evidence:** Switching to 1/1/1 symmetric experts improved MAR recall from 52.6% to 69.3% (+16.7 points) while improving overall accuracy from 77.0% to 78.4% with fewer parameters.

### 2. Temperature Scaling Improves Both Calibration and Accuracy
Post-hoc temperature scaling with T=1.96 was expected to only improve calibration (ECE). Instead, it improved accuracy on all three classes simultaneously (+4.2 points overall). The softened softmax pulled overconfident wrong predictions back across the decision boundary.

**Evidence:** ECE improved from 0.1157 to 0.0383, and all three class recalls improved: MCAR +5.0, MAR +4.3, MNAR +5.1.

### 3. Bimodal Generator Identifiability
Per-generator analysis (Experiment 8) revealed that MAR generators split into two groups: easily detected (ColBlocks 100%, CrossClass 100%) vs near-zero (Weak 0%, Interactive 0%, MixedPred 16%). Some MAR mechanisms produce missingness patterns genuinely indistinguishable from MNAR given only the observed pattern.

### 4. Class-Balanced Prior Hurts
Balancing the class prior (1/3 each) actually worsened performance because it forced the model to see more of the hard/ambiguous generators. The uniform prior naturally downweights confusing generators.

### 5. Brier Score Loss is Too Weak
Brier score's quadratic penalty provides weaker gradients than cross-entropy's log penalty, especially near p→0. Combined with label smoothing, the loss landscape became too flat for the model to find sharp class boundaries.

---

## Data Files

### `docs/data/all_experiments.json`
Consolidated JSON with all experiment results. Generate on Forge with:
```bash
python scripts/consolidate_results.py --runs-dir /mnt/artifacts/project_lacuna/runs
```

Schema:
```json
{
  "experiments": [
    {
      "name": "Experiment 5 — ...",
      "report": {
        "summary": {"accuracy": 0.77, "mcar_acc": 0.934, ...},
        "confusion_matrix": {"matrix": [[...]], "precision": [...], ...},
        "calibration": {"ece": 0.1338, "bins": [...]},
        "confidence_analysis": {...},
        "per_generator_accuracy": {...},
        "probability_distributions": {...},
        "entropy": {...},
        "selective_accuracy": {...}
      }
    }
  ]
}
```

### `docs/figures/`
Generated figures (PDF + PNG) and LaTeX tables. Generate with:
```bash
python scripts/generate_dissertation_figures.py
```

Produces:
- `metric_progression.pdf` — Bar chart of accuracy/recall across experiments
- `mar_journey.pdf` — Line plot of MAR recall/precision/F1 progression
- `calibration_comparison.pdf` — ECE comparison across experiments
- `confusion_matrix_exp{5,9,10}.pdf` — Heatmap confusion matrices
- `tables/experiment_summary.csv` — Full metrics CSV
- `tables/main_results.tex` — LaTeX table for main results
- `tables/confusion_matrix_exp{5,9,10}.tex` — LaTeX confusion matrices

---

## Experiment Progression

| Exp | Name | Change | Accuracy | MAR Recall | ECE | Status |
|-----|------|--------|----------|------------|-----|--------|
| 5 | Baseline | First full training run | 77.0% | 52.6% | 0.134 | ✅ Baseline |
| 7 | Brier + Balanced | 3 changes at once | 54.8% | 30.7% | 0.287 | ❌ Failed |
| 8 | Balanced Prior Only | Isolate prior effect | 62.6% | 34.8% | 0.233 | ❌ Regressed |
| 9 | **1/1/1 Symmetric** | **Fix architecture** | **78.4%** | **69.3%** | **0.116** | ✅ Key fix |
| 10 | **+Temperature** | **Post-hoc T=1.96** | **82.6%** | **73.6%** | **0.038** | ✅ Final |

Experiments 1-4 were infrastructure (architecture, tests, generators, data pipeline). Experiment 6 was evaluation tooling. These don't have quantitative results.

---

## Source Files (in Project Codebase)

### Key Model Code
- `lacuna/models/assembly.py` — Model assembly and config (LacunaModelConfig, create_lacuna_model)
- `lacuna/models/moe.py` — Mixture of Experts (MoEConfig, GatingNetwork, MixtureOfExperts)
- `lacuna/models/encoder/` — Transformer encoder and tokenization
- `lacuna/models/reconstruction/` — Reconstruction heads for self-supervised pretraining
- `lacuna/training/calibration.py` — Temperature scaling (find_optimal_temperature, apply_temperature_scaling)
- `lacuna/training/report.py` — Evaluation metrics (generate_eval_report)
- `lacuna/training/trainer.py` — Training loop
- `lacuna/generators/` — 110 missingness generators (MCAR/MAR/MNAR)

### Key Scripts
- `scripts/infer.py` — **User-facing inference**: give it a CSV with NaN, get mechanism classification + recommended action
- `scripts/train_semisynthetic.py` — Training with semi-synthetic data
- `scripts/evaluate.py` — Full evaluation with reporting
- `scripts/calibrate.py` — Post-hoc temperature scaling
- `scripts/consolidate_results.py` — Gather all results into one JSON
- `scripts/generate_dissertation_figures.py` — Generate figures and LaTeX tables

### Sample Output
- `docs/sample_output.txt` — Three example inference outputs (MAR, MCAR, MNAR) showing what a user would see

### Experiment Log
- `experiments/JOURNAL.md` — Complete experiment journal with methodology, results, and interpretation

### Configs
- `configs/training/semisynthetic_full.yaml` — Production training config
- `configs/generators/lacuna_tabular_110.yaml` — Full generator registry (110 generators)

---

## End-User Interface

A researcher with tabular data containing missing values uses Lacuna via:

```bash
python scripts/infer.py --input my_data.csv --checkpoint calibrated.pt
```

**Input:** Any CSV with numeric columns and NaN values representing missingness.

**Output:** A report containing:
1. **Classification probabilities:** P(MCAR), P(MAR), P(MNAR)
2. **Confidence level:** Based on the maximum probability
3. **Recommended action:** One of three actions from Bayes-optimal decision rule:
   - **GREEN (MCAR):** Complete-case analysis or simple imputation
   - **YELLOW (MAR):** Multiple imputation (MICE, Amelia) or likelihood methods (EM, FIML)
   - **RED (MNAR):** Sensitivity analysis, selection models (Heckman), pattern-mixture models
4. **Bayes risk analysis:** Expected risk for each action given the posterior probabilities

See `docs/sample_output.txt` for three example outputs showing MAR, MCAR, and MNAR classifications.

The Bayes decision rule uses an asymmetric loss matrix that encodes the practical cost of each misclassification. For example, treating MNAR data as MCAR (risk = 1.0) is worse than treating MCAR data as MAR (risk = 0.2), because the former produces biased results while the latter is merely conservative.

---

## Generating Dissertation Outputs

### On Forge (GPU server):
```bash
# 1. Consolidate all experiment results
python scripts/consolidate_results.py --runs-dir /mnt/artifacts/project_lacuna/runs

# 2. Generate figures and tables
python scripts/generate_dissertation_figures.py
```

### On local machine (no GPU needed):
```bash
# If all_experiments.json is already generated, just run the figure script
python scripts/generate_dissertation_figures.py
```

### In dissertation LaTeX:
```latex
\input{figures/tables/main_results.tex}
\input{figures/tables/confusion_matrix_exp10.tex}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/mar_journey.pdf}
\caption{MAR detection performance across ablation studies...}
\end{figure}
```

---

## Hardware and Reproducibility

- **Training hardware:** NVIDIA RTX 5070 Ti, CUDA 12.x
- **Training time:** ~28 minutes per experiment (100 epochs)
- **Calibration time:** ~3 seconds (grid search over temperature)
- **Evaluation time:** ~2.5 seconds (800 samples, 50 batches)
- **Random seed:** Default from config (reproducible within same hardware)
- **Software:** Python 3.x, PyTorch, custom `lacuna` package
