"""
scripts/run_detectability_study.py

E-JUSTIFY/E-FALSIFY detectability study — execution exactly per the FROZEN spec
(`PROPOSAL-E-JUSTIFY-E-FALSIFY-detectability-study.md`; PI-approved 2026-06-07). Architectural review of
the Stage-2 detectability head. H0 = derived stack (locked coverage gate + posterior info-gain);
H_A = post-hoc representation->oracle probes (ridge primary / fixed-HP MLP secondary; NEVER wired).

Frozen design (no tuning; thresholds locked in the spec; gate locked in runs/detectability_gate.json
BEFORE training, commit b9b0c8d):
  subjects   2 pools x 8 seeds, D3 recipe VERBATIM (binary delta0-vs-2.5, LODSurveyExampleSource tau=.70
             => subjects are top_coding-trained; own_value cells evaluate footprint-specificity)
  cells      (idiom in {top_coding, own_value}) x dataset x continuous target column x delta in
             {0,.75,1.25,2.5} x rate .3; 200 examples/cell, deterministic per-cell RNG
  I_oracle   1 - 2*BE_profiled; top_coding via lod_oracle_cell, own_value via compute_profiled_cell;
             n=384 (the per-example row regime), n_mc=800 (pre-hoc; SE ~.012); delta=0 => 0 exactly
  I_gain     mean over cell of KL(calibrated posterior || uniform) in BITS = 1 - H2(p)
  probes     ridge alpha=1.0 (primary) + MLP 32-unit/200ep/lr1e-3 (secondary), trained on the
             2 in-family anchors' cells (cps1988, psid1976), targets = I_oracle(cell)
  per-cell calibration (F4 tertile, pre-hoc): M.ece over the cell's 200 examples (mechanical)

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_detectability_study.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.ingestion import RawDataset
from lacuna.feasibility.profiled_oracle import compute_profiled_cell
from lacuna.feasibility.xmodel import ConditionalGaussian
from lacuna.survey import metrics as M
from lacuna.survey.batching import make_example, make_lod_example
from lacuna.survey.column_batching import collate_columns
from lacuna.survey.example_source import LODSurveyExampleSource
from lacuna.survey.level1_train import Level1Config, train_level1
from lacuna.survey.lod_oracle import lod_oracle_cell

RB = Path("/mnt/data/lacuna/role_b")
GATE = json.loads(Path("runs/detectability_gate.json").read_text())
OUT = Path("runs/detectability_study.json")
SEEDS = [2026, 7, 99, 13, 41, 57, 88, 101]
DELTAS = [0.0, 0.75, 1.25, 2.5]
IDIOMS = ("top_coding", "own_value")
N_EX, MAX_ROWS, RATE, TAU_Q, BETA1 = 200, 384, 0.3, 0.70, 1.0
ORACLE_N, ORACLE_MC = 384, 800
MIN_CARD = 30

LABOR = ["survey_cps1985", "survey_cps1988", "survey_psid1976", "survey_psid7682", "survey_workinghours"]
NHANES = ["rb_nhanes_weight", "rb_nhanes_poverty", "rb_nhanes_income"]
ANCHORS = ["survey_cps1988", "survey_psid1976"]
MODELS = {  # held-out D -> (pool datasets, eval datasets, gate-pool key)
    "nhanes": (LABOR, NHANES + ["rb_scf2022_wealth_cont"] + ANCHORS, "pool_nhanes"),
    "hmda": (LABOR + NHANES, ["survey_hmda", "rb_scf2022_wealth_cont"] + ANCHORS, "pool_hmda"),
}
cat = create_default_catalog()


def cont_base(name):
    if name.startswith("rb_"):
        df = pd.read_csv(RB / f"{name}.csv")
        mat, feats = df.to_numpy(np.float32), list(df.columns)
    else:
        raw = cat.load(name)
        mat, feats = np.asarray(raw.data, np.float32), list(raw.feature_names)
    idx = [j for j in range(mat.shape[1])
           if np.unique(mat[:, j][np.isfinite(mat[:, j])]).size >= MIN_CARD]
    if len(idx) < 2:
        raise ValueError(f"{name}: <2 continuous columns")
    return RawDataset(data=mat[:, idx], feature_names=tuple(feats[j] for j in idx),
                      source="role_b_projected" if name.startswith("rb_") else "native", name=name)


def _cfg():
    return Level1Config(delta_grid=[0.0, 2.5], beta1_range=(1.0, 1.0), target_rate=RATE,
                        max_rows=MAX_ROWS, batch_size=16, train_size=600, max_epochs=40, patience=6,
                        val_size=140, test_size=140, m=16, e_col=32, coarse_scheme="binary")


def _zscore(x):
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd if sd > 0 else 1.0)


def fit_xmodel(base, t_idx):
    """ConditionalGaussian fit on (max-|corr| predictor, target) z-scored columns (P2.2c recipe)."""
    X = torch.from_numpy(np.asarray(base.data, np.float64))
    zt = _zscore(X[:, t_idx])
    best_p, best_abs = None, -1.0
    for c in range(X.shape[1]):
        if c == t_idx or float(X[:, c].std()) == 0:
            continue
        zc = _zscore(X[:, c])
        corr = float((zt * zc).mean())
        if abs(corr) > best_abs:
            best_abs, best_p = abs(corr), c
    return ConditionalGaussian.fit(_zscore(X[:, best_p]).float(), zt.float())


def all_cells():
    """Sorted unique cells: (dataset, t_idx, t_name, idiom, delta). Deterministic enumeration."""
    names = sorted({d for _, (_, ev, _) in MODELS.items() for d in ev})
    cells = []
    for n in names:
        b = cont_base(n)
        for t in range(len(b.feature_names)):
            for idm in IDIOMS:
                for d in DELTAS:
                    cells.append((n, t, b.feature_names[t], idm, d))
    return sorted(cells)


def gen_examples(base, t_idx, idiom, delta, rng):
    out = []
    for _ in range(N_EX):
        if idiom == "top_coding":
            out.append(make_lod_example(base, beta1=BETA1, delta=delta, target_rate=RATE,
                                        tau_quantile=TAU_Q, rng=rng.spawn(), max_rows=MAX_ROWS,
                                        target_idx=t_idx))
        else:
            out.append(make_example(base, beta1=BETA1, delta=delta, target_rate=RATE,
                                    rng=rng.spawn(), max_rows=MAX_ROWS, target_idx=t_idx))
    return out


def oracle_cell(base, t_idx, idiom, delta, rng):
    if delta == 0.0:
        return 0.0, 0.5
    xm = fit_xmodel(base, t_idx)
    if idiom == "top_coding":
        c = lod_oracle_cell(xm, delta=delta, beta1=BETA1, tau_quantile=TAU_Q, target_rate=RATE,
                            n=ORACLE_N, rng=rng, n_mc=ORACLE_MC, n_nodes=96)
        be = c["profiled_bayes_error"]
    else:
        c = compute_profiled_cell(delta, BETA1, RATE, ORACLE_N, rng, xmodel=xm, n_mc=ORACLE_MC)
        be = c["profiled_bayes_error"]
    return max(0.0, 1.0 - 2.0 * be), be


@torch.no_grad()
def eval_cell(model, examples, cfg):
    """(I_gain bits, per-cell ECE, entropy bits, mean p1, reps [N,32], p1 [N])."""
    model.eval()
    P, reps = [], []
    for s in range(0, len(examples), cfg.batch_size):
        cb = collate_columns(examples[s:s + cfg.batch_size], max_rows=cfg.max_rows)
        P.append(model.predict_proba(cb))
        reps.append(model.phi(cb.target_values, cb.value_mask))
    P = torch.cat(P); reps = torch.cat(reps)
    p1 = P[:, 1].clamp(1e-9, 1 - 1e-9)
    h2 = -(p1 * torch.log2(p1) + (1 - p1) * torch.log2(1 - p1))
    y = (examples[0].answer_sheet.delta > 0) * 1
    labels = torch.full((len(P),), y, dtype=torch.long)
    return (float((1.0 - h2).mean()), float(M.ece(P, labels)["ece"]), float(h2.mean()),
            float(p1.mean()), reps.numpy(), p1.numpy())


def fit_probes(feats, targets, seed):
    """Ridge (primary) + fixed-HP MLP (secondary). Returns predict fns."""
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0).fit(feats, targets)
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(feats.shape[1], 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1))
    Xt = torch.from_numpy(feats).float(); yt = torch.from_numpy(targets).float().unsqueeze(1)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for _ in range(200):
        opt.zero_grad(); loss = torch.nn.functional.mse_loss(mlp(Xt), yt); loss.backward(); opt.step()
    mlp.eval()
    return (lambda f: ridge.predict(f),
            lambda f: mlp(torch.from_numpy(f).float()).detach().numpy().ravel())


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    bases = {n: cont_base(n) for n in sorted({d for _, (p, ev, _) in MODELS.items() for d in p + ev})}
    cells = all_cells()
    print(f"unique cells: {len(cells)}  (datasets x targets x idioms x deltas)")

    # ---- oracle sweep (seed-independent, cached) + example generation (deterministic per cell) ----
    print("=" * 100); print("ORACLE SWEEP + EXAMPLE GENERATION"); print("=" * 100)
    I_or, BE, EX = {}, {}, {}
    for i, (n, t, tn, idm, d) in enumerate(cells):
        I_or[(n, t, idm, d)], BE[(n, t, idm, d)] = oracle_cell(bases[n], t, idm, d, RNGState(seed=4242 + i))
        EX[(n, t, idm, d)] = gen_examples(bases[n], t, idm, d, RNGState(seed=777 + i))
        if d == DELTAS[-1]:
            print(f"  {n}.{tn:10} {idm:11} I_oracle(d=2.5)={I_or[(n, t, idm, d)]:.3f} (BE={BE[(n,t,idm,d)]:.3f})")

    # ---- gate state per eval cell (locked JSON; anchors = probe-training only, ungated) ----
    def gated_out(pool_key, dataset, t_name):
        for ev in ("nhanes", "hmda", "wealth"):
            rec = GATE["coverage"][pool_key].get(ev, {}).get(f"{dataset}.{t_name}")
            if rec is not None:
                return rec["gated_out"], rec["cov"]
        return None, None  # anchor (in-pool)

    # ---- subjects + per-cell estimates ----
    cfg = _cfg()
    rows = []  # one row per (D, seed, cell)
    spear = {"gain": [], "ridge": [], "mlp": []}
    spear_ind = {"ridge": [], "mlp": []}
    f4_pooled_ece = []
    for D, (pool_names, eval_names, pool_key) in MODELS.items():
        pool = [bases[n] for n in pool_names]
        va = [cont_base("survey_chile")]
        test_src_pool = [bases[n] for n in eval_names if n not in ANCHORS and n != "rb_scf2022_wealth_cont"]
        print("=" * 100); print(f"SUBJECTS held-out={D} pool={pool_names}"); print("=" * 100)
        for seed in SEEDS:
            torch.manual_seed(seed)
            out = train_level1(LODSurveyExampleSource(pool, tau_quantile=TAU_Q),
                               LODSurveyExampleSource(va, tau_quantile=TAU_Q),
                               LODSurveyExampleSource(test_src_pool, tau_quantile=TAU_Q),
                               cfg, RNGState(seed=seed), kind="ablation", run_id=f"det-{D}-{seed}",
                               git_commit=git, timestamp="2026-06-07T18:00:00Z")
            model = out["model"]
            anchor_feats, anchor_tgts, held_cells = [], [], []
            ho_P, ho_y = [], []
            for (n, t, tn, idm, d) in cells:
                if n not in eval_names:
                    continue
                gain, cell_ece, ent, mp1, reps, p1 = eval_cell(model, EX[(n, t, idm, d)], cfg)
                go, cov = gated_out(pool_key, n, tn)
                rec = {"D": D, "seed": seed, "dataset": n, "target": tn, "idiom": idm, "delta": d,
                       "I_oracle": I_or[(n, t, idm, d)], "I_gain": gain, "cell_ece": cell_ece,
                       "entropy": ent, "mean_p1": mp1, "gated_out": go, "cov": cov,
                       "kind": ("anchor" if n in ANCHORS else
                                ("control" if n == "rb_scf2022_wealth_cont" else "held_out"))}
                if rec["kind"] == "anchor":
                    anchor_feats.append(reps); anchor_tgts.append(np.full(len(reps), rec["I_oracle"]))
                else:
                    held_cells.append((rec, reps))
                    if rec["kind"] == "held_out" and not go:
                        ho_P.append(p1); ho_y.append(np.full(len(p1), 1 if d > 0 else 0))
                rows.append(rec)
            # probes (trained on anchors only)
            pr_ridge, pr_mlp = fit_probes(np.concatenate(anchor_feats), np.concatenate(anchor_tgts), seed)
            for rec, reps in held_cells:
                rec["I_ridge"] = float(np.mean(pr_ridge(reps)))
                rec["I_mlp"] = float(np.mean(pr_mlp(reps)))
            # per-seed pooled F4 ECE on gated-in held-out examples
            P = torch.from_numpy(np.concatenate(ho_P)); yv = torch.from_numpy(np.concatenate(ho_y)).long()
            P2 = torch.stack([1 - P, P], dim=1)
            f4_pooled_ece.append(float(M.ece(P2, yv)["ece"]))

    # ---- per-seed Spearman over gated-in HELD-OUT cells, pooled across both D ----
    for seed in SEEDS:
        sel = [r for r in rows if r["seed"] == seed and r["kind"] == "held_out" and not r["gated_out"]]
        oracle = [r["I_oracle"] for r in sel]
        spear["gain"].append(float(spearmanr(oracle, [r["I_gain"] for r in sel]).statistic))
        spear["ridge"].append(float(spearmanr(oracle, [r["I_ridge"] for r in sel]).statistic))
        spear["mlp"].append(float(spearmanr(oracle, [r["I_mlp"] for r in sel]).statistic))
        # in-dist context: anchors (probe TRAINING cells — optimistic by construction, context only)
        anc = [r for r in rows if r["seed"] == seed and r["kind"] == "anchor"]
        spear_ind["ridge"].append(float(spearmanr([r["I_oracle"] for r in anc],
                                                  [r["I_gain"] for r in anc]).statistic))

    def ms(a):
        a = np.array(a); return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))

    # F2 / J2: idiom separation on gated-in held-out delta>0 cells, per seed
    d_gain, d_probe, d_oracle = [], [], None
    for seed in SEEDS:
        sel = [r for r in rows if r["seed"] == seed and r["kind"] == "held_out"
               and not r["gated_out"] and r["delta"] > 0]
        tc = [r for r in sel if r["idiom"] == "top_coding"]; ov = [r for r in sel if r["idiom"] == "own_value"]
        d_gain.append(np.mean([r["I_gain"] for r in tc]) - np.mean([r["I_gain"] for r in ov]))
        d_probe.append(np.mean([r["I_ridge"] for r in tc]) - np.mean([r["I_ridge"] for r in ov]))
        if d_oracle is None:
            d_oracle = float(np.mean([r["I_oracle"] for r in tc]) - np.mean([r["I_oracle"] for r in ov]))

    # F3 from the locked gate file
    wealth_states = [s for pk in GATE["coverage"] for s in GATE["coverage"][pk]["wealth"].values()]
    hit = float(np.mean([s["gated_out"] for s in wealth_states]))
    covered = ([s for s in GATE["coverage"]["pool_nhanes"]["nhanes"].values()] +
               [s for s in GATE["coverage"]["pool_hmda"]["hmda"].values()])
    ff = float(np.mean([s["gated_out"] for s in covered]))

    # F4 tertile (pre-hoc operationalization): all eval cells (held-out + control), mean cell ECE
    # across seeds, worst tertile -> majority gated-out
    cellkeys = sorted({(r["D"], r["dataset"], r["target"], r["idiom"], r["delta"])
                       for r in rows if r["kind"] != "anchor"})
    cell_ece, cell_go = [], []
    for k in cellkeys:
        rs = [r for r in rows if (r["D"], r["dataset"], r["target"], r["idiom"], r["delta"]) == k]
        cell_ece.append(np.mean([r["cell_ece"] for r in rs])); cell_go.append(rs[0]["gated_out"])
    order = np.argsort(cell_ece)[::-1]
    worst = order[:len(order) // 3]
    tertile_out_frac = float(np.mean([cell_go[i] for i in worst]))

    # ---- criteria ----
    g_m, g_se = ms(spear["gain"]); r_m, r_se = ms(spear["ridge"]); m_m, m_se = ms(spear["mlp"])
    dg_m, dg_se = ms(d_gain); dp_m, dp_se = ms(d_probe)
    e_m, e_se = ms(f4_pooled_ece)
    diff_r = np.array(spear["ridge"]) - np.array(spear["gain"]); dr_m, dr_se = ms(diff_r)
    diff_m = np.array(spear["mlp"]) - np.array(spear["gain"]); dm_m, dm_se = ms(diff_m)

    F1 = (g_m - g_se) >= 0.50
    F2 = (dg_m > 2 * dg_se) and (np.sign(dg_m) == np.sign(d_oracle))
    F3 = (hit >= 0.90) and (ff <= 0.20)
    F4 = (e_m <= 0.20) and (tertile_out_frac > 0.5)
    J1 = ((dr_m - dr_se) >= 0.15) or ((dm_m - dm_se) >= 0.15)
    J2 = (not F2) and (dp_m > 2 * dp_se)
    J3 = J1  # J1/J2 computed leave-domain-out by construction; in-dist reported for context

    print("=" * 100); print("CRITERIA (locked)"); print("=" * 100)
    print(f"F1 Spearman(I_gain,I_oracle) gated-in OOF: {g_m:.3f}±{g_se:.3f} (mean-SE={g_m-g_se:.3f}; "
          f"need >=0.50) -> {'PASS' if F1 else 'FAIL'}   per-seed {np.round(spear['gain'],3).tolist()}")
    print(f"F2 idiom sep I_gain: {dg_m:+.4f}±{dg_se:.4f} (need >2SE & sign==oracle {d_oracle:+.3f}) "
          f"-> {'PASS' if F2 else 'FAIL'}")
    print(f"F3 gate: wealth hit={hit:.3f} (>=0.90), covered false-flag={ff:.3f} (<=0.20) "
          f"-> {'PASS' if F3 else 'FAIL'}")
    print(f"F4 gated-in pooled ECE={e_m:.3f}±{e_se:.3f} (<=0.20) & worst-tertile gated-out frac="
          f"{tertile_out_frac:.2f} (>0.5) -> {'PASS' if F4 else 'FAIL'}")
    print(f"J1 probe margin: ridge {dr_m:+.3f}±{dr_se:.3f} | mlp {dm_m:+.3f}±{dm_se:.3f} "
          f"(need mean-SE>=0.15) -> {'PASS' if J1 else 'FAIL'}   "
          f"[ridge OOF {r_m:.3f}±{r_se:.3f}, mlp OOF {m_m:.3f}±{m_se:.3f}]")
    print(f"J2 probe idiom sep where I_gain fails: {dp_m:+.4f}±{dp_se:.4f} "
          f"-> {'PASS' if J2 else ('n/a (F2 passed)' if F2 else 'FAIL')}")
    print(f"   context: anchor in-dist Spearman(I_gain,I_oracle) {ms(spear_ind['ridge'])[0]:.3f}")

    # ---- verdict per the locked outcome table ----
    if not F3:
        verdict = ("F3 FAIL -> finding against the D2 metric's COLUMN-LEVEL sufficiency; redirect to the "
                   "support metric, NOT a head; both estimators blocked on a valid gate.")
    elif F1 and F2 and F4:
        verdict = "F1-F4 PASS -> HEAD STRUCK; detectability = derived metric + governance reporting."
    elif (not (F1 and F2)) and J1 and (J2 or F2) and J3:
        verdict = "F1/F2 FAIL + J PASS -> head survives review (in-manifold estimator role only, gated)."
    elif not (F1 and F2):
        verdict = ("F1/F2 FAIL + J FAIL -> neither estimator trustworthy in-manifold; runtime "
                   "detectability DEFERRED (oracle-validated eval-time claims only).")
    else:
        verdict = "F4 FAIL alone -> partial F3; redirect to the support metric with ECE decomposition."
    print("\nVERDICT:", verdict)

    OUT.write_text(json.dumps({
        "git": git, "seeds": SEEDS, "criteria": {
            "F1": {"pass": bool(F1), "mean": g_m, "se": g_se, "per_seed": spear["gain"]},
            "F2": {"pass": bool(F2), "delta_gain": dg_m, "se": dg_se, "oracle_delta": d_oracle},
            "F3": {"pass": bool(F3), "wealth_hit": hit, "false_flag": ff},
            "F4": {"pass": bool(F4), "pooled_ece": e_m, "se": e_se, "tertile_out_frac": tertile_out_frac},
            "J1": {"pass": bool(J1), "ridge_margin": dr_m, "ridge_se": dr_se,
                   "mlp_margin": dm_m, "mlp_se": dm_se, "ridge_oof": r_m, "mlp_oof": m_m},
            "J2": {"pass": bool(J2), "delta_probe": dp_m, "se": dp_se}},
        "verdict": verdict, "rows": rows}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
