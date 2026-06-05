"""
scripts/audit_distributional_stream_live.py

GATE 1 — stream-live / usage audit (PI-mandated before ANY scientific interpretation of the
distributional-stream A/B). Proves the rep-ECDF stream actually ENTERS the computation and is USED,
so a null A/B cannot be a silent-plumbing artifact. Small/CPU/deterministic — this is a MECHANISM
check, not a performance run; numbers here are not Lacuna results.

Checks (PI list):
  1 stream enabled on the model object
  2 manifest records schema + out_dim
  3 head input dim changes by the expected amount (off vs on)
  4 stream output nonconstant across examples AND differs across δ (init + post-train)
  5 gradients flow through the stream back to the encoder
  6 stream/head/encoder parameters update during training
  7 ablation sensitivity: zero / shuffle the stream output at eval -> do predictions change?
  8 first-layer head weight-norms by input block (evidence | target-summary | fixed-ECDF)
  9 training curves stream-off vs stream-on

Run: python -u scripts/audit_distributional_stream_live.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.batching import collate
from lacuna.survey.coarse_bins import assign_bins
from lacuna.survey.conditioned_head import create_target_conditioned_model
from lacuna.survey.consequence_features import N_FEATURES
from lacuna.survey.example_source import LODSurveyExampleSource
from lacuna.survey.loss import rps_loss, uniform_rps
from lacuna.survey.train import TrainConfig, _make_examples, train_delta_prior

TRAIN = ["survey_cps1988", "survey_yrbss"]
TEST = ["survey_cps1985", "survey_workinghours"]
GRID = [0.0, 2.5]  # binary δ contrast (clean LOD presence/absence)
SCHEME = "binary"
EPOCHS = 10


def _cfg(stream_on):
    return TrainConfig(
        delta_grid=GRID, beta1_range=(0.0, 2.0), target_rate=0.3,
        max_rows=256, max_cols=12, batch_size=12, train_batches_per_epoch=20,
        max_epochs=EPOCHS, patience=EPOCHS, val_size=72, test_size=72,
        hidden_dim=64, evidence_dim=32, n_layers=2, n_heads=4, dropout=0.05,
        target_conditioned=True, consequence_features=True,
        rep_ecdf_pooling=stream_on, n_shape_probes=4, coarse_scheme=SCHEME, model_kind="auto",
    )


def _model(stream_on, rng):
    cfg = _cfg(stream_on)
    return create_target_conditioned_model(
        hidden_dim=cfg.hidden_dim, evidence_dim=cfg.evidence_dim, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, max_cols=cfg.max_cols, dropout=cfg.dropout,
        num_bins=2, n_consequence_features=N_FEATURES,
        rep_ecdf_pooling=stream_on, n_shape_probes=cfg.n_shape_probes, rng=rng,
    ), cfg


def _labels(db):
    return assign_bins(SCHEME, db.delta)


def _stream_vec(model, db):
    """Return the rep-ECDF stream output [B, out_dim] for a batch (eval, no grad)."""
    model.eval()
    with torch.no_grad():
        enc = model.encoder(db.tokens.tokens, db.tokens.row_mask, db.tokens.col_mask,
                            return_intermediates=True)
        tgt = model._gather_target(enc["token_representations"], db.target_idx)
        return model.rep_pool(tgt, db.tokens.row_mask)


def _head_pieces(model, db):
    """Replicate the conditioned-head forward, exposing the concat blocks (for ablation + norms)."""
    enc = model.encoder(db.tokens.tokens, db.tokens.row_mask, db.tokens.col_mask,
                        return_intermediates=True)
    evidence = enc["evidence"]
    tgt = model._gather_target(enc["token_representations"], db.target_idx)
    summary = model.rep_pool(tgt, db.tokens.row_mask)
    cons = model.consequence_norm(db.consequence)
    return evidence, summary, cons


def _train(model, cfg, datasets, rng):
    """Minimal training loop logging per-epoch val RPS; returns (curve, before/after snapshots)."""
    pool = [create_default_catalog().load(n) for n in datasets]
    src = LODSurveyExampleSource(pool, tau_quantile=0.70)
    val_ex = _make_examples(src, cfg, cfg.val_size, rng.spawn(), stratify=True)
    val_db = collate(val_ex, max_rows=cfg.max_rows, max_cols=cfg.max_cols)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    snap = lambda t: t.detach().clone()
    before = {
        "proj": snap(model.rep_pool.proj.weight) if model.rep_ecdf_pooling else None,
        "head0": snap(model.head.net[0].weight),
        "enc_q": snap(model.encoder.transformer_layers[0].q_proj.weight),
    }
    curve, train_rng = [], rng.spawn()
    for _ in range(cfg.max_epochs):
        model.train()
        for _b in range(cfg.train_batches_per_epoch):
            ex = _make_examples(src, cfg, cfg.batch_size, train_rng.spawn())
            db = collate(ex, max_rows=cfg.max_rows, max_cols=cfg.max_cols)
            logits = model(db.tokens, db.target_idx, db.consequence)
            loss = rps_loss(logits, _labels(db))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        model.eval()
        with torch.no_grad():
            vl = model(val_db.tokens, val_db.target_idx, val_db.consequence)
            curve.append(round(float(rps_loss(vl, _labels(val_db)).item()), 4))
    after = {
        "proj": snap(model.rep_pool.proj.weight) if model.rep_ecdf_pooling else None,
        "head0": snap(model.head.net[0].weight),
        "enc_q": snap(model.encoder.transformer_layers[0].q_proj.weight),
    }
    return curve, before, after


def _sep(vec, deltas):
    """Separation of a [B, D] representation between δ=0 and δ>0 groups: ||Δmean|| / pooled std."""
    d = deltas.numpy()
    a, b = vec[d == 0.0].numpy(), vec[d > 0.0].numpy()
    if len(a) == 0 or len(b) == 0:
        return None
    dmean = np.linalg.norm(a.mean(0) - b.mean(0))
    pooled = np.sqrt(0.5 * (a.var(0).mean() + b.var(0).mean())) + 1e-9
    return float(dmean / pooled)


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    torch.manual_seed(0)
    Path("runs").mkdir(exist_ok=True)
    cat = create_default_catalog()
    rep = {"git": git, "checks": {}}
    P = lambda k, v: print(f"  [{k}] {v}")
    print("=" * 96); print("GATE 1 — DISTRIBUTIONAL-STREAM LIVE / USAGE AUDIT (binary LOD; small CPU mechanism check)")
    print("=" * 96)

    # build off/on models
    m_off, cfg_off = _model(False, RNGState(seed=1))
    m_on, cfg_on = _model(True, RNGState(seed=1))

    # 1 enabled
    rep["checks"]["1_enabled"] = {"off": m_off.rep_ecdf_pooling, "on": m_on.rep_ecdf_pooling,
                                  "on_has_rep_pool": hasattr(m_on, "rep_pool")}
    P("1 enabled", f"off={m_off.rep_ecdf_pooling} on={m_on.rep_ecdf_pooling} "
                   f"on.rep_pool={hasattr(m_on,'rep_pool')}  -> {'PASS' if (m_on.rep_ecdf_pooling and not m_off.rep_ecdf_pooling) else 'FAIL'}")

    # 2 manifest schema (real short main-path manifest)
    pool_tr = [cat.load(n) for n in TRAIN]; pool_te = [cat.load(n) for n in TEST]
    src = lambda p: LODSurveyExampleSource(p, tau_quantile=0.70)
    out = train_delta_prior(src(pool_tr), src(pool_te), src(pool_te), _cfg(True), RNGState(seed=2),
                            kind="ablation", run_id="gate1-manifest", git_commit=git,
                            timestamp="2026-06-05T00:00:00Z")
    ma = out["manifest"]["model_arch"]
    rep["checks"]["2_manifest"] = {"target_summary": ma["target_summary"], "schema": ma["rep_ecdf_schema"]}
    P("2 manifest", f"target_summary={ma['target_summary']} schema={ma['rep_ecdf_schema']}")

    # 3 head input dim
    e, h = cfg_on.evidence_dim, cfg_on.hidden_dim
    in_off, in_on = m_off.head.net[0].in_features, m_on.head.net[0].in_features
    exp_off, exp_on = e + h + N_FEATURES, e + m_on.rep_pool.out_dim + N_FEATURES
    rep["checks"]["3_head_dim"] = {"off": in_off, "on": in_on, "expected_off": exp_off,
                                   "expected_on": exp_on, "stream_block": m_on.rep_pool.out_dim}
    P("3 head dim", f"off={in_off}(exp {exp_off}) on={in_on}(exp {exp_on}) "
                    f"stream_block={m_on.rep_pool.out_dim}  -> {'PASS' if (in_off==exp_off and in_on==exp_on) else 'FAIL'}")

    # eval batch (mixed δ) for 4/5/7/8
    eval_ex = _make_examples(src(pool_te), _cfg(True), 48, RNGState(seed=5), stratify=True)
    eval_db = collate(eval_ex, max_rows=256, max_cols=12)

    # 4a stream nonconstant + δ-separation at INIT
    s_init = _stream_vec(m_on, eval_db)
    sep_init = _sep(s_init, eval_db.delta)
    nonconst_init = float(s_init.std(0).mean())
    P("4a init", f"stream std-across-examples={nonconst_init:.4f} (nonconstant={nonconst_init>1e-4}) "
                 f"δ-separation={sep_init:.3f}")

    # train on/off (curves + snapshots)
    curve_on, before_on, after_on = _train(m_on, _cfg(True), TRAIN, RNGState(seed=7))
    curve_off, before_off, after_off = _train(m_off, _cfg(False), TRAIN, RNGState(seed=7))

    # 4b post-train
    s_post = _stream_vec(m_on, eval_db)
    sep_post = _sep(s_post, eval_db.delta)
    nonconst_post = float(s_post.std(0).mean())
    rep["checks"]["4_stream_signal"] = {"init_nonconst": nonconst_init, "init_sep": sep_init,
                                        "post_nonconst": nonconst_post, "post_sep": sep_post}
    P("4b post", f"stream std={nonconst_post:.4f} δ-separation={sep_post:.3f}  "
                 f"-> {'PASS' if (nonconst_post>1e-4 and sep_post and sep_post>0.1) else 'CONCERN'}")

    # 5 gradient flow through stream to encoder
    m_on.train(); m_on.zero_grad()
    logits = m_on(eval_db.tokens, eval_db.target_idx, eval_db.consequence)
    rps_loss(logits, _labels(eval_db)).backward()
    g_enc = float(sum(p.grad.abs().sum() for p in m_on.encoder.parameters() if p.grad is not None))
    g_proj = float(m_on.rep_pool.proj.weight.grad.abs().sum())
    rep["checks"]["5_grad_flow"] = {"encoder_grad": g_enc, "rep_proj_grad": g_proj}
    P("5 grad flow", f"encoder_grad={g_enc:.3e} rep_proj_grad={g_proj:.3e}  "
                     f"-> {'PASS' if (g_enc>0 and g_proj>0) else 'FAIL'}")

    # 6 param updates
    def chg(a, b):
        return float((a - b).norm() / (b.norm() + 1e-9))
    upd = {"rep_proj": chg(after_on["proj"], before_on["proj"]),
           "head0_on": chg(after_on["head0"], before_on["head0"]),
           "encoder_q_on": chg(after_on["enc_q"], before_on["enc_q"])}
    rep["checks"]["6_param_update"] = upd
    P("6 updates", f"rel-L2 Δ: rep_proj={upd['rep_proj']:.3f} head0={upd['head0_on']:.3f} "
                   f"enc_q={upd['encoder_q_on']:.3f}  -> {'PASS' if all(v>1e-3 for v in upd.values()) else 'FAIL'}")

    # 7 ablation sensitivity (zero / shuffle stream at eval on the trained model)
    m_on.eval()
    with torch.no_grad():
        ev, summ, cons = _head_pieces(m_on, eval_db)
        perm = torch.from_numpy(RNGState(seed=11).shuffle_indices(summ.shape[0])).long()
        def head(s): return m_on.head(torch.cat([ev, s, cons], -1)) / m_on.temperature.clamp(min=1e-6)
        lab = _labels(eval_db)
        base_l, zero_l, shuf_l = head(summ), head(torch.zeros_like(summ)), head(summ[perm])
        def met(l):
            p = torch.softmax(l, -1)
            return float(rps_loss(l, lab).item()), float(roc_auc_score(lab.numpy(), p[:, 1].numpy()))
        b_rps, b_auc = met(base_l); z_rps, z_auc = met(zero_l); s_rps, s_auc = met(shuf_l)
        d_zero = float((base_l - zero_l).abs().mean()); d_shuf = float((base_l - shuf_l).abs().mean())
    rep["checks"]["7_ablation"] = {"base": [b_rps, b_auc], "zeroed": [z_rps, z_auc],
                                   "shuffled": [s_rps, s_auc], "mean_abs_logit_delta_zero": d_zero,
                                   "mean_abs_logit_delta_shuffle": d_shuf}
    P("7 ablation", f"base rps/auc={b_rps:.3f}/{b_auc:.3f} | zeroed={z_rps:.3f}/{z_auc:.3f} | "
                    f"shuffled={s_rps:.3f}/{s_auc:.3f} | mean|Δlogit| zero={d_zero:.3f} shuf={d_shuf:.3f}  "
                    f"-> {'USED' if max(d_zero,d_shuf)>1e-2 else 'IGNORED'}")

    # 8 head weight block norms (per-input-column L2, normalized by block width)
    def blocks(model, summary_dim):
        W = model.head.net[0].weight.detach()  # [hidden, cond_dim]
        col = W.norm(dim=0)  # per-input-column norm
        e0 = model.encoder.config.evidence_dim
        ev_n = float(col[:e0].mean())
        sm_n = float(col[e0:e0 + summary_dim].mean())
        cn_n = float(col[e0 + summary_dim:e0 + summary_dim + N_FEATURES].mean())
        return {"evidence": ev_n, "target_summary": sm_n, "fixed_ecdf": cn_n}
    bn_on = blocks(m_on, m_on.rep_pool.out_dim)
    bn_off = blocks(m_off, m_off.encoder.config.hidden_dim)
    rep["checks"]["8_block_norms"] = {"on": bn_on, "off": bn_off}
    P("8 block norms ON ", f"evidence={bn_on['evidence']:.3f} stream={bn_on['target_summary']:.3f} "
                           f"fixedECDF={bn_on['fixed_ecdf']:.3f}")
    P("8 block norms OFF", f"evidence={bn_off['evidence']:.3f} mean-target={bn_off['target_summary']:.3f} "
                           f"fixedECDF={bn_off['fixed_ecdf']:.3f}")

    # 9 curves
    rep["checks"]["9_curves"] = {"uniform_rps": uniform_rps(2), "off": curve_off, "on": curve_on}
    P("9 curves OFF", f"val-rps/epoch {curve_off}")
    P("9 curves ON ", f"val-rps/epoch {curve_on}")

    # overall gate verdict
    used = max(d_zero, d_shuf) > 1e-2
    live = (m_on.rep_ecdf_pooling and in_on == exp_on and g_proj > 0 and upd["rep_proj"] > 1e-3
            and nonconst_post > 1e-4)
    rep["gate1_live"] = bool(live); rep["gate1_used"] = bool(used)
    verdict = ("PASS — stream is LIVE (enabled, correct dims, gradients+updates, nonconstant) and USED "
               "(ablating it changes predictions). Proceed to Gate 2 (full-scale GPU)."
               if (live and used) else
               "FAIL — stream is not genuinely live/used; fix plumbing/integration before interpretation.")
    print("\n" + "=" * 96); print(f"GATE 1 VERDICT: {verdict}"); print("=" * 96)
    rep["verdict"] = verdict
    Path("runs/gate1-stream-live-audit.json").write_text(json.dumps(rep, indent=2))
    print("saved: runs/gate1-stream-live-audit.json")


if __name__ == "__main__":
    main()
