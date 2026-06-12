"""
scripts/diag_t1_network_fit.py

RIG-VALIDITY diagnostic for the T1 network arm (NOT tuning; per the no-train-and-blame discipline).
The T1 run shows the network at chance on everything incl. MCAR-vs-not (trivially learnable, v1.0
history). Discriminating question: can the network arm FIT ITS OWN TRAINING DATA at all?

  - If TRAIN macro-AUC also ~0.5 after training => the harness is broken (tokenization shim / loss
    wiring / optimizer path) — a bug; find and fix, then rerun T1 unchanged.
  - If TRAIN fits high but in-domain VAL stays ~0.5 => genuine failure to generalize even in-domain
    (a real result; verdict scored as-is).

Prints per-epoch: train loss components, TRAIN macro-AUC (subset), VAL macro-AUC. One split
(held-out=wealth => train domains labor+nhanes+hmda), one seed, reduced sizes for speed.
Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/diag_t1_network_fit.py
"""

import numpy as np
import torch

import importlib

m = importlib.import_module("scripts.run_t1_showdown")

N_TR, N_VA, EPOCHS = 480, 192, 15
SEED = 2026


def main():
    print(f"device={m.DEVICE}")
    tr = m.make_examples(["hmda", "labor", "nhanes"], N_TR, 71000)
    va = m.make_examples(["hmda", "labor", "nhanes"], N_VA, 75000)
    ytr = np.array([e["y"] for e in tr])
    yva = np.array([e["y"] for e in va])
    print(f"train {len(tr)} val {len(va)}; class balance tr {np.bincount(ytr)} va {np.bincount(yva)}")

    torch.manual_seed(SEED)
    cfg = m.LacunaModelConfig(hidden_dim=128, evidence_dim=64, n_layers=4, n_heads=4,
                              max_cols=m.MAX_COLS, use_missingness_features=False)
    model = m.LacunaModel(cfg).to(m.DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_par:,}")
    opt = torch.optim.Adam(model.parameters(), lr=m.LR_RATE)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 400))

    for ep in range(EPOCHS):
        model.train()
        mrng = np.random.default_rng(SEED * 1000 + ep)
        perm = np.random.default_rng(SEED + ep).permutation(len(tr))
        ce_sum, rl_sum, nb = 0.0, 0.0, 0
        for s in range(0, len(tr), m.BS):
            ex = [tr[i] for i in perm[s:s + m.BS]]
            b = m._to(m.to_batch(ex, mrng), m.DEVICE)
            out = model(b, compute_reconstruction=True, compute_decision=False)
            y = torch.tensor([e["y"] for e in ex], device=m.DEVICE)
            ce = m.class_cross_entropy(out.posterior.p_class, y)
            rl, _ = m.multi_head_reconstruction_loss(out.reconstruction, b.original_values,
                                                     b.reconstruction_mask, b.row_mask, b.col_mask)
            (ce + rl).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            ce_sum += float(ce); rl_sum += float(rl); nb += 1
        Ptr = m.net_probs(model, tr[:N_VA], np.random.default_rng(777))
        Pva = m.net_probs(model, va, np.random.default_rng(777))
        tr_auc = m.macro_auc(Ptr, ytr[:N_VA])
        va_auc = m.macro_auc(Pva, yva)
        # gradient sanity: p_class spread
        print(f"ep {ep:2d} | CE {ce_sum/nb:.4f} recon {rl_sum/nb:.4f} | "
              f"TRAIN-fit AUC {tr_auc:.3f} | VAL AUC {va_auc:.3f} | "
              f"p_class std {Pva.std(axis=0).round(3)}")

    print("\nDIAGNOSIS: train-fit ~0.5 => HARNESS BUG; train-fit high + val ~0.5 => real in-domain "
          "generalization failure; both high => T1 run's failure needs a different explanation.")


if __name__ == "__main__":
    main()
