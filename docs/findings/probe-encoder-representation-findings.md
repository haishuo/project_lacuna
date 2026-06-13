# Representation Probe — findings (Architecture-Change-Audit §8)

*Diagnostic findings. Status: **for PI review.** No architecture was built or changed; this is the §8
reuse-vs-rewrite probe the patch's A/B could not provide. Governed by `NORTH-STAR.md` (§4.9 weak-proxy
discipline, §5 oracle/metrics) and `ARCHITECTURE-CHANGE-AUDIT-distributional-stream.md`. Script:
`scripts/probe_encoder_representation.py`; raw JSON: `runs/probe-encoder-representation.json`.*

## 0. Question

Does the **unchanged** BERT-style backbone's representation linearly **contain / preserve** the LOD δ
signal that raw observed-value ECDF features carry out-of-family? This separates *"backbone destroys the
signal → rewrite (Option C)"* from *"head/objective fails to use a signal the reps retain → patch/salvage."*

## 1. Result (binary LOD δ0-vs-δ2.5; held-out leave-datasets-out; train=8 / test=2 survey datasets)

| representation (held-out OOF) | AUC | preservation R² overall | preservation R² tail-feats |
|---|---|---|---|
| **raw-ECDF features** (column-primary, the known signal) | **0.722** | — | — |
| frozen **random-init** encoder — best linear probe | 0.546 | −3.29 | −3.05 |
| frozen **δ-trained** encoder (18 ep) — best linear probe | 0.530 | −0.88 | −2.13 |
| full trained model (Gate-2, reference) | ~0.59 | — | — |

Per-source probe breakdown (δ-trained): evidence-only 0.523, target-reps 0.530, evidence+target 0.523.
Preservation = Ridge reconstruction of the 17 raw ECDF features **from the encoder reps**; R² < 0 means
the reps predict the held-out raw features *worse than the test mean* (no transferable linear retention).

## 2. Reading (reported, cross-checked — not a standalone kill, per §4.9)

1. **The backbone's averaged representations do not carry the transferable OOF signal.** Best frozen
   probe ≈ **0.52–0.55** vs raw-ECDF **0.722** — barely above chance. This holds for **both** the
   random-init *and* the δ-trained encoder, so it is **not** a "trained-badly" artifact: 18 epochs of
   end-to-end δ training left the reps no more readable than random init.
2. **The reps do not even linearly preserve the discriminative upper-tail features** (R² ≤ 0, both
   encoders). The preservation probe is **positive-capable** (a high R² would have proven retention);
   its failure is therefore meaningful, not just "the linear δ-classifier is weak."
3. **Mechanistically coherent with the fitness thesis.** The reps I probe are across-row **averages**
   (`evidence` = attention-mean over rows; target = masked-mean over rows). Averaging destroys order
   statistics, so quantile/tail information is not recoverable from them — exactly what the table shows.
4. **Direct representation-family contrast.** A **column-primary distribution** representation (raw
   ECDF) carries **0.722**; the **row-primary averaged** backbone representation carries **~0.52**. The
   representational *family*, not just training, governs whether the signal is accessible.
5. **Convergent, mutually-reinforcing evidence** (no single proxy is a kill, but they agree):
   real from-scratch model 0.59 ≪ 0.754 baseline (Gate-2, the *real* model not a proxy) **+** frozen
   probe ~0.52 across both encoder states **+** preservation R² ≤ 0. Three independent angles point the
   same way: the current backbone is the binding constraint for this signal.

## 3. Honest caveats (do not over-read)

- **Weak-proxy negative (§4.9).** A frozen *linear* probe negative cannot, alone, license a rewrite. The
  weight here comes from **convergence** (real model + both encoder states + positive-capable
  preservation probe), not any one number.
- **Gate-1 nuance, not contradicted.** Gate-1 showed the rep-ECDF *pooling* output is δ-separated
  **in-distribution** (1.58) — i.e. per-row reps do carry per-row value info. The probe measures
  **out-of-family transfer** of *averaged* summaries; the two are consistent: per-row info exists, but
  the transferable order-statistic summary does not survive the backbone's averaging/representation.
- **Negative R² conflates "absent" with "non-transferable encoding."** Either way the backbone does not
  yield a *transferable within-column distribution* representation — which is what δ needs.
- **Not a claim about identifiability.** The oracle ceiling still binds; this is about representation,
  not about beating non-identifiability.

## 4. Decision bearing (§8 criteria)

Per the audit's pre-registered criteria: probe ≪ raw-ECDF **and** reps do not preserve the tail features
⇒ the branch is **"backbone discards / does not transferably surface the within-column order-stat
signal → supports rewrite / Option C (column-primary)."** This is now **sufficiently supported to
justify designing Option C** (the column-primary distribution architecture), with the final reuse-vs-
rewrite call confirmed when an Option-C representation is shown to recover the signal OOF (or fails to,
which would instead implicate the transfer ceiling / regime, per DECISION-MEMO §9). It does **not**
authorize an implementation yet — spec first.
