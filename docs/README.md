# Lacuna Documentation Index

Docs are organized by **type** (durable) and the current working set is flagged here (navigable).
When in doubt, start with this file.

---

## ★ START HERE — current working set (the recoverability reformulation)

The project reset to a new design philosophy on 2026-06-12. These are the live documents:

| doc | what it is |
|---|---|
| [proposals/PROPOSAL-recoverability-reformulation.md](proposals/PROPOSAL-recoverability-reformulation.md) | **The thesis of record.** Recoverability as the object; functional bounds; certified coverage; R0–R3 output; the three trained-network slots. Read first. |
| [LITERATURE-REVIEW.md](LITERATURE-REVIEW.md) | Positions all 7 design limbs vs the state of the art (20+ sources); novelty map; carried risks. |
| [proposals/GENERATOR-DESIGN-charter.md](proposals/GENERATOR-DESIGN-charter.md) | Generators as the load-bearing soundness condition; 4-layer realism; validation protocol; build order. |
| [findings/T-review-findings.md](findings/T-review-findings.md) | The locked T1/T2/T3 verdict (fail/fail/fail) that ended the matrix-detector approach. |
| [findings/REAL-MISSINGNESS-stage1-findings.md](findings/REAL-MISSINGNESS-stage1-findings.md) | Matrix channel settled NO on real labels; semantic signal exists & transfers. |
| [proposals/PROPOSAL-semantic-channel-pivot.md](proposals/PROPOSAL-semantic-channel-pivot.md) | The text→behavior channel (Arm 1/2/3 design). |
| [proposals/PROPOSAL-real-missingness-showdown-and-learned-generator.md](proposals/PROPOSAL-real-missingness-showdown-and-learned-generator.md) | Real-label showdown + learned conditional generator (Part B). |
| [data/DATA-CITATIONS.md](data/DATA-CITATIONS.md) | Every external dataset used + on-disk path + role. Cite before first use. |

**Next build step:** generator audit → realism-gate harness (see the charter §9).

---

## Folder map

- **`proposals/`** — specs, plans, pre-registrations (`PROPOSAL-*`, `PREREGISTRATION-*`). Mix of live (above) and historical-arc specs.
- **`findings/`** — experimental results (`*-findings`, `*-RESULT`, `probe-*`, the `feasibility-*` arc, phase summaries).
- **`decisions/`** — Architecture Decision Records (`0001`–`0005`), decision memos, consolidation memos. Durable rulings.
- **`experiments/`** — the dated lab notebook (`2026-04-*`), evidence standards, planned experiments.
- **`architecture/`** — architecture investigations/audits and the (now-historical) survey-arc `MASTER` + inference-object design docs.
- **`data/`** — data inventory, role-B projection, acquisition framework, survey-realism, citations.
- **`v1.0/`** — the original MoE/BERT-era mechanism + realism docs (reproduced baseline lineage).
- **root** — `NORTH-STAR.md` (governance charter), `ARCHITECTURE.md`, `CHANGELOG.md`, `LITERATURE-REVIEW.md`, this index.

---

## The arc, in one paragraph (so the history is legible)

**v1.0** (`v1.0/`, `NORTH-STAR.md`): MoE/BERT dataset-level mechanism classifier, ~92% — later found rate-confounded and never feature-nulled. **Survey arc** (`architecture/MASTER-*`, P1/P2 `feasibility-*` in `findings/`, most of `proposals/`): pivoted to a δ-prior/governance tool; column-primary φ-spine; extensive feasibility laddering — concluded the matrix channel is statistics-sufficient or non-identifiable. **Network load-bearing review** (`proposals/PREREGISTRATION-network-load-bearing-review.md`, `findings/T-review-findings.md`): pre-registered T1/T2/T3 → fail/fail/fail → scrap/pivot. **Real-missingness + semantic** (`findings/REAL-MISSINGNESS-*`, semantic proposals): matrix settled NO on real labels; text→behavior signal found. **Recoverability reformulation** (current working set): the synthesis — measure recoverability, bound it, certify on plasmode, anchor δ from the literature.

> Historical documents are preserved as written (a research record); their inline links may use
> pre-reorg flat paths. Use this index and the folder structure to navigate. Code references were
> updated to the new paths.
