# Experiment log

This directory records every training sweep / ablation that has produced
evidence consulted in a decision. Companion to `docs/decisions/` — decisions
live there, evidence lives here, they cross-reference.

## Convention

- **Completed experiments:** one Markdown file per sweep, named
  `YYYY-MM-DD-short-slug.md`. Immutable once the experiment is run; if a
  later experiment supersedes its conclusion, write a new entry that
  cross-references rather than editing history.
- **Planned experiments:** single living file `PLANNED.md`. Entries move
  out of `PLANNED.md` into dated completed files when they run. This means
  at any moment `PLANNED.md` reflects the current queue.

## What each entry captures

```markdown
- Status
- Date run (or "planned")
- Goal / hypothesis
- Configuration (specs, seeds, cache, runtime)
- Results (metrics table with CIs; or "pending")
- Interpretation (what the numbers mean, plainly)
- Decisions taken or pending (link to ADR if one resulted)
- Raw data location (CSV path on /mnt/artifacts)
- Follow-ups queued or required
```

The goal is that a committee member reading these in chronological order
can reconstruct the research trajectory — what we asked, what we found,
what we did about it, what we asked next — without needing to reread code
or re-run anything. Evidence is durable; opinions about the evidence go
into ADRs.
