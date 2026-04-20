# Decision records

Architecture Decision Records (ADRs) for Lacuna. One file per decision,
numbered sequentially. Each record captures:

- **Context:** what was known at the time, what evidence drove the decision
- **Decision:** what was chosen
- **Consequences:** what this forecloses, what it enables, what we'd need to revisit it

ADRs are immutable once written. If a later decision supersedes an earlier
one, write a new ADR and link to the predecessor — do not edit history. The
point of this folder is that a committee member (or future-you) can read these
in order and reconstruct why the code looks the way it does.
