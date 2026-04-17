"""
lacuna.analysis

Offline statistical analysis utilities.

This package contains analyses that run OUTSIDE the model forward pass:
- Generator validation (verify synthetic generators produce claimed mechanisms)
- Ablation statistics (paired tests, bootstrap CIs, permutation tests) [Phase 2]

Everything here depends on pystatistics and runs on CPU. Nothing in this
package may be imported from the training hot path — these utilities allocate
Python objects, call into classical statistical solvers, and do not preserve
autograd.
"""
