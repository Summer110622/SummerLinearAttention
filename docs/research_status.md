# Research status — SLA v0.1

This package is an **untrained NumPy reference implementation**. It is not the complete set of previously prototyped research experiments.

Earlier synthetic investigations reported that:
- Null-aware readout improved error over a tuned SDA comparison on a structured-value/outlier noisy-query task with held-out seeds.
- Utility-based eviction improved a four-task macro-average, but **severely degraded recent-item recall**.
- Softmax mixing blurred opposing values attached to highly similar keys, despite correct top-1 retrieval.
- A fixed 50:50 SDA/SLA hybrid was not uniformly superior to standalone SDA or SLA.

These are preliminary, task-dependent findings; **they do not establish a trained-language-model win over KDA, GDN, or Full Attention**.

## Next research tasks

1. Partition archive into recent-protected and utility-selected regions, subject to one fixed budget.
2. Calibrate null / residual / sharpness gates on separate development seeds.
3. Include same-memory simple KV baselines and contradiction/near-duplicate stress tests.
4. Evaluate fully trained language models, GPU throughput, latency and true peak RAM.

Companion reference: https://github.com/Summer110622/SummerDeltaAttention
