# Summer Linear Attention (SLA)

Research reference for **bounded external-memory linear attention**. SLA combines a recurrent delta-rule state, a FIFO archive of original key/value pairs, and a softmax readout with a null/state fallback. This is **not a trained LLM or optimized GPU kernel**.

Companion project: [SummerDeltaAttention](https://github.com/Summer110622/SummerDeltaAttention).

## Quick start

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m examples.quickstart
```

## Formula

With normalized keys and queries, update the state as

```text
Sbar = alpha * S
S = Sbar + beta * outer(k, v - Sbar.T @ k)
```

Readout uses `weights = softmax([null_logit, scale * archive_keys @ q])` and

```text
y = weights[0] * (S.T @ q) + sum_i weights[i+1] * archive_value[i]
```

An experimental shared-state SDA/SLA hybrid can add
`lambda * sum_i weights[i+1] * (S.T @ (q - archive_key[i]))`,
where `0 <= lambda <= 1`. This is *soft* residual correction, not exact top-1 SDA.

## Limitations

- The FIFO archive has a **fixed capacity B** and exact linear scan. Memory is O(dk*dv + B*(dk+dv)); read compute is O(dk*dv + B*(dk+dv)). No constant-memory exact recall claim.
- Softmax can mix distinct near-duplicate keys with contradictory values. Exact semantic-key recall is not guaranteed.
- The null candidate is not a calibrated probability of an absent answer.
- Utility eviction is not the default: earlier synthetic stress tests showed significant recency regressions.
- Prior synthetic experiments do **not** establish superiority over trained KDA, GDN or Full Attention.
- Persistent array memory and Python process peak memory differ.

See [research status](docs/research_status.md). MIT License; unreviewed research.
