"""Experimental SLA: fixed-capacity FIFO KV plus a delta-rule state.

No training, retrieval index, or optimized GPU implementation.
"""
import numpy as np


class SLAMemory:
    def __init__(self, key_dim, value_dim, *, capacity=32, beta=0.05,
                 alpha=1.0, scale=32.0, null_logit=0.0, dtype=np.float64):
        if key_dim < 1 or value_dim < 1 or capacity < 0:
            raise ValueError("invalid dimensions or capacity")
        if not 0 < beta <= 1 or not 0 <= alpha <= 1:
            raise ValueError("invalid state parameters")
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(null_logit):
            raise ValueError("invalid logits")
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        self.dk, self.dv, self.capacity = key_dim, value_dim, capacity
        self.beta, self.alpha = beta, alpha
        self.scale, self.null_logit = scale, null_logit
        self.state = np.zeros((key_dim, value_dim), dtype=self.dtype)
        self.keys = np.zeros((capacity, key_dim), dtype=self.dtype)
        self.values = np.zeros((capacity, value_dim), dtype=self.dtype)
        self.ids = [None] * capacity
        self.size = 0

    def _unit(self, vector):
        x = np.asarray(vector, dtype=self.dtype)
        if x.shape != (self.dk,) or not np.isfinite(x).all():
            raise ValueError("invalid key or query")
        p = np.max(np.abs(x))
        if p == 0:
            raise ValueError("zero key/query")
        y = x.astype(np.float64) / float(p)
        return np.asarray(y / np.linalg.norm(y), dtype=self.dtype)

    def write(self, key, value, *, key_id=None):
        k = self._unit(key)
        v = np.asarray(value, dtype=self.dtype)
        if v.shape != (self.dv,) or not np.isfinite(v).all():
            raise ValueError("invalid value")
        if key_id is not None and not isinstance(key_id, (int, np.integer)):
            raise TypeError("key_id must be integer")
        v = v.copy()
        # Validation completed before mutations.
        self.state *= self.alpha
        self.state += self.beta * np.outer(k, v - k @ self.state)
        if self.capacity == 0:
            return
        match = next((i for i in range(self.size)
                      if key_id is not None and self.ids[i] == int(key_id)), None)
        if match is not None:
            self.keys[match], self.values[match] = k, v
            return
        if self.size < self.capacity:
            slot = self.size
            self.size += 1
        else:
            self.keys[:-1] = self.keys[1:]
            self.values[:-1] = self.values[1:]
            self.ids[:-1] = self.ids[1:]
            slot = self.capacity - 1
        self.keys[slot], self.values[slot], self.ids[slot] = k, v, key_id

    def read(self, query, *, residual_weight=0.0, details=False):
        if not 0 <= residual_weight <= 1:
            raise ValueError("residual_weight outside [0,1]")
        q = self._unit(query)
        fast = q @ self.state
        if not self.size:
            return (fast.copy(), {"null_weight": 1.0}) if details else fast.copy()
        scores = self.scale * (self.keys[:self.size] @ q)
        logits = np.concatenate(([self.null_logit], scores))
        w = np.exp(logits - np.max(logits))
        w = w / np.sum(w)
        out = w[0] * fast + w[1:] @ self.values[:self.size]
        if residual_weight:
            out += residual_weight * ((1 - w[0]) * fast - (w[1:] @ self.keys[:self.size]) @ self.state)
        out = np.asarray(out, dtype=self.dtype)
        return (out, {"null_weight": float(w[0])}) if details else out

    @property
    def archive_size(self):
        return self.size

    @property
    def allocated_array_bytes(self):
        """NumPy payload only; excludes Python IDs and temporary arrays."""
        return self.state.nbytes + self.keys.nbytes + self.values.nbytes
