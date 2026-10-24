"""Run with python -m examples.quickstart."""
import numpy as np
from summer_linear_attention import SLAMemory

m = SLAMemory(3, 2, capacity=2, dtype=np.float32)
m.write([1, 0, 0], [2, 3], key_id=1)
m.write([0, 1, 0], [4, 5], key_id=2)
print("SLA:", m.read([1, 0, 0]))
print("Soft residual hybrid:", m.read([1, 0, 0], residual_weight=0.5))
print("NumPy allocated bytes:", m.allocated_array_bytes)
