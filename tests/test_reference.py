import unittest
import numpy as np
from summer_linear_attention import SLAMemory


class SLATests(unittest.TestCase):
    def test_zero_capacity_delta(self):
        m = SLAMemory(2, 1, capacity=0, beta=0.5)
        m.write([1, 0], [4])
        np.testing.assert_allclose(m.read([1, 0]), [2])

    def test_float32_rewrite_same_id(self):
        m = SLAMemory(2, 1, dtype=np.float32)
        k = [0.6404226422309875, 0.10490011423826218]
        m.write(k, [1], key_id=7)
        m.write(k, [9], key_id=7)
        self.assertEqual(m.archive_size, 1)
        np.testing.assert_allclose(m.values[0], [9])

    def test_different_ids_same_key(self):
        m = SLAMemory(2, 1)
        m.write([1, 0], [1], key_id=1)
        m.write([1, 0], [2], key_id=2)
        self.assertEqual(m.archive_size, 2)

    def test_fifo_eviction(self):
        m = SLAMemory(3, 1, capacity=2)
        for i in range(3):
            m.write(np.eye(3)[i], [i], key_id=i)
        self.assertEqual(m.ids, [1, 2])

    def test_invalid_write_is_atomic(self):
        m = SLAMemory(2, 1)
        with self.assertRaises(ValueError):
            m.write([0, 0], [1])
        self.assertEqual(m.archive_size, 0)
        np.testing.assert_array_equal(m.state, np.zeros((2, 1)))

    def test_huge_normalization(self):
        m = SLAMemory(2, 1, capacity=1)
        m.write([1e200, 1e200], [3])
        self.assertTrue(np.isfinite(m.read([1e-200, 1e-200])).all())

    def test_hybrid_lambda_zero(self):
        m = SLAMemory(2, 1)
        m.write([1, 0], [3])
        np.testing.assert_allclose(m.read([1, 0]), m.read([1, 0], residual_weight=0))

    def test_allocated_bytes(self):
        m = SLAMemory(32, 16, capacity=100)
        self.assertEqual(m.allocated_array_bytes, (32*16 + 100*48)*8)

    def test_nonzero_lambda_on_identical_single_key(self):
        m = SLAMemory(2, 1, capacity=1)
        m.write([1, 0], [3])
        np.testing.assert_allclose(m.read([1, 0]), m.read([1, 0], residual_weight=1), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
