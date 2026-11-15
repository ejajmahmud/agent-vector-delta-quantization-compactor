"""Automated test suite for Vector Quantizer."""
import unittest
from agent_vector_delta_quantization_compactor.core import VectorQuantizer

class TestVectorQuantizer(unittest.TestCase):
    def test_quantize_dequantize_similarity(self):
        vec = [1.0, 2.0, 3.0, 4.0, 5.0]
        q = VectorQuantizer.quantize(vec)
        self.assertEqual(len(q.q_bytes), len(vec))
        restored = VectorQuantizer.dequantize(q)
        sim = VectorQuantizer.cosine_similarity(vec, restored)
        self.assertGreater(sim, 0.99)
