"""Core vector delta quantization compactor."""
import math
from typing import List, Tuple

class QuantizedVector:
    def __init__(self, q_bytes: bytes, min_val: float, max_val: float):
        self.q_bytes = q_bytes
        self.min_val = min_val
        self.max_val = max_val

class VectorQuantizer:
    @staticmethod
    def quantize(vector: List[float]) -> QuantizedVector:
        min_v = min(vector)
        max_v = max(vector)
        diff = max_v - min_v if max_v != min_v else 1.0

        q_bytes = bytearray()
        for v in vector:
            norm = (v - min_v) / diff
            q_val = min(255, max(0, int(norm * 255)))
            q_bytes.append(q_val)
        return QuantizedVector(bytes(q_bytes), min_v, max_v)

    @staticmethod
    def dequantize(q: QuantizedVector) -> List[float]:
        diff = q.max_val - q.min_val
        res = []
        for b in q.q_bytes:
            val = q.min_val + (b / 255.0) * diff
            res.append(val)
        return res

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
