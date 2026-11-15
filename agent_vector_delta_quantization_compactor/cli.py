"""CLI for Vector Quantizer."""
import argparse
from .core import VectorQuantizer

def main():
    parser = argparse.ArgumentParser(description="Vector Delta Quantizer CLI")
    parser.add_argument("--demo", action="store_true", help="Demonstrate vector quantization and similarity")
    args = parser.parse_args()

    vec = [0.12, -0.45, 0.88, 0.05, -0.92]
    q = VectorQuantizer.quantize(vec)
    deq = VectorQuantizer.dequantize(q)
    sim = VectorQuantizer.cosine_similarity(vec, deq)

    print(f"Original: {vec}")
    print(f"Quantized byte length: {len(q.q_bytes)} bytes")
    print(f"Cosine Similarity (Original vs Dequantized): {sim:.5f}")

if __name__ == "__main__":
    main()
