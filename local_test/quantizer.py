"""Minimal Int8 quantization layer - drop-in replacement for vector operations"""

import numpy as np
from typing import Tuple, Dict, List

class Int8Quantizer:
    """Lightweight quantizer for search - NO changes to Bedrock KB or OpenSearch"""

    @staticmethod
    def quantize_vector(vector: List[float]) -> Tuple[np.ndarray, Dict]:
        """Convert single vector to int8 with metadata for scoring"""
        vec = np.array(vector, dtype=np.float32)

        # Store min/max for dequantization
        min_val = float(vec.min())
        max_val = float(vec.max())

        # Scale to int8 range [-128, 127]
        if max_val == min_val:
            quantized = np.zeros_like(vec, dtype=np.int8)
        else:
            normalized = (vec - min_val) / (max_val - min_val)
            scaled = normalized * 255 - 128
            quantized = np.round(scaled).astype(np.int8)

        metadata = {"min": min_val, "max": max_val}
        return quantized, metadata

    @staticmethod
    def quantize_batch(vectors: List[List[float]]) -> Tuple[List[np.ndarray], List[Dict]]:
        """Quantize multiple vectors"""
        quantized_vectors = []
        metadatas = []

        for vec in vectors:
            q_vec, meta = Int8Quantizer.quantize_vector(vec)
            quantized_vectors.append(q_vec)
            metadatas.append(meta)

        return quantized_vectors, metadatas

    @staticmethod
    def similarity_score(query_vec: np.ndarray, query_meta: Dict,
                        doc_vec: np.ndarray, doc_meta: Dict) -> float:
        """Calculate cosine similarity between int8 vectors with accurate scoring"""

        # Convert int8 back to approximate float for scoring
        q_float = (query_vec.astype(np.float32) + 128) / 255 * (query_meta["max"] - query_meta["min"]) + query_meta["min"]
        d_float = (doc_vec.astype(np.float32) + 128) / 255 * (doc_meta["max"] - doc_meta["min"]) + doc_meta["min"]

        # Cosine similarity
        norm_q = np.linalg.norm(q_float)
        norm_d = np.linalg.norm(d_float)

        if norm_q < 1e-8 or norm_d < 1e-8:
            return 0.0

        return float(np.dot(q_float, d_float) / (norm_q * norm_d))
